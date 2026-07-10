"""MAP-Elites illumination of the syrinx instrument space.

Each candidate *instrument* (genome) is played by a fixed bank of gestures;
every resulting call is embedded by BirdNET, projected onto the first two PCA
axes, and dropped into the archive cell it lands in — keeping the call of
highest nonlinear-richness fitness per cell. The archive that emerges is a
navigable palette of synthetic calls spread across the perceptual manifold.

Run:  uv run python run_mapelites.py
      uv run python run_mapelites.py generations=300 offspring=32
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".7")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
import time

import hydra
import numpy as np
import jax
import jax.numpy as jnp
from omegaconf import DictConfig, OmegaConf

from callsong import genome as gm
from callsong import synth, gestures, features, audio_io
from callsong.birdnet import BirdNetEmbedder, ParallelEmbedder, DescriptorProjector
from callsong.archive import Archive

log = logging.getLogger("mapelites")


def render_calls(render, phys_rows, alpha_rows, beta_rows, noise_rows,
                 chunk=128):
    """Render (instrument, gesture) pairs -> numpy (B, T), in sub-batches.

    The scan is sequential in time but parallel across the batch, so wall time
    is ~flat up to a few hundred calls; we chunk to stay within the 4 GB card
    rather than materialising one giant vmap."""
    out = []
    n = len(phys_rows)
    for i in range(0, n, chunk):
        sl = slice(i, i + chunk)
        wave = render(jnp.asarray(phys_rows[sl]), jnp.asarray(alpha_rows[sl]),
                      jnp.asarray(beta_rows[sl]), jnp.asarray(noise_rows[sl]))
        out.append(np.asarray(wave.block_until_ready()))
    return np.concatenate(out, axis=0)


def expand_pairs(genomes, bank, rng):
    """Every instrument x every gesture. Returns phys, alpha, beta, noise and
    bookkeeping arrays (genome index, gesture id)."""
    n_inst = len(genomes)
    n_g, _, T = bank.shape
    phys = gm.decode(genomes)                       # (n_inst, P)
    phys_rows = np.repeat(phys, n_g, axis=0)         # (n_inst*n_g, P)
    alpha_rows = np.tile(bank[:, 0], (n_inst, 1))    # (n_inst*n_g, T)
    beta_rows = np.tile(bank[:, 1], (n_inst, 1))
    noise_rows = rng.uniform(-1, 1, size=(n_inst * n_g, T)).astype(np.float32)
    inst_idx = np.repeat(np.arange(n_inst), n_g)
    gest_id = np.tile(np.arange(n_g), n_inst)
    return (phys_rows.astype(np.float32), alpha_rows.astype(np.float32),
            beta_rows.astype(np.float32), noise_rows, inst_idx, gest_id)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    log.info("\n" + OmegaConf.to_yaml(cfg))
    rundir = os.getcwd()  # hydra chdir's into the run dir
    rng = np.random.default_rng(cfg.seed)
    T = int(cfg.sr * cfg.duration)
    fitw = OmegaConf.to_container(cfg.fitness)

    log.info(f"jax devices: {jax.devices()}")
    bank = gestures.make_gesture_bank(cfg.seed, cfg.n_gestures, T,
                                      cfg.gesture_ctrl, cfg.gesture_sine, cfg.sr)
    render = synth.make_renderer(cfg.sr, cfg.oversample, T)
    log.info(f"loading BirdNET (tensorflow-cpu), embed_workers={cfg.embed_workers}...")
    embedder = (ParallelEmbedder(cfg.embed_workers) if cfg.embed_workers > 1
                else BirdNetEmbedder())

    def evaluate(genomes):
        """Render + embed + score a set of instruments. Returns per-call records."""
        phys, alpha, beta, noise, inst_idx, gest_id = expand_pairs(genomes, bank, rng)
        waves = render_calls(render, phys, alpha, beta, noise, cfg.render_chunk)
        embs = embedder.embed_many(waves, cfg.sr)
        recs = []
        for k in range(len(waves)):
            fit, feats = features.compute(waves[k], cfg.sr, fitw)
            recs.append(dict(genome=genomes[inst_idx[k]], emb=embs[k],
                             gesture_id=int(gest_id[k]), fitness=fit,
                             feats=feats, wave=waves[k]))
        return recs

    # ── bootstrap: random instruments -> fit PCA + archive bounds, seed archive ─
    log.info(f"bootstrap: {cfg.bootstrap} random instruments "
             f"x {cfg.n_gestures} gestures = "
             f"{cfg.bootstrap * cfg.n_gestures} calls")
    t0 = time.time()
    boot_genomes = gm.random_genomes(rng, cfg.bootstrap)
    boot_recs = evaluate(boot_genomes)
    embs = np.stack([r["emb"] for r in boot_recs])
    if cfg.projector_path:
        projector = DescriptorProjector.load(hydra.utils.to_absolute_path(cfg.projector_path))
        log.info(f"loaded PCA projector from {cfg.projector_path}")
    else:
        projector = DescriptorProjector.fit(embs)
        log.info("fit PCA projector on bootstrap population")
    projector.save(os.path.join(rundir, "projector.npz"))
    log.info(f"PCA descriptor bounds: {projector.bounds.tolist()}")

    archive = Archive(projector.bounds, cfg.archive_resolution)
    for r in boot_recs:
        d = projector.project(r["emb"])[0]
        archive.add(r["genome"], d, r["fitness"], r["gesture_id"], r["feats"])
    log.info(f"bootstrap done in {time.time() - t0:.1f}s — "
             f"coverage {100 * archive.coverage():.1f}%  "
             f"QD {archive.qd_score():.1f}  best {archive.best()[1]:.3f}")

    # ── MAP-Elites loop ──────────────────────────────────────────────────────
    for gen in range(1, cfg.generations + 1):
        parents = archive.sample_elites(rng, cfg.offspring)
        offspring = gm.mutate(rng, parents, cfg.mutation_sigma)
        n_cx = int(cfg.crossover_prob * cfg.offspring)
        if n_cx > 0:
            mates = archive.sample_elites(rng, n_cx)
            offspring[:n_cx] = gm.mutate(
                rng, gm.crossover(rng, parents[:n_cx], mates),
                cfg.mutation_sigma)

        recs = evaluate(offspring)
        added = 0
        for r in recs:
            d = projector.project(r["emb"])[0]
            added += archive.add(r["genome"], d, r["fitness"],
                                 r["gesture_id"], r["feats"])

        if gen % cfg.log_every == 0 or gen == cfg.generations:
            log.info(f"gen {gen:4d}  +{added:2d}/{len(recs)}  "
                     f"coverage {100 * archive.coverage():5.1f}%  "
                     f"QD {archive.qd_score():7.1f}  "
                     f"best {archive.best()[1]:.3f}")
            archive.save(os.path.join(rundir, "archive.npz"))

    # ── outputs ──────────────────────────────────────────────────────────────
    archive.save(os.path.join(rundir, "archive.npz"))
    audio_io.archive_scatter_png(os.path.join(rundir, "archive_scatter.png"),
                                 archive, projector)

    # Dump the top-k calls (re-render the winning instrument+gesture).
    m = archive.occupied_mask()
    fit_flat = np.where(m, archive.fitness, -np.inf).ravel()
    top = np.argsort(fit_flat)[::-1][:cfg.save_top_k]
    top = [t for t in top if np.isfinite(fit_flat[t])]
    calldir = os.path.join(rundir, "calls")
    os.makedirs(calldir, exist_ok=True)
    top_waves, top_titles = [], []
    for rank, flat_idx in enumerate(top):
        i, j = np.unravel_index(flat_idx, archive.fitness.shape)
        genome = archive.genomes[i, j]
        gid = int(archive.gesture_id[i, j])
        phys = gm.decode(genome[None])
        noise = rng.uniform(-1, 1, size=(1, T)).astype(np.float32)
        wave = render_calls(render, phys, bank[gid, 0][None], bank[gid, 1][None], noise)[0]
        nm = f"rank{rank:02d}_cell{i}-{j}_fit{archive.fitness[i, j]:.3f}"
        audio_io.save_wav(os.path.join(calldir, nm + ".wav"), wave, cfg.sr)
        top_waves.append(wave)
        top_titles.append(nm)
    audio_io.montage_png(os.path.join(rundir, "top_calls.png"),
                         top_waves, cfg.sr, top_titles)
    log.info(f"done. coverage {100 * archive.coverage():.1f}%  "
             f"{archive.n_filled()} elites. outputs in {rundir}")


if __name__ == "__main__":
    main()
