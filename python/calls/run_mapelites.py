"""MAP-Elites illumination of the syrinx performance space.

A genome is an instrument *and* the motor gesture that plays it. Each renders
one call, which BirdNET embeds, projected onto the first two PCA axes and
dropped into the archive cell it lands in — keeping the fittest call per cell.
The archive that emerges is a navigable palette of synthetic calls spread
across the perceptual manifold.

Run:  uv run python run_mapelites.py
      uv run python run_mapelites.py generations=300 offspring=32
"""

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".7")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import logging
import signal
import sys
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


def build_batch(genomes, n_samples, sr, rng):
    """Genomes -> (instrument phys, alpha, beta, noise), one call per genome.

    Each genome carries its own CPG parameters, so the motor path is built here
    rather than drawn from a shared bank."""
    phys = gm.decode(genomes)
    inst = phys[:, gm.INSTRUMENT].astype(np.float32)
    paths = gestures.paths_from_params(phys[:, gm.GESTURE], n_samples, sr)
    noise = rng.uniform(-1, 1, size=(len(genomes), n_samples)).astype(np.float32)
    return inst, paths[:, 0], paths[:, 1], noise


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Turn SIGTERM into a normal exit so atexit runs and the BirdNET worker
    # pool is torn down; spawned workers otherwise outlive an interrupted run.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    log.info("\n" + OmegaConf.to_yaml(cfg))
    rundir = os.getcwd()  # hydra chdir's into the run dir
    rng = np.random.default_rng(cfg.seed)
    T = int(cfg.sr * cfg.duration)
    fitw = OmegaConf.to_container(cfg.fitness)

    log.info(f"jax devices: {jax.devices()}")
    log.info(f"genome: {gm.N_PARAMS} params "
             f"({gm.N_INSTRUMENT} instrument + {gm.N_GESTURE} gesture)")
    render = synth.make_renderer(cfg.sr, cfg.oversample, T)
    log.info(f"loading BirdNET (tensorflow-cpu), embed_workers={cfg.embed_workers}...")
    embedder = (ParallelEmbedder(cfg.embed_workers) if cfg.embed_workers > 1
                else BirdNetEmbedder())

    def evaluate(genomes):
        """Render + embed + score a set of genomes. Returns per-call records.

        Chunked end to end: the motor paths are as big as the audio, so
        building them all up front costs more memory than the render itself."""
        waves = []
        for i in range(0, len(genomes), cfg.render_chunk):
            sl = slice(i, i + cfg.render_chunk)
            phys, alpha, beta, noise = build_batch(genomes[sl], T, cfg.sr, rng)
            waves.append(render_calls(render, phys, alpha, beta, noise,
                                      cfg.render_chunk))
        waves = np.concatenate(waves, axis=0)
        embs = embedder.embed_many(waves, cfg.sr)
        recs = []
        for k in range(len(waves)):
            fit, feats = features.compute(waves[k], cfg.sr, fitw)
            recs.append(dict(genome=genomes[k], emb=embs[k], fitness=fit,
                             feats=feats, wave=waves[k]))
        return recs

    # ── bootstrap: random instruments -> fit PCA + archive bounds, seed archive ─
    log.info(f"bootstrap: {cfg.bootstrap} random genomes "
             f"= {cfg.bootstrap} calls")
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
        archive.add(r["genome"], d, r["fitness"], r["feats"])
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
            added += archive.add(r["genome"], d, r["fitness"], r["feats"])

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
        inst, alpha, beta, noise = build_batch(genome[None], T, cfg.sr, rng)
        wave = render_calls(render, inst, alpha, beta, noise)[0]
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
