"""Re-render the audio and plots for a saved archive.

``run_mapelites.py`` checkpoints ``archive.npz`` every ``log_every``
generations but only writes the scatter, the montage and ``calls/*.wav`` at the
very end — so a run that is interrupted keeps its search results and loses all
of its audio. This script rebuilds those outputs from any checkpoint.

    uv run python scripts/render_archive.py                     # latest run
    uv run python scripts/render_archive.py outputs/2026-.../ --top_k 32
"""

import argparse
import glob
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".7")

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callsong import genome as gm
from callsong import synth, audio_io, gestures
from callsong.archive import Archive
from callsong.birdnet import DescriptorProjector


def load_archive(path: str) -> Archive:
    z = np.load(path)
    a = Archive(z["bounds"], int(z["res"]))
    a.fitness = z["fitness"]
    a.genomes = z["genomes"]
    a.descriptors = z["descriptors"]
    return a


def spread_select(archive, mask: np.ndarray, k: int) -> list[int]:
    """Flat indices of ``k`` elites spread across the descriptor space.

    Taking the top-k by fitness is the wrong way to audition a MAP-Elites run:
    fitness is one scalar, and its extreme tail concentrates on whichever
    corner of the space maximises it — on this project's runs, two thirds of
    the top 24 came out of a single gesture with the same syllable rate. That
    throws away the diversity the archive exists to hold. Farthest-point
    sampling over the cell descriptors, seeded by the best elite, keeps the
    strongest call while making the rest as unlike each other as possible.
    """
    flat = np.flatnonzero(mask.ravel())
    d = archive.descriptors.reshape(-1, 2)[flat]
    fit = archive.fitness.ravel()[flat]

    lo, hi = np.nanmin(d, axis=0), np.nanmax(d, axis=0)
    u = (d - lo) / np.where(hi - lo > 0, hi - lo, 1.0)

    chosen = [int(np.argmax(fit))]
    dist = np.linalg.norm(u - u[chosen[0]], axis=1)
    while len(chosen) < min(k, len(flat)):
        nxt = int(np.argmax(dist))
        if dist[nxt] <= 0.0:
            break
        chosen.append(nxt)
        dist = np.minimum(dist, np.linalg.norm(u - u[nxt], axis=1))
    return [int(flat[c]) for c in chosen]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("rundir", nargs="?", default=None)
    ap.add_argument("--top_k", type=int, default=24)
    ap.add_argument("--select", choices=("spread", "top"), default="spread",
                    help="'spread' samples elites across the descriptor space "
                         "(default); 'top' takes the highest fitness")
    ap.add_argument("--sr", type=int, default=48_000)
    ap.add_argument("--oversample", type=int, default=6)
    ap.add_argument("--duration", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rundir = args.rundir or sorted(glob.glob(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "outputs", "*", "")))[-1]
    print("run:", rundir)

    archive = load_archive(os.path.join(rundir, "archive.npz"))
    if archive.genomes.shape[-1] != gm.N_PARAMS:
        raise SystemExit(
            f"{rundir} holds {archive.genomes.shape[-1]}-parameter genomes but "
            f"this genome has {gm.N_PARAMS} ({', '.join(gm.NAMES)}). The run "
            "predates a change to PARAM_SPEC and cannot be re-rendered; its "
            "wavs, if any, are still valid.")
    n_samples = int(args.sr * args.duration)

    m = archive.occupied_mask()
    print(f"{m.sum()} elites, {100 * m.mean():.1f}% coverage, "
          f"QD {archive.fitness[m].sum():.1f}")

    proj_path = os.path.join(rundir, "projector.npz")
    projector = (DescriptorProjector.load(proj_path)
                 if os.path.exists(proj_path) else None)
    audio_io.archive_scatter_png(os.path.join(rundir, "archive_scatter.png"),
                                 archive, projector)

    render = synth.make_renderer(args.sr, args.oversample, n_samples)
    rng = np.random.default_rng(args.seed)

    fit_flat = np.where(m, archive.fitness, -np.inf).ravel()
    if args.select == "top":
        top = [t for t in np.argsort(fit_flat)[::-1][:args.top_k]
               if np.isfinite(fit_flat[t])]
    else:
        top = spread_select(archive, m, args.top_k)

    calldir = os.path.join(rundir, "calls")
    os.makedirs(calldir, exist_ok=True)
    # Clear previous renders: names encode cell and fitness, so a re-render with
    # a different selection would otherwise sit alongside the old set rather
    # than replace it, and you would audition both without noticing.
    stale = glob.glob(os.path.join(calldir, "rank*.wav"))
    for p in stale:
        os.remove(p)
    if stale:
        print(f"cleared {len(stale)} previous renders")
    waves, titles = [], []
    for rank, flat_idx in enumerate(top):
        i, j = np.unravel_index(flat_idx, archive.fitness.shape)
        phys = gm.decode(archive.genomes[i, j][None])
        inst = phys[:, gm.INSTRUMENT].astype(np.float32)
        paths = gestures.paths_from_params(phys[:, gm.GESTURE], n_samples, args.sr)
        noise = rng.uniform(-1, 1, size=(1, n_samples)).astype(np.float32)
        wave = np.asarray(render(jnp.asarray(inst), jnp.asarray(paths[:, 0]),
                                 jnp.asarray(paths[:, 1]),
                                 jnp.asarray(noise)))[0]
        nm = f"rank{rank:02d}_cell{i}-{j}_fit{archive.fitness[i, j]:.3f}"
        audio_io.save_wav(os.path.join(calldir, nm + ".wav"), wave, args.sr)
        waves.append(wave)
        titles.append(nm)
    audio_io.montage_png(os.path.join(rundir, "top_calls.png"), waves,
                         args.sr, titles)
    print(f"wrote {len(waves)} calls + archive_scatter.png + top_calls.png "
          f"to {rundir}")


if __name__ == "__main__":
    main()
