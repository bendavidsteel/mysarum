"""
render_floraform.py — Grow and render a floraform to an image.

Usage:
    uv run python render_floraform.py
    uv run python render_floraform.py --frames 500 --resolution 1024
    uv run python render_floraform.py --seed 42 --growth-rate 5.0 --bending 0.1
    uv run python render_floraform.py --growth-mode nca
"""

import argparse
import logging
import os
import sys
import time

# Disable JAX's 75% VRAM preallocation: the top-k refine_mesh pipeline needs
# more XLA scratch than fits if JAX has already claimed most of the GPU on a
# 4GB card (GTX 1650 Max-Q). On-demand allocation costs a bit on first use
# but avoids CUDA_ERROR_OUT_OF_MEMORY for the rest of the render.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp

import growth_halfedge_jax as g

# Single compiled graph per (substeps, growth_mode) pair instead of relying on
# inner-op JIT caching. n_substeps and growth_mode are static so the substep
# scan and the lenia/nca branch trace once.
_jit_physics_step = jax.jit(
    g.batched_physics_step,
    static_argnames=("n_substeps", "growth_mode", "fix_boundary",
                     "pin_ground_cap"),
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger("floraform")


def parse_args():
    p = argparse.ArgumentParser(description="Grow and render a floraform.")

    # Simulation
    p.add_argument("--frames", type=int, default=300, help="Simulation frames")
    p.add_argument("--seed", type=int, default=7, help="RNG seed")
    p.add_argument("--shape", choices=["disc", "sphere"], default="disc",
                   help="Initial mesh topology")
    p.add_argument("--rings", type=int, default=4, help="Initial disc rings")
    p.add_argument("--sphere-lat", type=int, default=8,
                   help="UV-sphere latitude bands (sphere only)")
    p.add_argument("--sphere-lon", type=int, default=12,
                   help="UV-sphere longitude segments (sphere only)")
    p.add_argument("--sphere-radius-mult", type=float, default=2.0,
                   help="Sphere radius as multiple of spring_len (sphere only)")
    p.add_argument("--growth-mode", choices=["lenia", "nca"], default="lenia",
                   help="Growth dynamics for vertex_state")

    # XPBD Physics
    p.add_argument("--spring-len", type=float, default=30.0)
    p.add_argument("--compliance", type=float, default=0.0)
    p.add_argument("--bending-compliance", type=float, default=0.001)
    p.add_argument("--repulsion-dist", type=float, default=80.0)
    p.add_argument("--repulsion-strength", type=float, default=4.0)
    p.add_argument("--bulge-strength", type=float, default=5.0)
    p.add_argument("--stiffness", type=float, default=400.0)
    p.add_argument("--damping", type=float, default=10.0)
    p.add_argument("--dt", type=float, default=0.02)
    p.add_argument("--max-edge-len", type=float, default=50.0)

    # Growth / CA
    p.add_argument("--growth-rate", type=float, default=1.0)
    p.add_argument("--state-dt", type=float, default=0.02)
    p.add_argument("--kernel-mu", type=float, default=4.0)
    p.add_argument("--kernel-sigma", type=float, default=1.0)
    p.add_argument("--growth-mu", type=float, default=0.5)
    p.add_argument("--growth-sigma", type=float, default=0.2)

    # MeshNCA
    p.add_argument("--state-dims", type=int, default=8,
                   help="vertex_state channels (NCA mode)")
    p.add_argument("--hidden-dims", type=int, default=32)
    p.add_argument("--mlp-layers", type=int, default=2)

    # Anisotropic growth (orthogonal to growth mode; strength 0 = isotropic)
    p.add_argument("--anisotropy-dir", type=float, nargs=3,
                   metavar=("X", "Y", "Z"), default=[0.0, 0.0, 1.0],
                   help="Preferred-growth axis (arbitrary vec3, normalized internally)")
    p.add_argument("--anisotropy-strength", type=float, default=0.0,
                   help="0 = isotropic, 1 = grow only along --anisotropy-dir")

    # Rendering
    p.add_argument("--resolution", type=int, default=1024, help="Image resolution")
    p.add_argument("-o", "--output", type=str, default="floraform_render.png")

    # Topology & batching
    p.add_argument("--substeps", type=int, default=10,
                   help="Physics substeps per topology update (batched in one JIT call)")
    p.add_argument("--max-splits", type=int, default=15,
                   help="Max edge splits per topology update")

    return p.parse_args()


def progress_bar(current, total, width=40, extra=""):
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    pct = frac * 100
    line = f"\r  [{bar}] {pct:5.1f}% ({current}/{total}) {extra}"
    sys.stderr.write(line)
    sys.stderr.flush()


def main():
    args = parse_args()

    key = jax.random.PRNGKey(args.seed)

    if args.growth_mode == "lenia":
        state_dims = 1
        mlp_params = {}
    else:
        state_dims = args.state_dims
        key, mlp_key = jax.random.split(key)
        mlp_params = g.make_mlp_params(
            mlp_key, state_dims,
            hidden_state_dims=args.hidden_dims,
            num_mlp_layers=args.mlp_layers,
        )

    params = g.default_params(
        spring_len=args.spring_len,
        compliance=args.compliance,
        bending_compliance=args.bending_compliance,
        repulsion_dist=args.repulsion_dist,
        repulsion_strength=args.repulsion_strength,
        bulge_strength=args.bulge_strength,
        stiffness=args.stiffness,
        damping=args.damping,
        dt=args.dt,
        max_edge_len=args.max_edge_len,
        growth_rate=args.growth_rate,
        state_dt=args.state_dt,
        kernel_mu=args.kernel_mu,
        kernel_sigma=args.kernel_sigma,
        growth_mu=args.growth_mu,
        growth_sigma=args.growth_sigma,
        vertex_state_mlp_params=mlp_params,
        anisotropy_dir=jnp.array(args.anisotropy_dir, dtype=jnp.float32),
        anisotropy_strength=args.anisotropy_strength,
    )

    if args.shape == "disc":
        log.info("Building initial disc (n_rings=%d, spring_len=%.1f)",
                 args.rings, params.spring_len)
        state = g.make_disc(
            n_rings=args.rings,
            spring_len=params.spring_len,
            center=(args.resolution / 2, args.resolution / 2, 0.0),
            state_dims=state_dims,
        )
        state = g.seed_boundary_state(state, channel=0, value=1.0)
    else:
        log.info(
            "Building initial sphere (n_lat=%d, n_lon=%d, radius=%.1f)",
            args.sphere_lat, args.sphere_lon,
            params.spring_len * args.sphere_radius_mult,
        )
        state = g.make_sphere(
            width=args.resolution, height=args.resolution, params=params,
            n_lat=args.sphere_lat, n_lon=args.sphere_lon,
            radius_multiplier=args.sphere_radius_mult,
            state_dims=state_dims,
        )
        key, seed_key = jax.random.split(key)
        state = g.seed_random_state(state, seed_key, channel=0, low=0.0, high=1.0)

    cheb_coeffs, order = g.chebyshev_ring_coeffs(
        params.kernel_mu, params.kernel_sigma
    )

    nv = int(jnp.sum(state.vertex_idx != -1))
    log.info(
        "seed=%d  verts=%d  cheb_order=%d  mode=%s  frames=%d",
        args.seed, nv, order, args.growth_mode, args.frames,
    )
    log.info(
        "compliance=%.3f  bending_compliance=%.4f  stiffness=%.0f  "
        "growth_rate=%.1f  max_edge_len=%.2f",
        params.compliance, params.bending_compliance, params.stiffness,
        params.growth_rate, params.max_edge_len,
    )

    # ── Warm-up JIT ──
    # Compile physics graph and prime the host-side topology ops on the
    # initial mesh so iter 0 doesn't pay first-call costs.
    log.info("JIT compiling batched_physics_step (substeps=%d)...", args.substeps)
    t_jit = time.time()

    state, key = _jit_physics_step(
        state, params, cheb_coeffs, key, args.substeps, args.growth_mode,
    )
    jax.block_until_ready(state.vertex_pos)
    state, key, nv = g.split_long_edges(state, params, key, max_splits=args.max_splits)
    state = g.refine_mesh(state)

    log.info("JIT done in %.1fs", time.time() - t_jit)

    # ── Simulation loop ──
    # Each "iteration" = substeps physics frames + 1 topology update.
    n_iters = args.frames // args.substeps
    total_frames = n_iters * args.substeps
    log.info("Running %d iterations x %d substeps = %d physics frames...",
             n_iters, args.substeps, total_frames)
    t0 = time.time()

    for it in range(n_iters):
        state, key = _jit_physics_step(
            state, params, cheb_coeffs, key, args.substeps, args.growth_mode,
        )

        # split_long_edges already counts active verts internally (host side)
        # and returns nv — reuse it for progress + limit check, no extra sync.
        state, key, nv = g.split_long_edges(
            state, params, key, max_splits=args.max_splits,
        )
        state = g.refine_mesh(state)

        frame = (it + 1) * args.substeps
        elapsed = time.time() - t0
        fps = frame / max(elapsed, 0.01)
        eta = (total_frames - frame) / max(fps, 0.01)
        progress_bar(
            frame, total_frames,
            extra=f"{nv} verts  {fps:.1f} fps  ETA {eta:.0f}s",
        )

        if nv >= g.MAX_VERTICES - 200:
            sys.stderr.write("\n")
            log.warning("Vertex limit reached at frame %d", frame)
            break

    sys.stderr.write("\n")
    nf = int(jnp.sum(state.face_idx != -1))
    elapsed = time.time() - t0
    log.info("Simulation done: %d verts, %d faces in %.1fs", nv, nf, elapsed)

    # ── Render ──
    log.info("Rendering %dx%d...", args.resolution, args.resolution)
    renderer = g.JAXNvdiffrastRenderer(args.resolution, args.resolution)
    image = renderer.render(state)

    from PIL import Image
    img = Image.fromarray(image)
    img.save(args.output)
    log.info("Saved to %s", args.output)


if __name__ == "__main__":
    main()
