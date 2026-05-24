"""
nca_sweep.py — unified growth sweep + diversity scoring + curation.

Single hydra-driven entry point. Sweep config under `conf/sweep/<name>.yaml`
selects which set of configs to run. Each config picks `growth_mode`:
    nca / nca_bipolar    untrained MLP rules
    phototropic          deterministic light + resources + gravity, no MLP
    lenia                Chebyshev graph CA (single-channel)

Each sweep produces:
    out_dir/imgs/*.png    rendered tiles
    out_dir/features.npz  shape-descriptor matrix + tags
    out_dir/index.json    per-run metadata
    out_dir/gallery.png   diversity-picked subset of N tiles

If `compare_to=/path/to/features.npz` is passed, also prints Vendi-score
comparison against that previous sweep.

Example:
    conda run -n base python nca_sweep.py
    conda run -n base python nca_sweep.py sweep=phototropic
    conda run -n base python nca_sweep.py sweep=bipolar compare_to=outputs/.../features.npz
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image, ImageDraw, ImageFont

import growth_halfedge_jax as g
import nca_shape_descriptors as desc


log = logging.getLogger("nca-sweep")

_jit_physics_step = jax.jit(
    g.batched_physics_step,
    static_argnames=("n_substeps", "growth_mode"),
)


# ── Seeding helpers ─────────────────────────────────────────────────────────

def _active(state):
    return state.vertex_idx != -1


def _seed_multi(state, key, n_points, channel=0, value=1.0):
    idx = jnp.where(_active(state), size=g.MAX_VERTICES, fill_value=-1)[0]
    n_active = jnp.sum(_active(state))
    chosen = jax.random.permutation(key, n_active)[:n_points]
    picks = idx[chosen]
    return state._replace(vertex_state=state.vertex_state.at[picks, channel].set(value))


def _seed_gradient(state, channel=0, axis=2, low=-0.1, high=1.0):
    active = _active(state)
    coord = state.vertex_pos[:, axis]
    cmin = jnp.min(jnp.where(active, coord, jnp.inf))
    cmax = jnp.max(jnp.where(active, coord, -jnp.inf))
    norm = (state.vertex_pos[:, axis] - cmin) / (cmax - cmin + 1e-9)
    val = jnp.where(active, low + (high - low) * norm, state.vertex_state[:, channel])
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(val))


def _seed_ring(state, channel=0, value=1.0):
    active = _active(state)
    z = state.vertex_pos[:, 2]
    zmin = jnp.min(jnp.where(active, z, jnp.inf))
    zmax = jnp.max(jnp.where(active, z, -jnp.inf))
    zmid = 0.5 * (zmin + zmax)
    band = 0.15 * (zmax - zmin + 1e-6)
    on_ring = active & (jnp.abs(z - zmid) < band)
    vs = jnp.where(on_ring, value, state.vertex_state[:, channel])
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(vs))


def _seed_antipodal(state, channel=0, value=1.0):
    idx = jnp.where(_active(state), size=g.MAX_VERTICES, fill_value=-1)[0]
    return state._replace(
        vertex_state=state.vertex_state.at[idx[0], channel].set(value),
    )


def _seed_apex(state, channel=0, value=1.0):
    """Set channel to value on the single highest-z vertex (the dome apex)."""
    active = _active(state)
    z = jnp.where(active, state.vertex_pos[:, 2], -jnp.inf)
    apex = jnp.argmax(z)
    return state._replace(
        vertex_state=state.vertex_state.at[apex, channel].set(value),
    )


def _seed_top_ring(state, channel=0, value=0.5, top_frac=0.2):
    """Set channel to value on the top `top_frac` of vertices by z."""
    active = _active(state)
    z = jnp.where(active, state.vertex_pos[:, 2], -jnp.inf)
    zmax = jnp.max(z)
    zmin = jnp.min(jnp.where(active, state.vertex_pos[:, 2], jnp.inf))
    cutoff = zmax - top_frac * (zmax - zmin)
    on_top = active & (state.vertex_pos[:, 2] >= cutoff)
    new_ch = jnp.where(on_top, value, state.vertex_state[:, channel])
    return state._replace(vertex_state=state.vertex_state.at[:, channel].set(new_ch))


def apply_seed(state, key, pattern, state_dims, mode="nca"):
    if pattern == "boundary":
        state = g.seed_boundary_state(state, channel=0, value=1.0)
    elif pattern == "random":
        key, sk = jax.random.split(key)
        state = g.seed_random_state(state, sk, channel=0, low=0.0, high=1.0)
    elif pattern == "antipodal":
        state = _seed_antipodal(state)
    elif pattern == "multi3":
        key, sk = jax.random.split(key)
        state = _seed_multi(state, sk, 3)
    elif pattern == "multi7":
        key, sk = jax.random.split(key)
        state = _seed_multi(state, sk, 7)
    elif pattern == "gradient_z":
        state = _seed_gradient(state)
    elif pattern == "ring":
        state = _seed_ring(state)
    elif pattern == "apex":
        state = _seed_apex(state, channel=0, value=0.5)
    elif pattern == "top_ring":
        state = _seed_top_ring(state, channel=0, value=0.3)
    elif pattern == "uniform_low":
        active = _active(state)
        vs = jnp.where(active, 0.1, state.vertex_state[:, 0])
        state = state._replace(
            vertex_state=state.vertex_state.at[:, 0].set(vs),
        )
    else:
        raise ValueError(f"Unknown seed_pattern: {pattern}")

    if mode == "phototropic":
        if state_dims >= 3:
            active = _active(state)
            ch2 = jnp.where(active, 1.0, state.vertex_state[:, 2])
            state = state._replace(
                vertex_state=state.vertex_state.at[:, 2].set(ch2),
            )
    else:
        for c in range(1, state_dims):
            key, sk = jax.random.split(key)
            state = g.seed_random_state(state, sk, channel=c, low=-0.3, high=0.3)
    return state


# ── Per-config defaults (merged onto sweep entries) ─────────────────────────

CONFIG_DEFAULTS = dict(
    mlp_seed=101, shape="disc", seed_pattern="boundary",
    mlp_layers=2, hidden_dims=32, state_dims=8,
    mlp_scale=0.5, state_dt=0.07, growth_rate=5.0,
    growth_mode="nca",
    rings=4, sphere_lat=6, sphere_lon=10, sphere_radius_mult=2.0,
    hemi_lat=4, hemi_lon=10, hemi_radius_mult=1.5,
    light_dir=[0.0, 0.0, 1.0],
    light_pos=[400.0, 400.0, 1500.0],
    light_decay_dist=1e6,
    light_ambient=0.05,
    gravity_strength=0.0,
    tissue_decay=0.05,
    tissue_saturation=2.0,
    resource_diffusion=0.5,
    resource_decay=0.02,
    ground_source_value=1.0,
    ground_z=0.0,
)


def _build_initial(cfg, params, resolution):
    if cfg["shape"] == "disc":
        return g.make_disc(
            n_rings=cfg["rings"],
            spring_len=params.spring_len,
            center=(resolution / 2, resolution / 2, 0.0),
            state_dims=cfg["state_dims"],
        )
    if cfg["shape"] == "hemisphere":
        return g.make_hemisphere(
            width=resolution, height=resolution, params=params,
            n_lat=cfg["hemi_lat"], n_lon=cfg["hemi_lon"],
            radius_multiplier=cfg["hemi_radius_mult"],
            state_dims=cfg["state_dims"],
            ground_z=cfg["ground_z"],
        )
    return g.make_sphere(
        width=resolution, height=resolution, params=params,
        n_lat=cfg["sphere_lat"], n_lon=cfg["sphere_lon"],
        radius_multiplier=cfg["sphere_radius_mult"],
        state_dims=cfg["state_dims"],
    )


def _resolve_light_pos(light_pos_cfg, resolution):
    lp = list(light_pos_cfg)
    if lp[0] is None or (isinstance(lp[0], float) and lp[0] < 0):
        lp[0] = resolution / 2
    if lp[1] is None or (isinstance(lp[1], float) and lp[1] < 0):
        lp[1] = resolution / 2
    return jnp.array(lp, dtype=jnp.float32)


def run_one(cfg, frames, substeps, resolution, max_edge_len, max_splits, renderer):
    key = jax.random.PRNGKey(cfg["mlp_seed"])
    key, mlp_key = jax.random.split(key)
    mlp_params = g.make_mlp_params(
        mlp_key, cfg["state_dims"],
        hidden_state_dims=cfg["hidden_dims"],
        num_mlp_layers=cfg["mlp_layers"],
        scale=cfg["mlp_scale"],
    )
    light_dir = jnp.array(list(cfg["light_dir"]), dtype=jnp.float32)
    light_pos = _resolve_light_pos(cfg["light_pos"], resolution)
    params = g.default_params(
        max_edge_len=max_edge_len,
        growth_rate=cfg["growth_rate"],
        state_dt=cfg["state_dt"],
        vertex_state_mlp_params=mlp_params,
        light_dir=light_dir,
        light_pos=light_pos,
        light_decay_dist=float(cfg["light_decay_dist"]),
        light_ambient=float(cfg["light_ambient"]),
        gravity_strength=float(cfg["gravity_strength"]),
        tissue_decay=float(cfg["tissue_decay"]),
        tissue_saturation=float(cfg["tissue_saturation"]),
        resource_diffusion=float(cfg["resource_diffusion"]),
        resource_decay=float(cfg["resource_decay"]),
        ground_z=float(cfg["ground_z"]),
        ground_source_value=float(cfg["ground_source_value"]),
    )
    state = _build_initial(cfg, params, resolution)
    key, sk = jax.random.split(key)
    state = apply_seed(
        state, sk, cfg["seed_pattern"], cfg["state_dims"],
        mode=cfg["growth_mode"],
    )
    cheb_coeffs, _ = g.chebyshev_ring_coeffs(params.kernel_mu, params.kernel_sigma)

    n_iters = frames // substeps
    nv = int(jnp.sum(_active(state)))
    growth = [nv]
    t0 = time.time()
    for _ in range(n_iters):
        state, key = _jit_physics_step(
            state, params, cheb_coeffs, key, substeps, cfg["growth_mode"],
        )
        state, key, nv = g.split_long_edges(state, params, key, max_splits=max_splits)
        state = g.refine_mesh(state)
        growth.append(int(nv))
        if int(nv) >= g.MAX_VERTICES - 200:
            break
    elapsed = time.time() - t0
    has_nan = bool(jnp.any(jnp.isnan(state.vertex_pos)))

    img = renderer.render(state) if not has_nan else None
    descriptors = None
    if not has_nan and int(nv) > 30:
        verts_j, faces_j, n_active = g.extract_mesh_for_nvdiffrast(state)
        n_active = int(n_active)
        faces_np = np.asarray(faces_j)[:n_active]
        used = np.unique(faces_np)
        remap = -np.ones(int(verts_j.shape[0]), np.int64)
        remap[used] = np.arange(len(used))
        verts_np = np.asarray(verts_j)[used].astype(np.float64)
        faces_np = remap[faces_np]
        try:
            descriptors = desc.mesh_descriptors(verts_np, faces_np, k_spectrum=15)
        except Exception as e:
            log.warning("descriptor failure: %s", e)

    return dict(
        cfg=cfg, image=img, descriptors=descriptors,
        growth_curve=growth, elapsed=elapsed, has_nan=has_nan,
        n_verts=int(nv), ok=(not has_nan and descriptors is not None),
    )


# ── Tile labeling ────────────────────────────────────────────────────────────

def _label_image(img_arr, lines):
    img = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13,
        )
    except OSError:
        font = ImageFont.load_default()
    pad = 4
    w = max(draw.textbbox((0, 0), ln, font=font)[2] for ln in lines) + 2 * pad
    h = sum(draw.textbbox((0, 0), ln, font=font)[3] for ln in lines) + 2 * pad
    draw.rectangle((0, 0, w, h), fill=(0, 0, 0))
    y = pad
    for ln in lines:
        draw.text((pad, y), ln, fill=(240, 240, 240), font=font)
        y += draw.textbbox((0, 0), ln, font=font)[3]
    return img


def _make_sheet(tiles, cols, out_path):
    rows = (len(tiles) + cols - 1) // cols
    tw, th = tiles[0].size
    sheet = Image.new("RGB", (cols * tw, rows * th), (20, 20, 20))
    for i, tile in enumerate(tiles):
        rr, cc = divmod(i, cols)
        sheet.paste(tile, (cc * tw, rr * th))
    sheet.save(out_path)


def _tag(cfg):
    return f"s{cfg['mlp_seed']:04d}_{cfg['shape']}_{cfg['seed_pattern']}_L{cfg['mlp_layers']}H{cfg['hidden_dims']}S{cfg['state_dims']}_dt{cfg['state_dt']:.2f}_gr{cfg['growth_rate']:.1f}"


# ── Hydra entry ──────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "imgs")
    os.makedirs(img_dir, exist_ok=True)

    with open(os.path.join(out_dir, "resolved_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    renderer = g.JAXNvdiffrastRenderer(cfg.resolution, cfg.resolution)

    raw_entries = OmegaConf.to_container(cfg.sweep.configs, resolve=True)
    entries = []
    for raw in raw_entries:
        merged = {**CONFIG_DEFAULTS, **raw}
        entries.append(merged)

    log.info("Sweep '%s': %d configs, frames=%d res=%d",
             cfg.sweep.name, len(entries), cfg.frames, cfg.resolution)

    features = []
    tags = []
    index = []
    n_skip = 0
    for i, entry in enumerate(entries):
        tag = f"{i:03d}_{_tag(entry)}"
        log.info("[%d/%d] %s", i + 1, len(entries), tag)
        try:
            r = run_one(
                entry, cfg.frames, cfg.substeps, cfg.resolution,
                cfg.max_edge_len, cfg.max_splits, renderer,
            )
        except Exception as e:
            log.warning("  FAILED: %s", e)
            n_skip += 1
            continue
        if not r["ok"]:
            log.info("  skip (nan=%s nv=%d)", r["has_nan"], r["n_verts"])
            n_skip += 1
            continue

        img_path = os.path.join(img_dir, f"{tag}.png")
        Image.fromarray(r["image"]).save(img_path)
        features.append(r["descriptors"]["features"])
        tags.append(tag)
        index.append({
            **entry, "tag": tag, "img": img_path,
            "n_verts": r["n_verts"], "elapsed": r["elapsed"],
            "growth_curve": r["growth_curve"],
        })
        log.info("  ok nv=%d %.1fs", r["n_verts"], r["elapsed"])

    if not features:
        log.error("No successful runs.")
        return

    X = np.stack(features, 0)
    np.savez(os.path.join(out_dir, "features.npz"),
             X=X, tags=np.array(tags))
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    vs_full = desc.vendi_score(X)
    log.info("=" * 60)
    log.info("Sweep '%s' Vendi score (%d items): %.3f",
             cfg.sweep.name, len(X), vs_full)

    n_pick = min(cfg.n_pick, len(X))
    picks = desc.diversity_pick(X, n_pick)
    vs_sub = desc.vendi_score(X[picks])
    log.info("Top-%d diversity picks Vendi: %.3f", n_pick, vs_sub)

    pick_tiles = []
    for idx in picks:
        info = index[idx]
        img = np.asarray(Image.open(info["img"]).convert("RGB"))
        lines = [
            f"s{info['mlp_seed']:04d} {info['shape']}/{info['seed_pattern']}",
            f"L{info['mlp_layers']} H{info['hidden_dims']} S{info['state_dims']}",
            f"dt={info['state_dt']:.2f} gr={info['growth_rate']:.1f}",
            f"nv={info['n_verts']}",
        ]
        pick_tiles.append(_label_image(img, lines))
    _make_sheet(pick_tiles, cols=4,
                out_path=os.path.join(out_dir, "gallery.png"))
    log.info("Saved gallery to %s/gallery.png", out_dir)

    if cfg.compare_to:
        d = np.load(cfg.compare_to, allow_pickle=True)
        X_prev = d["X"]
        vs_prev = desc.vendi_score(X_prev)
        log.info("=" * 60)
        log.info("Comparison:")
        log.info("  this sweep ('%s', n=%d): Vendi=%.3f",
                 cfg.sweep.name, len(X), vs_full)
        log.info("  this top-%d picks:           Vendi=%.3f", n_pick, vs_sub)
        log.info("  previous (%s, n=%d):         Vendi=%.3f",
                 cfg.compare_to, len(X_prev), vs_prev)

    log.info("Done: %d ok, %d skipped", len(X), n_skip)


if __name__ == "__main__":
    main()
