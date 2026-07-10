"""map_elites.py — MAP-Elites diversity search over floraform growth.

Illuminates a 2D archive of shapes: the grid axes are the *behaviour
descriptors* (shape diversity) and each cell keeps the single most **printable**
genome that lands in it (the fitness). So the archive is a diverse gallery of
shapes, each represented by its most print-friendly variant.

  archive axes : compactness (surface-area / volume^2/3)  ×  print height (z)
  genotype     : ~18 continuous environmental genes + a normal-displacement
                 growth mix (feature 5). Fixed: phototropic mode, hemisphere,
                 state_dims=5 (Gray-Scott morphogens on).
  fitness      : -printability_penalty  (weighted overhang + layers +
                 support-volume + surface-area; lower penalty = fitter)

The mesh always starts as a hemisphere flat-side-down, so build direction is
+z and "overhang" = downward-facing surface beyond the self-support angle.

Run:
    GROWTH_MAX_VERTICES=4000 conda run -n base python map_elites.py
    conda run -n base python map_elites.py n_evals=400 sigma=0.1
Resume (re-seed the archive from a previous run and keep searching):
    conda run -n base python map_elites.py resume_from=outputs/<run>/archive.npz
"""

import json
import logging
import os
import sys

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("GROWTH_MAX_VERTICES", "4000")

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from PIL import Image

import growth_halfedge_jax as g
import nca_sweep as sweep
import printability as pr


log = logging.getLogger("map-elites")


# ── Genotype ─────────────────────────────────────────────────────────────────
# Each gene is (name, low, high, log?). The normalised genome lives in [0,1]^G;
# decode() maps it to real values, then gene_overlay() turns those into a cfg
# overlay merged onto sweep.CONFIG_DEFAULTS.

GENES = [
    ("light_theta", 0.0, 1.3, False),        # polar angle of light from +z
    ("light_phi", 0.0, 2 * np.pi, False),     # azimuth
    ("gravity_strength", 0.0, 80.0, False),
    ("resource_diffusion", 0.1, 0.9, False),
    ("resource_decay", 0.005, 0.2, True),
    ("tissue_decay", 0.01, 0.10, True),   # capped: high decay stalls growth
    ("growth_rate", 3.0, 9.0, False),     # size lever (with state_dt)
    ("state_dt", 0.05, 0.09, False),      # per-substep growth step (size lever)
    ("occlusion_strength", 0.0, 3.0, False),
    ("growth_budget_gate", 0.0, 1.0, False),
    ("growth_field_smooth", 0.0, 0.6, False),
    ("growth_mode_mix", 0.0, 1.0, False),
    ("inflation_strength", 10.0, 60.0, False),
    ("morphogen_coupling", 0.0, 0.8, False),
    ("morphogen_feed", 0.02, 0.062, False),
    ("morphogen_kill", 0.05, 0.07, False),
    ("aniso_strength", 0.0, 1.0, False),
    ("aniso_theta", 0.0, np.pi, False),
    ("aniso_phi", 0.0, 2 * np.pi, False),
]
G_DIM = len(GENES)


def decode(genome):
    """Normalised genome ([0,1]^G) → dict of real gene values."""
    out = {}
    for gv, (name, lo, hi, is_log) in zip(genome, GENES):
        if is_log:
            out[name] = float(np.exp(np.log(lo) + gv * (np.log(hi) - np.log(lo))))
        else:
            out[name] = float(lo + gv * (hi - lo))
    return out


def _dir_from_angles(theta, phi):
    return [float(np.sin(theta) * np.cos(phi)),
            float(np.sin(theta) * np.sin(phi)),
            float(np.cos(theta))]


def gene_overlay(vals, fixed):
    """Real gene values → cfg overlay (merged onto CONFIG_DEFAULTS + fixed)."""
    overlay = dict(fixed)
    overlay.update(
        light_dir=_dir_from_angles(vals["light_theta"], vals["light_phi"]),
        gravity_strength=vals["gravity_strength"],
        resource_diffusion=vals["resource_diffusion"],
        resource_decay=vals["resource_decay"],
        tissue_decay=vals["tissue_decay"],
        growth_rate=vals["growth_rate"],
        state_dt=vals["state_dt"],
        occlusion_strength=vals["occlusion_strength"],
        growth_budget_gate=vals["growth_budget_gate"],
        growth_field_smooth=vals["growth_field_smooth"],
        growth_mode_mix=vals["growth_mode_mix"],
        inflation_strength=vals["inflation_strength"],
        morphogen_coupling=vals["morphogen_coupling"],
        morphogen_feed=vals["morphogen_feed"],
        morphogen_kill=vals["morphogen_kill"],
        anisotropy_strength=vals["aniso_strength"],
        anisotropy_dir=_dir_from_angles(vals["aniso_theta"], vals["aniso_phi"]),
        occlusion_radius=0.0,
        occlusion_cone=0.4,
    )
    return overlay


# ── Archive ──────────────────────────────────────────────────────────────────

class Archive:
    def __init__(self, bins_c, bins_h, c_range, h_range):
        self.bins_c, self.bins_h = bins_c, bins_h
        self.c_range, self.h_range = c_range, h_range
        self.fitness = np.full((bins_c, bins_h), -np.inf)
        self.genomes = np.zeros((bins_c, bins_h, G_DIM))
        self.metrics = {}          # (i, j) -> metrics dict
        self.images = {}           # (i, j) -> uint8 image

    def cell(self, compactness, height):
        i = np.clip(int((compactness - self.c_range[0])
                        / (self.c_range[1] - self.c_range[0]) * self.bins_c),
                    0, self.bins_c - 1)
        j = np.clip(int((height - self.h_range[0])
                        / (self.h_range[1] - self.h_range[0]) * self.bins_h),
                    0, self.bins_h - 1)
        return int(i), int(j)

    def add(self, genome, fitness, compactness, height, metrics, image):
        i, j = self.cell(compactness, height)
        if fitness > self.fitness[i, j]:
            improved = np.isneginf(self.fitness[i, j])
            self.fitness[i, j] = fitness
            self.genomes[i, j] = genome
            self.metrics[(i, j)] = metrics
            if image is not None:
                self.images[(i, j)] = image
            return "new" if improved else "improved"
        return "kept"

    @property
    def n_filled(self):
        return int(np.isfinite(self.fitness).sum())

    def occupied_genomes(self):
        mask = np.isfinite(self.fitness)
        return self.genomes[mask]

    def save(self, path):
        """OOM-safe: write to a temp file then atomically replace. Passing a
        file handle to np.savez avoids the implicit `.npz` suffix rename."""
        tmp = path + ".tmp"
        with open(tmp, "wb") as fh:
            np.savez(fh, fitness=self.fitness, genomes=self.genomes,
                     bins_c=self.bins_c, bins_h=self.bins_h,
                     c_range=np.array(self.c_range),
                     h_range=np.array(self.h_range),
                     gene_names=np.array([n for n, *_ in GENES]))
        os.replace(tmp, path)

    def load_genomes(self, path):
        d = np.load(path, allow_pickle=True)
        fit = d["fitness"]
        mask = np.isfinite(fit)
        return d["genomes"][mask]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(genome, fixed, sim, weights, renderer):
    """Simulate one genome; return (fitness, compactness, height, metrics, img)
    or None if the growth failed (NaN / too few verts)."""
    overlay = gene_overlay(decode(genome), fixed)
    cfg = {**sweep.CONFIG_DEFAULTS, **overlay}
    r = sweep.run_one(
        cfg, sim["frames"], sim["substeps"], sim["resolution"],
        sim["max_edge_len"], sim["max_splits"], renderer,
        do_descriptors=False, do_render=True,
    )
    if not r["ok"] or r["mesh"] is None:
        return None
    verts, faces = r["mesh"]
    m = pr.print_metrics(verts, faces,
                         overhang_angle_deg=sim["overhang_angle_deg"],
                         layer_height=sim["layer_height"])
    penalty, _ = pr.printability_penalty(m, weights)
    return (-penalty, m["compactness"], m["height"], m,
            r["image"], r["n_verts"])


# ── Montage ──────────────────────────────────────────────────────────────────

def save_montage(archive, path, tile=96):
    nc, nh = archive.bins_c, archive.bins_h
    sheet = Image.new("RGB", (nc * tile, nh * tile), (18, 18, 18))
    for (i, j), img in archive.images.items():
        im = Image.fromarray(img).resize((tile, tile))
        # compactness → column (left=compact), height → row (bottom=tall)
        sheet.paste(im, (i * tile, (nh - 1 - j) * tile))
    sheet.save(path)


# ── Hydra entry ──────────────────────────────────────────────────────────────

@hydra.main(version_base="1.3", config_path="conf", config_name="map_elites")
def main(cfg: DictConfig):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", stream=sys.stderr)
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "resolved_config.yaml"), "w") as f:
        OmegaConf.save(cfg, f, resolve=True)

    rng = np.random.default_rng(cfg.seed)
    renderer = g.JAXNvdiffrastRenderer(cfg.resolution, cfg.resolution)

    fixed = dict(
        growth_mode="phototropic", shape="hemisphere", state_dims=5,
        seed_pattern="apex", mlp_seed=int(cfg.mlp_seed),
        hemi_lat=int(cfg.hemi_lat), hemi_lon=int(cfg.hemi_lon),
        hemi_radius_mult=float(cfg.hemi_radius_mult),
        hemi_closed=bool(cfg.hemi_closed),
        morphogen_steps=int(cfg.morphogen_steps),
    )
    sim = dict(
        frames=int(cfg.frames), substeps=int(cfg.substeps),
        resolution=int(cfg.resolution), max_edge_len=float(cfg.max_edge_len),
        max_splits=int(cfg.max_splits),
        overhang_angle_deg=float(cfg.overhang_angle_deg),
        layer_height=float(cfg.layer_height),
    )
    weights = OmegaConf.to_container(cfg.weights, resolve=True)

    archive = Archive(int(cfg.bins_compactness), int(cfg.bins_height),
                      tuple(cfg.compactness_range), tuple(cfg.height_range))

    seed_genomes = []
    if cfg.get("resume_from"):
        seed_genomes = list(archive.load_genomes(cfg.resume_from))
        log.info("Resuming from %s: %d seed genomes", cfg.resume_from,
                 len(seed_genomes))

    archive_path = os.path.join(out_dir, "archive.npz")
    montage_path = os.path.join(out_dir, "archive.png")
    n_evals = int(cfg.n_evals)
    init_batch = int(cfg.init_batch)
    sigma = float(cfg.sigma)
    n_dead = 0
    best_fit = -np.inf
    history = []

    for ev in range(n_evals):
        # Genome source: resume seeds → random init batch → mutate an elite.
        if seed_genomes:
            parent = seed_genomes.pop()
            genome = np.clip(parent + rng.normal(0, sigma, G_DIM), 0, 1)
        elif ev < init_batch or archive.n_filled == 0:
            genome = rng.random(G_DIM)
        else:
            elites = archive.occupied_genomes()
            parent = elites[rng.integers(len(elites))]
            genome = np.clip(parent + rng.normal(0, sigma, G_DIM), 0, 1)

        try:
            res = evaluate(genome, fixed, sim, weights, renderer)
        except Exception as e:
            log.warning("[%d] eval error: %s", ev, e)
            res = None
        if res is None:
            n_dead += 1
            history.append({"ev": ev, "status": "dead"})
            continue

        fitness, compactness, height, metrics, image, nverts = res
        status = archive.add(genome, fitness, compactness, height, metrics, image)
        best_fit = max(best_fit, fitness)
        history.append({
            "ev": ev, "status": status, "fitness": float(fitness),
            "compactness": float(compactness), "height": float(height),
            "n_verts": int(nverts),
            "overhang": float(metrics["overhang_fraction"]),
        })
        log.info("[%d/%d] %-8s fit=%.3f comp=%.1f h=%.0f overh=%.2f "
                 "filled=%d/%d nv=%d", ev + 1, n_evals, status, fitness,
                 compactness, height, metrics["overhang_fraction"],
                 archive.n_filled, archive.bins_c * archive.bins_h, nverts)

        if (ev + 1) % int(cfg.save_every) == 0:
            archive.save(archive_path)
            save_montage(archive, montage_path)
            with open(os.path.join(out_dir, "history.json"), "w") as f:
                json.dump(history, f)

    archive.save(archive_path)
    save_montage(archive, montage_path)
    with open(os.path.join(out_dir, "history.json"), "w") as f:
        json.dump(history, f)

    # Dump the best genome per filled cell as human-readable JSON.
    cells = []
    for (i, j), m in archive.metrics.items():
        cells.append({
            "cell": [i, j], "fitness": float(archive.fitness[i, j]),
            "genes": decode(archive.genomes[i, j]), "metrics": m,
        })
    with open(os.path.join(out_dir, "elites.json"), "w") as f:
        json.dump(cells, f, indent=2)

    log.info("=" * 60)
    log.info("Done: %d/%d cells filled, %d dead, best fitness=%.3f",
             archive.n_filled, archive.bins_c * archive.bins_h, n_dead, best_fit)
    log.info("Archive: %s  Montage: %s", archive_path, montage_path)


if __name__ == "__main__":
    main()
