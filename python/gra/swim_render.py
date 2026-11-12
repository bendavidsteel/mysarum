"""Swimmer video rendering: matplotlib artists built once and updated per frame,
with each video split into frame chunks rendered across processes and joined
by ffmpeg's concat demuxer (no re-encode).

Kept free of JAX so worker processes start fast and never touch the GPU.
"""

import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

BG = "#0b0f14"
GRID = "#1b2430"
TRAIL = "#e8c547"
FOOD = "#4fd6a0"
CHUNK = 120  # frames per job


class SwimmerView:
    """One swimmer panel. The camera follows the body; the grid is fixed in the fluid.

    Optional entries of sd: `view` (cx, cy, half width) fixes the camera; `coms`
    is a list of centre-of-mass traces, one trail each, when the panel holds
    several bodies; `source` (T, 3) is a nutrient source (x, y, range) per step,
    drawn as a glow that fades out at its range.
    """

    def __init__(self, ax, sd, every, label=None):
        from matplotlib.collections import LineCollection
        self.ax, self.sd, self.every = ax, sd, every
        ax.set_facecolor(BG)
        ax.set_aspect("equal")
        ax.axis("off")
        self.grid = LineCollection([], colors=GRID, linewidths=0.6, zorder=0)
        self.edges = LineCollection([], zorder=2)
        ax.add_collection(self.grid)
        ax.add_collection(self.edges)
        self.trails = [ax.plot([], [], color=TRAIL, lw=1.0, alpha=0.6, zorder=1)[0]
                       for _ in sd.get("coms", [None])]
        if "source" in sd:
            from matplotlib.colors import to_rgb
            g = np.linspace(-1, 1, 256)
            glow = np.zeros((256, 256, 4))
            glow[..., :3] = to_rgb(FOOD)
            glow[..., 3] = 0.45 * np.clip(1 - np.hypot(g[:, None], g[None]), 0, 1)
            self.glow = ax.imshow(glow, extent=(-1, 1, -1, 1), zorder=0.5, interpolation="bilinear")
            self.food = ax.scatter([0], [0], s=30, color=FOOD, edgecolors="white", linewidths=0.6, zorder=4)
        n = int(sd["active"].sum())
        self.nodes = ax.scatter(np.zeros(n), np.zeros(n), s=12, linewidths=0, zorder=3)
        if label:
            ax.text(0.03, 0.96, label, transform=ax.transAxes, color="#aab", fontsize=8, va="top")

    def update(self, t, i=None):
        """t indexes the full com trace; i (default t) indexes pos/x/act."""
        i = t if i is None else i
        sd, ax = self.sd, self.ax
        com, pos, extent, edges, active = sd["com"], sd["pos"], sd["extent"], sd["edges"], sd["active"]
        cx, cy = com[t]
        if "view" in sd:
            cx, cy, extent = sd["view"]
        ax.set_xlim(cx - extent, cx + extent)
        ax.set_ylim(cy - extent, cy + extent)
        gs = max(round(extent / 4), 1)
        xs = np.arange(np.floor((cx - extent) / gs) * gs, cx + extent, gs)
        ys = np.arange(np.floor((cy - extent) / gs) * gs, cy + extent, gs)
        self.grid.set_segments(
            [((x, cy - extent), (x, cy + extent)) for x in xs]
            + [((cx - extent, y), (cx + extent, y)) for y in ys])
        for trail, c in zip(self.trails, sd.get("coms", [com])):
            trail.set_data(c[:t + 1:self.every, 0], c[:t + 1:self.every, 1])
        if "source" in sd:
            sx, sy, rng = sd["source"][t]
            self.glow.set_extent((sx - rng, sx + rng, sy - rng, sy + rng))
            self.food.set_offsets([(sx, sy)])
        # colour = CA channels 0-2 as RGB; edge width = muscle contraction
        p, rgb = pos[i], 0.25 + 0.75 * np.clip(sd["x"][i], 0, 1)
        a = sd["act"][i][edges[:, 0], edges[:, 1]]
        self.edges.set_segments(np.stack([p[edges[:, 0]], p[edges[:, 2]]], 1))
        self.edges.set_color(0.5 * (rgb[edges[:, 0]] + rgb[edges[:, 2]]))
        self.edges.set_linewidth(0.5 + 2.5 * np.clip(-a, 0, 1))
        self.nodes.set_offsets(p[active])
        self.nodes.set_facecolor(rgb[active])


def _render_chunk(job):
    """Worker: render frames of one video spec into a part file."""
    import matplotlib
    matplotlib.use("Agg")
    import imageio.v2 as imageio
    import matplotlib.pyplot as plt
    spec, frames, path = job
    panels, cols = spec["panels"], spec["cols"]
    rows = (len(panels) + cols - 1) // cols
    cell, dpi = spec["cell"], spec["dpi"]
    fig, axes = plt.subplots(rows, cols, figsize=(cell * cols, cell * rows), dpi=dpi)
    fig.patch.set_facecolor(BG)
    fig.subplots_adjust(0, 0, 1, 1, 0.01, 0.01)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(panels):]:
        ax.set_facecolor(BG)
        ax.axis("off")
    views = [SwimmerView(ax, sd, spec["every"], lab) for ax, (sd, lab) in zip(axes, panels)]
    writer = imageio.get_writer(path, fps=spec["fps"], codec="libx264", quality=8,
                                macro_block_size=1)
    for i, t in enumerate(frames):
        for v in views:
            v.update(t, i)
        fig.canvas.draw()
        writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
    writer.close()
    plt.close(fig)
    return path


def _concat(parts, out):
    import imageio_ffmpeg
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.writelines(f"file '{p}'\n" for p in parts)
        lst = f.name
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "concat",
                    "-safe", "0", "-i", lst, "-c", "copy", str(out)], check=True)
    os.unlink(lst)
    for p in parts:
        os.unlink(p)


def render_videos(specs, workers=None):
    """Render many swimmer videos in parallel.

    Each spec: dict(path, panels=[(sd, label)], cols, cell, dpi, steps, every, fps).
    """
    tmp = Path(tempfile.mkdtemp(prefix="swim_render_"))
    jobs, parts = [], {}
    for i, spec in enumerate(specs):
        frames = list(range(0, spec["steps"], spec["every"]))
        parts[i] = []
        for c in range(0, len(frames), CHUNK):
            part = str(tmp / f"{i:03d}_{c // CHUNK:04d}.mp4")
            parts[i].append(part)
            fr = frames[c:c + CHUNK]
            # ship only this chunk's frames of the per-step recordings
            panels = [(dict(sd, pos=sd["pos"][fr], x=sd["x"][fr], act=sd["act"][fr]), lab)
                      for sd, lab in spec["panels"]]
            jobs.append((dict(spec, panels=panels), fr, part))
    # heaviest jobs (most panels) first so they don't straggle at the end
    jobs.sort(key=lambda j: -len(j[0]["panels"]))
    workers = workers or os.cpu_count()
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn")) as ex:
        list(ex.map(_render_chunk, jobs))
    for i, spec in enumerate(specs):
        _concat(parts[i], spec["path"])
    tmp.rmdir()
