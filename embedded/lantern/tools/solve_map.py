# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "polars", "scipy"]
# ///
"""Combine per-view LED detections into 3D positions and a neighbour graph.

Each view is the lantern turned clockwise (seen from above) by the view's angle in front of a
fixed, distant camera, so image x is a projection of the rotated horizontal plane and image y
is height. Every LED's (x, z, height) is fitted to the views that saw it, weighted by how
brightly it showed up, then gaps are filled along the strip.

    uv run tools/solve_map.py
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
from scipy.optimize import least_squares

MAP_DIR = Path(__file__).resolve().parent.parent / "mapping"


def load(min_snr: float) -> pl.DataFrame:
    frames = []
    for d in sorted(MAP_DIR.glob("view_*")):
        try:
            angle = float(d.name.removeprefix("view_"))
        except ValueError:
            continue
        frames.append(pl.read_parquet(d / "detections.parquet").with_columns(angle=pl.lit(angle)))
    df = pl.concat(frames, how="diagonal_relaxed")
    return df.filter(pl.col("snr") > min_snr)


def solve(df: pl.DataFrame, leds: list[tuple[int, int]]):
    """Fit per-LED 3D points plus a horizontal image offset per view (the axis may drift)."""
    angles = sorted(df["angle"].unique().to_list())
    idx = {led: k for k, led in enumerate(leds)}
    obs = df.select(
        pl.struct("output", "index").map_elements(lambda r: idx[(r["output"], r["index"])], return_dtype=pl.Int64).alias("k"),
        pl.col("angle").map_elements(angles.index, return_dtype=pl.Int64).alias("v"),
        "x", "y", "angle", "peak",
    )
    k, v = obs["k"].to_numpy(), obs["v"].to_numpy()
    theta = np.radians(obs["angle"].to_numpy())
    u, h = obs["x"].to_numpy(), obs["y"].to_numpy()
    # Glow seen through the far side of the paper is dimmer and blurrier; trust bright views more.
    w = np.sqrt(obs["peak"].to_numpy() / obs["peak"].max())

    n, nv = len(leds), len(angles)
    seen = np.bincount(k, minlength=n)

    def unpack(p):
        return p[: 3 * n].reshape(n, 3), np.concatenate([[0.0], p[3 * n :]])

    def residuals(p):
        pts, cx = unpack(p)
        # Clockwise turn seen from above moves the lantern's +x side away from the camera.
        proj = pts[k, 0] * np.cos(theta) + pts[k, 1] * np.sin(theta) + cx[v]
        return np.concatenate([w * (proj - u), w * (pts[k, 2] - h)])

    center = np.median(u)
    p0 = np.zeros(3 * n + nv - 1)
    pts0 = p0[: 3 * n].reshape(n, 3)
    for kk in range(n):
        m = k == kk
        pts0[kk] = [np.mean(u[m]) - center if m.any() else 0, 0, np.mean(h[m]) if m.any() else np.median(h)]
    p0[: 3 * n] = pts0.ravel()
    fit = least_squares(residuals, p0, loss="soft_l1", f_scale=10.0)
    pts, cx = unpack(fit.x)
    return pts, seen, cx, angles, np.abs(fit.fun[: len(u)] / np.maximum(w, 1e-6))


def fill_gaps(pts: np.ndarray, seen: np.ndarray, leds: list[tuple[int, int]]) -> np.ndarray:
    """Interpolate LEDs seen in fewer than two views from their neighbours on the same strip."""
    pts = pts.copy()
    outputs = np.array([o for o, _ in leds])
    for o in np.unique(outputs):
        ks = np.nonzero(outputs == o)[0]
        good = seen[ks] >= 2
        for dim in range(3):
            pts[ks[~good], dim] = np.interp(ks[~good], ks[good], pts[ks[good], dim])
    return pts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-snr", type=float, default=15.0)
    p.add_argument("--neighbours", type=int, default=6)
    args = p.parse_args()

    df = load(args.min_snr)
    every = pl.concat([pl.read_parquet(d / "detections.parquet") for d in MAP_DIR.glob("view_*")])
    leds = sorted({(o, i) for o, i in zip(every["output"], every["index"])})
    pts, seen, cx, angles, err = solve(df, leds)
    pts = fill_gaps(pts, seen, leds)

    # Centre on the lantern and scale so the median gap between consecutive LEDs is 1.
    pts -= pts.mean(axis=0)
    pts[:, 2] *= -1  # image y grows downwards
    outputs = np.array([o for o, _ in leds])
    gaps = np.linalg.norm(np.diff(pts, axis=0), axis=1)[outputs[1:] == outputs[:-1]]
    pts /= np.median(gaps)

    out = pl.DataFrame({
        "output": [o for o, _ in leds],
        "index": [i for _, i in leds],
        "x": pts[:, 0], "y": pts[:, 1], "z": pts[:, 2],
        "views_seen": seen,
    })
    out.write_parquet(MAP_DIR / "positions.parquet", compression="zstd")

    from scipy.spatial import cKDTree
    dist, nbr = cKDTree(pts).query(pts, k=args.neighbours + 1)
    edges = pl.DataFrame({
        "src": np.repeat(np.arange(len(leds)), args.neighbours),
        "dst": nbr[:, 1:].ravel(),
        "distance": dist[:, 1:].ravel(),
    })
    edges.write_parquet(MAP_DIR / "neighbours.parquet", compression="zstd")

    print(f"views {angles}, per-view x offsets {np.round(cx, 1)}")
    print(f"{len(leds)} LEDs; seen in >=2 views: {(seen >= 2).sum()}; median reprojection error {np.median(err):.1f}px")
    print(f"consecutive-LED gap spread (1 = median): p10 {np.percentile(gaps / np.median(gaps), 10):.2f}, "
          f"p90 {np.percentile(gaps / np.median(gaps), 90):.2f}")
    print(f"wrote {MAP_DIR / 'positions.parquet'} and neighbours.parquet")


if __name__ == "__main__":
    main()
