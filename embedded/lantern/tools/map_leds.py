# /// script
# requires-python = ">=3.11"
# dependencies = ["opencv-python-headless", "numpy", "polars", "pyserial"]
# ///
"""Light LEDs one at a time and locate each one in the webcam image.

    uv run tools/map_leds.py capture --view 0
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import polars as pl
import serial

OUT_DIR = Path(__file__).resolve().parent.parent / "mapping"


class Lantern:
    def __init__(self, port: str):
        self.serial = serial.Serial(port, 115200, timeout=2)
        # Opening the port can pulse EN; give the firmware time to boot.
        time.sleep(2.0)
        # Commands sent while the animation is running can be dropped, so retry the first one.
        for _ in range(10):
            self.serial.reset_input_buffer()
            try:
                self.show()
                return
            except RuntimeError:
                pass
        raise RuntimeError("lantern not answering")

    def show(self, pixels: list[tuple[int, int]] = (), color: tuple[int, int, int] = (0, 0, 0)):
        rgb = " ".join(map(str, color))
        cmds = ["c"] + [f"s {o} {i} {rgb}" for o, i in pixels] + ["w"]
        self.serial.write(("\n".join(cmds) + "\n").encode())
        reply = self.serial.readline().strip()
        if not reply.startswith(b"ok"):
            raise RuntimeError(f"lantern replied {reply!r}")

    def animate(self):
        self.serial.write(b"a\n")


class Camera:
    def __init__(self, index: int, exposure: int):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2: 1 = manual
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        for _ in range(30):
            self.cap.read()

    def grab(self, skip: int = 4) -> np.ndarray:
        """Grayscale float frame, dropping frames buffered before the LEDs changed."""
        for _ in range(skip):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError("camera read failed")
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    def grab_mean(self, n: int) -> np.ndarray:
        return np.mean([self.grab() for _ in range(n)], axis=0)


def lantern_mask(diff: np.ndarray) -> np.ndarray:
    """Dilated region lit up when every LED is on, so stray lights elsewhere are ignored."""
    blurred = cv2.GaussianBlur(diff, (0, 0), 8)
    mask = (blurred > 0.1 * blurred.max()).astype(np.uint8)
    return cv2.dilate(mask, np.ones((61, 61), np.uint8)).astype(bool)


def locate(diff: np.ndarray, mask: np.ndarray) -> dict:
    """Brightness-weighted centroid of the glow blob around the brightest point."""
    blurred = cv2.GaussianBlur(diff, (0, 0), 4) * mask
    _, peak, _, (px, py) = cv2.minMaxLoc(blurred)
    mask = (blurred > 0.5 * peak).astype(np.uint8)
    _, labels = cv2.connectedComponents(mask)
    blob = labels == labels[py, px]
    w = blurred * blob
    ys, xs = np.nonzero(blob)
    total = w[ys, xs].sum()
    return {
        "x": float((xs * w[ys, xs]).sum() / total),
        "y": float((ys * w[ys, xs]).sum() / total),
        "peak": float(peak),
        "area": int(blob.sum()),
    }


def capture(args):
    view_dir = OUT_DIR / f"view_{args.view}"
    view_dir.mkdir(parents=True, exist_ok=True)
    lantern = Lantern(args.port)
    cam = Camera(args.camera, args.exposure)

    lantern.show()
    background = cam.grab_mean(8)
    noise = float(np.std(cam.grab() - background))
    cv2.imwrite(str(view_dir / "background.jpg"), background.astype(np.uint8))

    color = tuple(args.color)
    leds = [(o, i) for o in args.outputs for i in range(args.leds)]
    lantern.show(leds, color)
    all_on = np.clip(cam.grab_mean(4) - background, 0, None)
    mask = lantern_mask(all_on)
    cv2.imwrite(str(view_dir / "all_on.jpg"), np.clip(all_on * 2, 0, 255).astype(np.uint8))
    lantern.show()

    rows, thumbs = [], []
    start = time.time()
    for n, (o, i) in enumerate(leds):
        if n % 50 == 0 and n:
            lantern.show()
            background = cam.grab_mean(4)
            rate = n / (time.time() - start)
            print(f"{n}/{len(leds)}  {rate:.1f} led/s  ~{(len(leds) - n) / rate:.0f}s left", flush=True)
        lantern.show([(o, i)], color)
        diff = np.clip(cam.grab() - background, 0, None)
        rows.append({"output": o, "index": i, **locate(diff, mask)})
        thumbs.append(cv2.resize(diff, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).clip(0, 255).astype(np.uint8))

    lantern.show()
    lantern.animate()

    df = pl.DataFrame(rows).with_columns(snr=pl.col("peak") / max(noise, 1e-3), view=pl.lit(args.view))
    df.write_parquet(view_dir / "detections.parquet", compression="zstd")
    np.savez_compressed(view_dir / "diffs.npz", diffs=np.stack(thumbs))
    overlay(df, background, view_dir / "overlay.jpg")
    print(f"noise {noise:.2f}; wrote {view_dir}")
    summary = df.group_by("output").agg(
        detected=(pl.col("snr") > args.min_snr).sum(),
        median_snr=pl.col("snr").median(),
    )
    print(summary.sort("output"))


def overlay(df: pl.DataFrame, background: np.ndarray, path: Path, min_snr: float = 8.0):
    img = cv2.cvtColor(np.clip(background * 3, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    colors = [(0, 0, 255), (0, 255, 0), (255, 128, 0), (0, 255, 255)]
    for o, group in df.filter(pl.col("snr") > min_snr).group_by("output"):
        pts = group.sort("index").select("x", "y").to_numpy().astype(np.int32)
        cv2.polylines(img, [pts], False, colors[o[0] % 4], 1)
        for x, y in pts:
            cv2.circle(img, (x, y), 3, colors[o[0] % 4], -1)
    cv2.imwrite(str(path), img)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    c = sub.add_parser("capture")
    c.add_argument("--view", required=True, help="label for this camera angle, e.g. 0, 90, 180, 270")
    c.add_argument("--port", default="/dev/ttyUSB0")
    c.add_argument("--camera", type=int, default=4)
    c.add_argument("--exposure", type=int, default=300)
    c.add_argument("--outputs", type=int, nargs="+", default=[0, 1])
    c.add_argument("--leds", type=int, default=45)
    c.add_argument("--color", type=int, nargs=3, default=[0, 0, 255], help="R G B used to light each LED")
    c.add_argument("--min-snr", type=float, default=8.0)
    c.set_defaults(func=capture)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
