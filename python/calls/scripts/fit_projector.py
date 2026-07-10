"""Fit the BirdNET-PCA descriptor projector on real recordings.

Two sources (either or both):

  * ``--audio_dir DIR``   — any local folder of audio files (wav/mp3/flac/ogg).
  * ``--xc``              — download from Xeno-canto across taxa (birds, frogs,
                           grasshoppers/insects, ...). Xeno-canto API v3
                           requires a personal API key (env ``XC_API_KEY``);
                           since 2025-10-10 it is needed to download recordings.

Every file is loaded at 48 kHz, sliced into 3-s segments, embedded by BirdNET,
and the first two principal components of all embeddings become the archive's
behaviour axes. Fitting on birds+frogs+insects makes those axes span a wide
cross-taxa timbral gamut, so synthetic calls are placed relative to real
animal sounds.

Examples
--------
    # local folder
    uv run python scripts/fit_projector.py --audio_dir ~/sounds --out projector.npz

    # Xeno-canto, multiple taxa (needs XC_API_KEY)
    XC_API_KEY=... uv run python scripts/fit_projector.py --xc \
        --groups birds,frogs,grasshoppers --per_group 80 --out projector.npz

Then point a run at it:
    uv run python run_mapelites.py projector_path=projector.npz
"""

import argparse
import glob
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import librosa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from callsong.birdnet import BirdNetEmbedder, DescriptorProjector, BIRDNET_SR

SEG = 3.0                       # BirdNET segment length (s)
AUDIO_EXT = ("*.wav", "*.mp3", "*.flac", "*.ogg", "*.WAV", "*.mp3")
XC_API = "https://xeno-canto.org/api/3/recordings"


def segments_from_file(path: str, seg_per_file: int, min_rms: float = 5e-3):
    """Load a file at 48 kHz mono and yield up to ``seg_per_file`` energetic
    3-s segments. Only the opening window of each file is loaded (recordings
    can be minutes long, and q:A clips front-load the target sound)."""
    load_dur = max(SEG * seg_per_file * 2 + 6.0, 30.0)
    try:
        y, _ = librosa.load(path, sr=BIRDNET_SR, mono=True, duration=load_dur)
    except Exception as e:  # noqa: BLE001 — skip unreadable files, keep going
        print(f"  skip {os.path.basename(path)}: {e}")
        return
    n = int(SEG * BIRDNET_SR)
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    n_seg = len(y) // n
    picked = 0
    for k in range(n_seg):
        if picked >= seg_per_file:
            break
        s = y[k * n:(k + 1) * n]
        if np.sqrt(np.mean(s**2)) >= min_rms:
            picked += 1
            yield s.astype(np.float32)


def gather_local(audio_dir: str, seg_per_file: int, max_files: int):
    files = []
    for ext in AUDIO_EXT:
        files.extend(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True))
    files = sorted(set(files))[:max_files] if max_files else sorted(set(files))
    print(f"local: {len(files)} files under {audio_dir}")
    segs, labels = [], []
    for f in files:
        for s in segments_from_file(f, seg_per_file):
            segs.append(s)
            labels.append("local")
    return segs, labels


def gather_xenocanto(groups, per_group, seg_per_file, cache_dir):
    import requests
    key = os.environ.get("XC_API_KEY")
    if not key:
        sys.exit("XC_API_KEY not set — get one from your xeno-canto account "
                 "(Account > API key) and export XC_API_KEY=...")
    os.makedirs(cache_dir, exist_ok=True)
    segs, labels = [], []
    for grp in groups:
        print(f"xeno-canto grp:{grp} — querying...")
        got = 0
        page = 1
        while got < per_group:
            r = requests.get(XC_API, params={"query": f"grp:{grp} q:A",
                                             "key": key, "page": page}, timeout=60)
            r.raise_for_status()
            data = r.json()
            recs = data.get("recordings", [])
            if not recs:
                break
            for rec in recs:
                if got >= per_group:
                    break
                url = rec.get("file") or ""
                if url.startswith("//"):
                    url = "https:" + url
                if not url:
                    continue
                dest = os.path.join(cache_dir, f"{grp}_{rec.get('id','x')}.mp3")
                if not os.path.exists(dest):
                    try:
                        au = requests.get(url, params={"key": key}, timeout=120)
                        au.raise_for_status()
                        with open(dest, "wb") as fh:
                            fh.write(au.content)
                    except Exception as e:  # noqa: BLE001
                        print(f"  dl fail {rec.get('id')}: {e}")
                        continue
                for s in segments_from_file(dest, seg_per_file):
                    segs.append(s)
                    labels.append(grp)
                got += 1
            if page >= int(data.get("numPages", 1)):
                break
            page += 1
        print(f"  grp:{grp} -> {got} recordings")
    return segs, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=None)
    ap.add_argument("--xc", action="store_true", help="download from Xeno-canto")
    ap.add_argument("--groups", default="birds,frogs,grasshoppers",
                    help="comma-separated xeno-canto grp values")
    ap.add_argument("--per_group", type=int, default=60)
    ap.add_argument("--seg_per_file", type=int, default=3)
    ap.add_argument("--max_files", type=int, default=0, help="0 = no cap (local)")
    ap.add_argument("--cache_dir", default="_xc_cache")
    ap.add_argument("--out", default="projector.npz")
    args = ap.parse_args()

    segs, labels = [], []
    if args.audio_dir:
        s, l = gather_local(args.audio_dir, args.seg_per_file, args.max_files)
        segs += s; labels += l
    if args.xc:
        s, l = gather_xenocanto([g.strip() for g in args.groups.split(",")],
                                args.per_group, args.seg_per_file, args.cache_dir)
        segs += s; labels += l
    if not segs:
        sys.exit("no audio segments gathered — pass --audio_dir and/or --xc")

    print(f"embedding {len(segs)} segments with BirdNET...")
    t0 = time.time()
    embedder = BirdNetEmbedder()
    embs = embedder.embed_many(np.stack(segs), BIRDNET_SR)
    print(f"embedded in {time.time() - t0:.1f}s")

    projector = DescriptorProjector.fit(embs)
    projector.save(args.out)
    labs, counts = np.unique(labels, return_counts=True)
    print("segments per source:", dict(zip(labs.tolist(), counts.tolist())))
    print(f"PCA bounds: {projector.bounds.tolist()}")
    print(f"saved projector -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
