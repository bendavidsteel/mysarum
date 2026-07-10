"""BirdNET embeddings + a 2-D PCA descriptor projector.

BirdNET (v2.4, EfficientNetB0 backbone) produces a 1024-D embedding per 3-s,
48 kHz segment. We use the first two principal components of those embeddings
(fit on a bootstrap population) as the MAP-Elites behaviour descriptor, so the
archive tiles the perceptual manifold that a bioacoustic model actually
resolves — synthetic calls laid out where real bird-sound-types would sit.

BirdNET runs through ``tensorflow-cpu`` and never touches the GPU, so it
coexists with the JAX synthesiser without fighting over the 4 GB card.
"""

from __future__ import annotations

import logging
import os

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
# birdnetlib logs a line per analysed segment; quiet it for bulk embedding.
logging.getLogger("birdnetlib").setLevel(logging.WARNING)

BIRDNET_SR = 48_000
EMB_DIM = 1024


class BirdNetEmbedder:
    """Lazy wrapper around a single BirdNET analyzer instance."""

    def __init__(self) -> None:
        from birdnetlib.analyzer import Analyzer
        self._analyzer = Analyzer()
        # In-memory buffer support differs across birdnetlib versions; probe once.
        try:
            from birdnetlib import RecordingBuffer  # noqa: F401
            self._has_buffer = True
        except Exception:
            self._has_buffer = False

    def embed(self, wave: np.ndarray, sr: int = BIRDNET_SR) -> np.ndarray:
        """Return the 1024-D embedding of the first 3-s segment of ``wave``."""
        import contextlib
        w = np.asarray(wave, dtype=np.float32)
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            if self._has_buffer:
                from birdnetlib import RecordingBuffer
                rec = RecordingBuffer(self._analyzer, w, sr)
            else:
                import tempfile
                import soundfile as sf
                from birdnetlib import Recording
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                sf.write(tmp.name, w, sr)
                rec = Recording(self._analyzer, tmp.name)
            rec.extract_embeddings()
        if not rec.embeddings:
            return np.zeros(EMB_DIM, dtype=np.float32)
        return np.asarray(rec.embeddings[0]["embeddings"], dtype=np.float32)

    def embed_many(self, waves: np.ndarray, sr: int = BIRDNET_SR) -> np.ndarray:
        """Embed a batch of waveforms (N, T) -> (N, 1024). Sequential (CPU)."""
        return np.stack([self.embed(w, sr) for w in waves])


class DescriptorProjector:
    """First-2-PCA projector with archive bounds, fit on bootstrap embeddings."""

    def __init__(self, mean: np.ndarray, components: np.ndarray,
                 bounds: np.ndarray) -> None:
        self.mean = mean                 # (1024,)
        self.components = components      # (2, 1024)
        self.bounds = bounds              # (2, 2): [[lo0,hi0],[lo1,hi1]]

    @classmethod
    def fit(cls, embeddings: np.ndarray, pct: float = 2.0) -> "DescriptorProjector":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=0).fit(embeddings)
        proj = (embeddings - pca.mean_) @ pca.components_.T
        lo = np.percentile(proj, pct, axis=0)
        hi = np.percentile(proj, 100.0 - pct, axis=0)
        bounds = np.stack([lo, hi], axis=1)  # (2,2)
        return cls(pca.mean_.astype(np.float32),
                   pca.components_.astype(np.float32), bounds.astype(np.float32))

    def project(self, embeddings: np.ndarray) -> np.ndarray:
        e = np.atleast_2d(embeddings)
        return (e - self.mean) @ self.components.T  # (N, 2)

    def save(self, path: str) -> None:
        np.savez(path, mean=self.mean, components=self.components, bounds=self.bounds)

    @classmethod
    def load(cls, path: str) -> "DescriptorProjector":
        d = np.load(path)
        return cls(d["mean"], d["components"], d["bounds"])
