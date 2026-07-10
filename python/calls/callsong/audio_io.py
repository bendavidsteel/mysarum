"""Audio + spectrogram output helpers."""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf


def save_wav(path: str, wave: np.ndarray, sr: int) -> None:
    w = np.asarray(wave, dtype=np.float32)
    peak = np.max(np.abs(w))
    if peak > 1e-6:
        w = 0.95 * w / peak
    sf.write(path, w, sr)


def spectrogram_png(path: str, wave: np.ndarray, sr: int, title: str = "",
                    fmax: float = 8000.0) -> None:
    w = np.asarray(wave, dtype=np.float32)
    S = librosa.amplitude_to_db(np.abs(librosa.stft(w, n_fft=1024, hop_length=256)),
                                ref=np.max)
    fig, ax = plt.subplots(figsize=(6, 3))
    librosa.display.specshow(S, sr=sr, hop_length=256, x_axis="time",
                             y_axis="log", ax=ax, cmap="magma")
    ax.set_ylim(100, fmax)
    if title:
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def archive_scatter_png(path: str, archive, projector=None) -> None:
    """Scatter of filled cells in descriptor space, coloured by fitness."""
    m = archive.occupied_mask()
    d = archive.descriptors[m]
    f = archive.fitness[m]
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(d[:, 0], d[:, 1], c=f, cmap="viridis", s=18)
    fig.colorbar(sc, ax=ax, label="fitness")
    ax.set_xlabel("BirdNET PC1")
    ax.set_ylabel("BirdNET PC2")
    ax.set_title(f"MAP-Elites archive — {archive.n_filled()} elites "
                 f"({100 * archive.coverage():.0f}% coverage)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def montage_png(path: str, waves: list[np.ndarray], sr: int,
                titles: list[str], ncols: int = 4) -> None:
    n = len(waves)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows),
                             squeeze=False)
    for k in range(nrows * ncols):
        ax = axes[k // ncols][k % ncols]
        if k < n:
            S = librosa.amplitude_to_db(
                np.abs(librosa.stft(waves[k], n_fft=1024, hop_length=256)),
                ref=np.max)
            librosa.display.specshow(S, sr=sr, hop_length=256, y_axis="log",
                                     ax=ax, cmap="magma")
            ax.set_ylim(100, 8000)
            ax.set_title(titles[k], fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
