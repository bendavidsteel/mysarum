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


def gesture_bank_png(path: str, bank, sr: int) -> None:
    """The motor score: every gesture's (alpha, beta) path, plus where the
    selected elites sit in the (modulation rate x tension sweep) archive."""
    from callsong import gestures as g

    n = len(bank.paths)
    fig, axes = plt.subplots(n, 2, figsize=(11, 1.15 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [3, 1]})
    t = np.arange(bank.paths.shape[-1]) / sr
    for k in range(n):
        ax = axes[k][0]
        ax.plot(t, bank.paths[k, 0], lw=0.8, color="#d1495b", label=r"$\alpha$")
        ax.axhline(g.ALPHA_ONSET, lw=0.6, ls=":", color="0.4")
        ax.set_ylim(g.ALPHA_MIN, g.ALPHA_MAX)
        ax.set_ylabel(f"g{k}", fontsize=7)
        ax2 = ax.twinx()
        ax2.plot(t, bank.paths[k, 1], lw=0.8, color="#00798c", label=r"$\beta$")
        ax2.set_ylim(g.BETA_MIN, g.BETA_MAX)
        for a in (ax, ax2):
            a.tick_params(labelsize=6)
        if k < n - 1:
            ax.set_xticklabels([])
        d, q = bank.descriptors[k], bank.quality[k]
        label = ("ramp probe" if not np.isfinite(d[0]) else
                 f"{2 ** d[0]:.1f} mod/s   sweep {d[1]:+.2f}   q={q:.2f}")
        ax.set_title(label, fontsize=7, loc="left")
    axes[-1][0].set_xlabel("time (s)", fontsize=7)

    gs = axes[0][1].get_gridspec()
    for row in axes:
        row[1].remove()
    ax = fig.add_subplot(gs[:, 1])
    fin = np.isfinite(bank.descriptors[:, 0])
    ax.scatter(bank.descriptors[fin, 0], bank.descriptors[fin, 1],
               c=bank.quality[fin], cmap="viridis", s=40)
    for k in np.flatnonzero(fin):
        ax.annotate(f"g{k}", bank.descriptors[k], fontsize=6,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_xlim(g.RATE_LOG2_LO, g.RATE_LOG2_HI)
    ax.set_ylim(g.SWEEP_LO, g.SWEEP_HI)
    ax.set_xlabel("log2 modulation rate (Hz)", fontsize=7)
    ax.set_ylabel("tension sweep", fontsize=7)
    ax.set_title(f"gesture archive\n{100 * bank.coverage:.0f}% of cells filled",
                 fontsize=7)
    ax.tick_params(labelsize=6)
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
