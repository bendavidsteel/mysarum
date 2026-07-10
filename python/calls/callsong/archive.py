"""MAP-Elites archive over the 2-D BirdNET-PCA descriptor space."""

from __future__ import annotations

import numpy as np

from . import genome as gm


class Archive:
    """A regular 2-D grid of elites; each cell keeps the highest-fitness call."""

    def __init__(self, bounds: np.ndarray, resolution: int) -> None:
        self.bounds = np.asarray(bounds, dtype=np.float64)  # (2,2)
        self.res = int(resolution)
        shape = (self.res, self.res)
        self.fitness = np.full(shape, -np.inf)
        self.genomes = np.zeros((*shape, gm.N_PARAMS), dtype=np.float32)
        self.descriptors = np.full((*shape, 2), np.nan)
        self.gesture_id = np.full(shape, -1, dtype=np.int32)
        self.feats: dict[tuple[int, int], dict] = {}

    def _cell(self, desc: np.ndarray) -> tuple[int, int]:
        (lo0, hi0), (lo1, hi1) = self.bounds
        f0 = (desc[0] - lo0) / (hi0 - lo0 + 1e-12)
        f1 = (desc[1] - lo1) / (hi1 - lo1 + 1e-12)
        i = int(np.clip(f0 * self.res, 0, self.res - 1))
        j = int(np.clip(f1 * self.res, 0, self.res - 1))
        return i, j

    def add(self, genome: np.ndarray, descriptor: np.ndarray, fitness: float,
            gesture_id: int, feats: dict | None = None) -> bool:
        i, j = self._cell(descriptor)
        if fitness > self.fitness[i, j]:
            self.fitness[i, j] = fitness
            self.genomes[i, j] = genome
            self.descriptors[i, j] = descriptor
            self.gesture_id[i, j] = gesture_id
            if feats is not None:
                self.feats[(i, j)] = feats
            return True
        return False

    def occupied_mask(self) -> np.ndarray:
        return np.isfinite(self.fitness)

    def coverage(self) -> float:
        return float(self.occupied_mask().mean())

    def n_filled(self) -> int:
        return int(self.occupied_mask().sum())

    def qd_score(self) -> float:
        m = self.occupied_mask()
        return float(self.fitness[m].sum()) if m.any() else 0.0

    def elite_genomes(self) -> np.ndarray:
        """All filled cells' genomes, shape (n_filled, N_PARAMS)."""
        return self.genomes[self.occupied_mask()]

    def sample_elites(self, rng: np.random.Generator, n: int) -> np.ndarray:
        elites = self.elite_genomes()
        if len(elites) == 0:
            return gm.random_genomes(rng, n)
        idx = rng.integers(0, len(elites), size=n)
        return elites[idx]

    def best(self) -> tuple[tuple[int, int], float]:
        m = self.occupied_mask()
        flat = np.where(m, self.fitness, -np.inf)
        i, j = np.unravel_index(np.argmax(flat), flat.shape)
        return (int(i), int(j)), float(flat[i, j])

    def save(self, path: str) -> None:
        np.savez(path, bounds=self.bounds, res=self.res, fitness=self.fitness,
                 genomes=self.genomes, descriptors=self.descriptors,
                 gesture_id=self.gesture_id)
