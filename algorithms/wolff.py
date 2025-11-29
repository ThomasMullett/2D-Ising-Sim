"""
algorithms.wolff.py

Contains a Wolff implementation written in pure python
"""


import numpy as np
from collections import deque
from .ising_updater import IsingUpdater, MeasureData


class WolffUpdater(IsingUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1.0) -> None:
        self.L = L
        self.T = T
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.lattice = np.ones((L, L), dtype=int)
        self.p_add = 1 - np.exp(-2 * J / T)  # cluster add probability

    def initialize(self, hot: bool=True) -> None:
        if hot:
            self.lattice = self.rng.choice([-1, 1], size=(self.L, self.L))
        else:
            self.lattice[:] = 1

    # Perform one Wolff cluster update using deque from python standard library
    def sweep(self) -> None:
        L = self.L
        i0, j0 = self.rng.integers(0, L, size=2)
        spin0 = self.lattice[i0, j0]

        # Bool numpy mask for whether the spin is in the cluster
        cluster = np.zeros((L, L), dtype=bool)
        cluster[i0, j0] = True
        queue = deque([(i0, j0)])

        while queue:
            i, j = queue.popleft()

            # Loop over neighbours
            for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                ni, nj = (i + di) % L, (j + dj) % L

                # Acceptance criterion
                if not cluster[ni, nj] and self.lattice[ni, nj] == spin0:
                    if self.rng.random() < self.p_add:
                        cluster[ni, nj] = True
                        queue.append((ni, nj))

        # Flip the cluster
        self.lattice[cluster] *= -1

    def measure(self) -> MeasureData:
        E = 0.0
        E -= np.sum(self.lattice * np.roll(self.lattice, 1, axis=0))
        E -= np.sum(self.lattice * np.roll(self.lattice, 1, axis=1))
        M = np.sum(self.lattice)
        return MeasureData(E / (self.L * self.L), M / (self.L * self.L))
    
    def copy_state(self) -> np.ndarray:
        return self.lattice.copy()
