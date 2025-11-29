"""
algorithms.checkerboard.py

Contains a checkerboard implementation in pure python
"""


import numpy as np
from .ising_updater import IsingUpdater, MeasureData


class CheckerboardUpdater(IsingUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1.0) -> None:
        self.L = L
        self.T = T
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.lattice = np.ones((L, L), dtype=int)

    def initialize(self, hot: bool=True) -> None:
        if hot:
            self.lattice = self.rng.choice([-1, 1], size=(self.L, self.L))
        else:
            self.lattice[:] = 1

    def sweep(self) -> None:
        L = self.L
        # Update both checkerboard sublattices
        for parity in [0, 1]:  # 0 = black, 1 = white
            for i in range(L):
                for j in range((i + parity) % 2, L, 2):
                    s = self.lattice[i, j]
                    nb = (
                        self.lattice[(i + 1) % L, j] +
                        self.lattice[(i - 1) % L, j] +
                        self.lattice[i, (j + 1) % L] +
                        self.lattice[i, (j - 1) % L]
                    )
                    dE = 2 * self.J * s * nb
                    if dE <= 0 or self.rng.random() < np.exp(-dE / self.T):
                        self.lattice[i, j] = -s

    def measure(self) -> MeasureData:
        E = 0.0
        E -= np.sum(self.lattice * np.roll(self.lattice, 1, axis=0))
        E -= np.sum(self.lattice * np.roll(self.lattice, 1, axis=1))
        M = np.sum(self.lattice)
        return MeasureData(E / (self.L * self.L), M / (self.L * self.L))
    
    def copy_state(self) -> np.ndarray:
        return self.lattice.copy()
