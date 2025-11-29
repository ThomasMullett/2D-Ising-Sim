"""
algorithms.wolff_numba.py

Contains a Wolff algorithm implementation written using Numba JIT
"""


import numpy as np
from numba import njit
from .ising_updater import IsingUpdater, MeasureData


# JIT wolff algorithm with custom queue datastructure to all nopython compilation
@njit
def _wolff_sweep_njit(lattice: np.ndarray, i0: int, j0: int, p_add: float, L: int):
    spin0 = lattice[i0, j0]

    # "Bool" array to represent if spins are in the cluster
    in_cluster = np.zeros((L, L), dtype=np.uint8)
    in_cluster[i0, j0] = True
    
    queue_i = np.empty(L * L, np.int64)
    queue_j = np.empty(L * L, np.int64)
    head = 0
    tail = 1
    queue_i[head] = i0
    queue_j[head] = j0

    # While elements in queue
    while head < tail:
        i, j = queue_i[head], queue_j[head]
        head += 1

        # Loop over neighbours
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = (i + di) % L, (j + dj) % L

            # Acceptance criterion
            if not in_cluster[ni, nj] and lattice[ni, nj] == spin0:
                if np.random.random() < p_add:
                    in_cluster[ni, nj] = True
                    queue_i[tail], queue_j[tail] = ni, nj
                    tail += 1

    # Flip the whole cluster
    for k in range(tail):
        i = queue_i[k]
        j = queue_j[k]
        lattice[i, j] = -lattice[i, j]


# Compiled energy sum
@njit
def _compute_energy_njit(lattice, J):
    L = lattice.shape[0]
    E = 0.0

    # sum over right and down neighbors
    for i in range(L):
        for j in range(L):
            E -= J * lattice[i, j] * lattice[(i + 1) % L, j]
            E -= J * lattice[i, j] * lattice[i, (j + 1) % L]

    return E


class WolffNumbaUpdater(IsingUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1.0) -> None:
        self.L = L
        self.T = T
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.lattice = np.ones((L, L), dtype=int)
        self.p_add = 1 - np.exp(-2 * J / T)  # Cluster add probability

        self._Nsites = self.L * self.L

    def initialize(self, hot: bool=True) -> None:
        if hot:
            self.lattice = self.rng.choice([-1, 1], size=(self.L, self.L))
        else:
            self.lattice[:] = 1

    def sweep(self) -> None:
        # Compute starting indicies
        i0, j0 = self.rng.integers(0, self.L, size=2)
        _wolff_sweep_njit(self.lattice, i0, j0, self.p_add, self.L)

    def measure(self) -> MeasureData:
        E = _compute_energy_njit(self.lattice, self.J)
        M = np.sum(self.lattice)
        return MeasureData(E / self._Nsites, M / self._Nsites)
    
    def copy_state(self) -> np.ndarray:
        return self.lattice.copy()
