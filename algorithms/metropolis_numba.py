"""
algorithms.metropolis_numba.py

Contains a Numba JIT compiled, serial, naive, metropolis algorithm
"""


import numpy as np
from numba import njit
from .ising_updater import IsingUpdater, MeasureData


# Perform N = L*L random single-spin Metropolis attempts, modifying lattice in-place
# Uses pre-computed random indicies and uniforms
@njit
def _metropolis_sweep_njit(lattice: np.ndarray, idx_i: np.ndarray, idx_j: np.ndarray, unifs: np.ndarray, T: float, J: float):
    L = lattice.shape[0]
    Nsites = idx_i.shape[0]
    for k in range(Nsites):
        i = idx_i[k]
        j = idx_j[k]
        s = lattice[i, j]
        # neighbor sum with periodic BCs
        nb =  (
                lattice[(i + 1) % L, j] +
                lattice[(i - 1) % L, j] +
                lattice[i, (j + 1) % L] +
                lattice[i, (j - 1) % L]
        )
        
        dE = 2.0 * J * s * nb

        if dE <= 0.0:
            lattice[i, j] = -s
        else:
            # acceptance test using pre-drawn uniform random number unifs[k]
            if unifs[k] < np.exp(-dE / T):
                lattice[i, j] = -s


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


class MetropolisNumbaUpdater(IsingUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1) -> None:
        self.L = L
        self.T = T
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.lattice = np.ones((L, L), dtype=int)

        self._Nsites = self.L * self.L
        self._idx_i = np.empty(self._Nsites, dtype=np.int64)
        self._idx_j = np.empty(self._Nsites, dtype=np.int64)
        self._unifs = np.empty(self._Nsites, dtype=np.float64)

    def initialize(self, hot: bool = True) -> None:
        if hot:
            self.lattice = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(self.L, self.L))
        else:
            self.lattice[:] = 1

    def sweep(self) -> None:
        # Pre-compute indicies and uniforms
        self._idx_i[:] = self.rng.integers(0, self.L, size=self._Nsites)
        self._idx_j[:] = self.rng.integers(0, self.L, size=self._Nsites)
        self._unifs[:] = self.rng.random(size=self._Nsites)

        _metropolis_sweep_njit(self.lattice, self._idx_i, self._idx_j, self._unifs, self.T, self.J)

    def measure(self) -> MeasureData:
        E = _compute_energy_njit(self.lattice, self.J)
        M = float(self.lattice.sum())
        return MeasureData(E / self._Nsites, M / self._Nsites)

    def copy_state(self) -> np.ndarray:
        return self.lattice.copy()


