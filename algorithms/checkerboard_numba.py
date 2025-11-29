"""
algorithms.checkerboard_numba.py

Contains the CPU multi-core python checkerboard algorithm accelerated with Numba JIT
"""


import numpy as np
from numba import njit, prange
from .ising_updater import IsingUpdater, MeasureData


# Perform one full checkerboard sweep on the lattice (both parities) in-place
# `rand_matrix` should be a (L, L) array of uniform [0, 1) random numbers.
@njit(parallel=True)
def _checkerboard_sweep_njit_parallel(lattice: np.ndarray, rand_matrix: np.ndarray, L: int, T: float, J: float):
    # update black and white sites separately
    for parity in (0, 1):
        for i in prange(L):
            for j in range((i + parity) % 2, L, 2):
                s = lattice[i, j]
                nb = (
                    lattice[(i + 1) % L, j]
                    + lattice[(i - 1) % L, j]
                    + lattice[i, (j + 1) % L]
                    + lattice[i, (j - 1) % L]
                )

                dE = 2.0 * J * s * nb
                if dE <= 0.0 or rand_matrix[i, j] < np.exp(-dE / T):
                    lattice[i, j] = -s


# Same sweep as above, just without the parallel flag so is serial
@njit
def _checkerboard_sweep_njit(lattice: np.ndarray, rand_matrix: np.ndarray, L: int, T: float, J: float):
    for parity in (0, 1):
        for i in range(L):
            for j in range((i + parity) % 2, L, 2):
                s = lattice[i, j]
                nb = (
                    lattice[(i + 1) % L, j]
                    + lattice[(i - 1) % L, j]
                    + lattice[i, (j + 1) % L]
                    + lattice[i, (j - 1) % L]
                )

                dE = 2.0 * J * s * nb
                if dE <= 0.0 or rand_matrix[i, j] < np.exp(-dE / T):
                    lattice[i, j] = -s


# Compiled energy sum
@njit
def _compute_energy_njit(lattice, J):
    L = lattice.shape[0]
    E = 0.0

    # sum over right and down neighbors - avoid double counting
    for i in range(L):
        for j in range(L):
            E -= J * lattice[i, j] * lattice[(i + 1) % L, j]
            E -= J * lattice[i, j] * lattice[i, (j + 1) % L]

    return E


class CheckerboardNumbaUpdater(IsingUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1) -> None:
        self.L = L
        self.T = T
        self.J = J
        self.rng = np.random.default_rng(seed)
        self.lattice = np.ones((L, L), dtype=int)

        self._Nsites = self.L * self.L

    def initialize(self, hot: bool = True) -> None:
        if hot:
            self.lattice = self.rng.choice(np.array([-1, 1], dtype=np.int8), size=(self.L, self.L))
        else:
            self.lattice[:] = 1

    def sweep(self) -> None:
        unifs = self.rng.random(size=(self.L, self.L))
        _checkerboard_sweep_njit(self.lattice, unifs, self.L, self.T, self.J)
        
    def measure(self) -> MeasureData:
        E = _compute_energy_njit(self.lattice, self.J)
        M = float(self.lattice.sum())
        return MeasureData(E / self._Nsites, M / self._Nsites)

    def copy_state(self) -> np.ndarray:
        return self.lattice.copy()
    

# Same as normal, just with parallel sweep
class CheckerboardNumbaParallelUpdater(CheckerboardNumbaUpdater):
    def sweep(self) -> None:
        unifs = self.rng.random(size=(self.L, self.L))
        _checkerboard_sweep_njit_parallel(self.lattice, unifs, self.L, self.T, self.J)