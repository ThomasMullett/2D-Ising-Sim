"""
algorithms.double_checkerboard_row_opencl.py

Contains a class for the double checkerboard row implementation
"""

import numpy as np
import pyopencl as cl
from .double_checkerboard_opencl import DoubleCheckerboardOpenCLUpdater

class DoubleCheckerboardRowOpenCLUpdater(DoubleCheckerboardOpenCLUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1, k: int = 1) -> None:
        super().__init__(L, T, seed, J=J, k=k)

        self.kernel = self.prg.double_checkerboard_row

        # Tile, local and global sizes
        self.t_size = 32
        self.g_size = (int(self.L/self.t_size), self.L)
        self.l_size = (1, self.t_size)
        self.local_spin_cache = cl.LocalMemory(np.dtype(np.int8).itemsize * (self.t_size+2)**2)