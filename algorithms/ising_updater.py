"""
algorithms.ising_updater.py

Contians the base IsingUpdator class that all implementations inherit from,
alongside a dataclass used to store and return simulation output
"""


from dataclasses import dataclass
import numpy as np


@dataclass
class MeasureData:
    energy: float
    magnetization: float


class IsingUpdater:
    def __init__(self, L: int, T: float, seed: int, J: float=1.0) -> None:
        raise NotImplementedError

    def initialize(self, hot: bool=True) -> None:
        raise NotImplementedError

    def sweep(self) -> None:
        raise NotImplementedError

    def measure(self) -> MeasureData:
        raise NotImplementedError
    
    def copy_state(self) -> np.ndarray:
        raise NotImplementedError