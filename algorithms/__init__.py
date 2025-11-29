"""
algorithms.__init__.py

Loads all algorithms and a dictionary used by config parser
"""


from .ising_updater import IsingUpdater, MeasureData
from .metropolis import MetropolisUpdater
from .metropolis_numba import MetropolisNumbaUpdater

from .checkerboard import CheckerboardUpdater
from .checkerboard_numba import CheckerboardNumbaUpdater
from .checkerboard_numba import CheckerboardNumbaParallelUpdater

from .wolff import WolffUpdater
from .wolff_numba import WolffNumbaUpdater

from .checkerboard_opencl import CheckerboardOpenCLUpdater
from .double_checkerboard_opencl import DoubleCheckerboardOpenCLUpdater
from .double_checkerboard_row_opencl import DoubleCheckerboardRowOpenCLUpdater


# Used by config parser
algos_dict = {
    "Metropolis": MetropolisUpdater,
    "MetropolisNumba": MetropolisNumbaUpdater,

    "Checkerboard": CheckerboardUpdater,
    "CheckerboardNumba": CheckerboardNumbaUpdater,
    "CheckerboardNumbaParallel": CheckerboardNumbaParallelUpdater,

    "Wolff": WolffUpdater,
    "WolffNumba": WolffNumbaUpdater,

    "CheckerboardOpenCL": CheckerboardOpenCLUpdater,
    "DoubleCheckerboardOpenCL": DoubleCheckerboardOpenCLUpdater,
    "DoubleCheckerboardRowOpenCL": DoubleCheckerboardRowOpenCLUpdater
}