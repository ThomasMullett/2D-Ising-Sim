# 2D-Ising-Sim
GPU-accelerated Monte Carlo simulations of the 2D Ising model using OpenCL, Numba, and C++/Python CPU implementations. Includes performance benchmarks, autocorrelation analysis, Wolff/Metropolis algorithms, and a research manuscript validating all methods.

## Features
- **OpenCL GPU kernels** (checkerboard + double-checkerboard tiling)
- **CPU implementations**: serial C++, OpenMP-parallel, Python + Numba JIT
- **Wolff cluster algorithm** (Numba)
- **Performance benchmarks:** time per spin, scaling with lattice size
- **Physics validation:** magnetization, Binder cumulants, residuals to Onsager’s solution
- **Autocorrelation analysis** near the critical point
- **Up to 4 orders of magnitude speedup** over optimised CPU versions

## Repository Structure
- /src – CPU, GPU and Numba implementations
- /utils - Utility functions
- /paper – Research manuscript (PDF + LaTeX)
