"""
algorithms.checkerboard_opencl.py

Contians the implementation for a regular checkerboard OpenCL algorithm
"""


import numpy as np
import pyopencl as cl
from .ising_updater import IsingUpdater, MeasureData


class CheckerboardOpenCLUpdater(IsingUpdater):
    def __init__(self, L: int, T: float, seed: int, J: float = 1) -> None:
        self.L = L
        self.T = T
        self.rng = np.random.default_rng(seed)
        self.J = J
        self._Nsites = self.L * self.L

        self.lattice = np.ones((L, L), dtype=np.int8)

        self.context = cl.create_some_context(interactive=False)
        self.queue = cl.CommandQueue(self.context)
        self.mf = cl.mem_flags

        self.spin_buffer = cl.Buffer(self.context, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=self.lattice)
        
        #rng_seeds = np.random.SeedSequence(seed).generate_state(L * L, dtype=np.uint64)
        rng_seeds =self.rng.random((self._Nsites, 2))
        self.rng_buffer = cl.Buffer(self.context, self.mf.READ_WRITE | self.mf.COPY_HOST_PTR, hostbuf=rng_seeds)

        self.prg = cl.Program(self.context, self._get_kernel_code()).build()
        self.kernel = self.prg.checkerboard

    # Initialise spins on the host and copy to GPU
    def initialize(self, hot: bool = True) -> None:
        if hot:
            self.lattice = self.rng.choice([-1, 1], size=(self.L, self.L)).astype(np.int8)
        else:
            self.lattice.fill(1)
        cl.enqueue_copy(self.queue, self.spin_buffer, self.lattice)

    # Perform one full Metropolis sweep using checkerboard updates on GPU
    def sweep(self) -> None:
        gsize = ((self._Nsites) // 2,)

        # two sublattice updates (0=even, 1=odd)
        for color in (0, 1):
            # launch kernel
            self.kernel(
                self.queue, gsize, None,
                self.spin_buffer,
                self.rng_buffer,
                np.float32(1 / self.T),
                np.float32(self.J),
                np.int32(self.L),
                np.int32(color)
            ).wait()

    # Can use _measure_cpu() or _measure_gpu() 
    def measure(self) -> MeasureData:
        # return self._measure_cpu()
        return self._measure_gpu()

    # Copy lattice from GPU and compute E, M on CPU
    def _measure_cpu(self) -> MeasureData:
        cl.enqueue_copy(self.queue, self.lattice, self.spin_buffer)
        self.queue.finish()

        E = 0.0
        E -= np.sum(self.lattice * np.roll(self.lattice, 1, axis=0))
        E -= np.sum(self.lattice * np.roll(self.lattice, 1, axis=1))
        M = float(np.sum(self.lattice))
        return MeasureData(E / self._Nsites, M / self._Nsites)
    
    # Measure energy and mag using parallel reduction
    def _measure_gpu(self) -> MeasureData:
        local_size = 256
        num_groups = (self._Nsites + local_size - 1) // local_size

        group_E = np.empty(num_groups, dtype=np.float32)
        group_M = np.empty(num_groups, dtype=np.float32)

        group_E_buf = cl.Buffer(self.context, self.mf.WRITE_ONLY, group_E.nbytes)
        group_M_buf = cl.Buffer(self.context, self.mf.WRITE_ONLY, group_M.nbytes)

        # Local storage to store and reduce a workgroups spin contributions in
        local_E_mem = cl.LocalMemory(np.dtype(np.float32).itemsize * local_size)
        local_M_mem = cl.LocalMemory(np.dtype(np.float32).itemsize * local_size)

        self.prg.compute_observables_reduce(
            self.queue, (num_groups * local_size,), (local_size,),
            self.spin_buffer,
            group_E_buf, group_M_buf,
            local_E_mem, local_M_mem,
            np.float32(self.J),
            np.int32(self.L)
        )

        cl.enqueue_copy(self.queue, group_E, group_E_buf)
        cl.enqueue_copy(self.queue, group_M, group_M_buf)
        self.queue.finish()

        E = np.sum(group_E) / (self._Nsites)
        M = np.sum(group_M) / (self._Nsites)
        return MeasureData(float(E), float(M))

    # Return a host copy of the lattice
    def copy_state(self) -> np.ndarray:
        cl.enqueue_copy(self.queue, self.lattice, self.spin_buffer)
        self.queue.finish()
        return self.lattice.copy()

    def _get_kernel_code(self) -> str:
        kernel_code = ""

        with open("algorithms/checkerboard.cl", "r") as file:
            kernel_code += file.read()

        with open("algorithms/compute_observables.cl", "r") as file:
            kernel_code += file.read()

        return kernel_code