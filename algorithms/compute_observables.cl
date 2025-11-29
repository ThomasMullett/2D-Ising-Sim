/*
algorithms.compute_observables.py

Contains a kernel to compute energy and magnetisation in parallel
This approuch is much faster than reading all the data back and
computing on CPU for large lattices
Can be used by all OpenCL implementations
*/


__kernel void compute_observables_reduce(
    __global const char *spins,
    __global float *group_E, __global float *group_M,
    __local float *local_E, __local float *local_M,
    const float J, const int L) {
        
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    int group_size = get_local_size(0);
    int total_sites = L * L;

    float E = 0.0f;
    float M = 0.0f;

    // Get individual spin contribution
    if (gid < total_sites) {
        int i = gid / L;
        int j = gid % L;

        int right = i * L + ((j + 1) % L);
        int down  = ((i + 1) % L) * L + j;

        char s = spins[gid];
        E = -J * s * (spins[right] + spins[down]);
        M = (float)s;
    }

    // Store partial results in local memory
    local_E[lid] = E;
    local_M[lid] = M;
    barrier(CLK_LOCAL_MEM_FENCE);

    // Parallel reduction within work group
    for (int stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            local_E[lid] += local_E[lid + stride];
            local_M[lid] += local_M[lid + stride];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // First thread in group writes result
    if (lid == 0) {
        int group_id = get_group_id(0);
        group_E[group_id] = local_E[0];
        group_M[group_id] = local_M[0];
    }
}