/*
algorithms.checkerboard.cl

Contains a MWC64X implemenation and a regular Metropolis updator
This is the kernel used by algorithms.checkerboard_opencl.py
*/


// A is the MWC multiplier constant, chosen for good statistical properties
#define MWC_A 4294883355U // (0xFFFE0000UL / 2^32) + 1, Marsaglia's constant

typedef struct {
    uint x; // Current MWC state (multiplier part)
    uint c; // Current MWC carry
} mwc_state_t;

// MWC64X, state must be passed by __global pointer
uint MWC64X(__global mwc_state_t *state) {
    uint x = state->x;
    uint c = state->c;

    // 1. Calculate result with xor of state components
    uint res = x ^ c; 

    // 2. Step the MWC generator (x_n = A * x_{n-1} + c_{n-1})
    uint hi = mul_hi(x, MWC_A); 
    x = x * MWC_A + c; 
    c = hi + (x < c); 

    // 3. Update the state in-place
    state->x = x;
    state->c = c;
    
    return res;
}

// Converts a 32-bit integer to a float in [0.0, 1.0)
float uniform_01(uint u) {
    // 2.3283064365386962890625e-10f is 1 / 2^32, multiplication is faster than division
    return (float)u * 2.3283064365386962890625e-10f; 
}



// Unused lcg random number generators
inline ulong lcg64(ulong *state) {
    *state = (*state * 6364136223846793005UL + 1UL);
    return *state;
}

inline float rand_uniform_lcg_float(ulong *state) {
    return (float)(lcg64(state) >> 40) * (1.0f / 16777216.0f); // 24-bit mantissa
}

inline float rand_uniform_lcg(ulong *state) {
    ulong a = 2862933555777941757;
    ulong c = 1442695040888963407;
    ulong m = 18446744073709551615;
    *state = a * *state + c;
    return (float)((double)(*state) / m);
}



__kernel void checkerboard( __global char *spins, __global mwc_state_t *rng_state,
                            const float beta, const float J, const int L, const int color) {
                                
    // Get position in spin lattice
    int gid = get_global_id(0);
    int halfL = L / 2;
    int i = gid / halfL;
    int j = (gid % halfL) * 2 + ((i + color) & 1);

    int idx = i * L + j;
    __global mwc_state_t *state = &rng_state[idx];

    // Periodic neighbour indices
    int up    = ((i - 1 + L) % L) * L + j;
    int down  = ((i + 1) % L) * L + j;
    int left  = i * L + ((j - 1 + L) % L);
    int right = i * L + ((j + 1) % L);

    // Calculate dE
    char s = spins[idx];
    float nb_sum = (float)(spins[up] + spins[down] + spins[left] + spins[right]);
    float dE = 2.0f * J * s * nb_sum;

    // Update spin
    float r = uniform_01(MWC64X(state));
    if (dE <= 0.0f || r < exp(-beta * dE))
        spins[idx] = -s;

    //rng_state[idx] = state; // Only needed if rng does not modify state in-place
}