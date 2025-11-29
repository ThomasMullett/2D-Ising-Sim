/*
algorithms.double_checkerboard.py

Contains the kernels for both row and spin pair double checkerboard implementations
*/


// A is the MWC multiplier constant, chosen for good statistical properties
#define MWC_A 4294883355U // (0xFFFE0000UL / 2^32) + 1, Marsaglia's constant

typedef struct {
    uint x; // Current MWC state (multiplier part)
    uint c; // Current MWC carry
} mwc_state_t;

// MWC64X, state must be passed by __private pointer
uint MWC64X(__private mwc_state_t *state) {
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



// Gets location for tile spin coords from a cache that allows a halo of spins
int get_spin_cache_index(int x, int y, int TSize) {
    return (x+1) + (y+1)*(TSize + 2);
}



// Spin pair implementation
__kernel void double_checkerboard(
        __global char *spins, __global mwc_state_t *rng_state,
        __local char *spin_cache, // (TSize + 2, TSize + 2)
        __constant const float boltz[5],
        const float beta, const float J, const int L, const int TSize, const int k_max, const int color) {

    // Figure out position, see utils.double_checkerboard_indices.py for more details
    int local_row = get_local_id(1);
    int local_col = get_local_id(0);

    int group_row = get_group_id(1);
    int group_col = get_group_id(0);

    int global_row = local_row + TSize*group_row;
    int global_col = 2*local_col + 2*TSize*group_col + TSize*((group_row + color) % 2);
    int global_idx = global_row*L + global_col;
    int global_idx_p = global_row*L + (global_col + 1) % L;

    // Figure out which spin is which colour
    int b_offset = (global_row + color) % 2;
    int w_offset = !b_offset;

    int local_b_col = 2*local_col + b_offset;
    int local_w_col = 2*local_col + w_offset;

    // Load spin pair
    spin_cache[get_spin_cache_index(2*local_col, local_row, TSize)] = spins[global_idx];
    spin_cache[get_spin_cache_index(2*local_col + 1, local_row, TSize)] = spins[global_idx_p];

    __private mwc_state_t state1 = rng_state[global_idx];
    __private mwc_state_t state2 = rng_state[global_idx_p];

    // Load halo spin if workitem is on the boundaries
    if (local_row == 0) {
        int global_row_up = (global_row - 1 + L) % L; // Up
        spin_cache[get_spin_cache_index(2*local_col, -1, TSize)] = spins[global_row_up*L + global_col];
        spin_cache[get_spin_cache_index(2*local_col + 1, -1, TSize)] = spins[global_row_up*L + global_col + 1];
    }
    if (local_row == TSize-1) {
        int global_row_down = (global_row + 1) % L; // Down
        spin_cache[get_spin_cache_index(2*local_col, TSize, TSize)] = spins[global_row_down*L + global_col];
        spin_cache[get_spin_cache_index(2*local_col + 1, TSize, TSize)] = spins[global_row_down*L + global_col + 1];
    }
    if (local_col == 0) {
        int global_col_left = (global_col - 1 + L) % L; // Left
        spin_cache[get_spin_cache_index(-1, local_row, TSize)] = spins[global_row*L + global_col_left];
    }
    if (local_col == TSize/2-1) {
        int global_col_right = (global_col + 2) % L; // Right
        spin_cache[get_spin_cache_index(TSize, local_row, TSize)] = spins[global_row*L + global_col_right];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over multi-hit parameter
    for (int k=0; k<k_max; k++) {
        // Update the black spin
        int index = get_spin_cache_index(local_b_col, local_row, TSize);
        char s = spin_cache[index];
        char nb_sum = ( spin_cache[get_spin_cache_index(local_b_col, local_row-1, TSize)] + 
                        spin_cache[get_spin_cache_index(local_b_col, local_row+1, TSize)] + 
                        spin_cache[get_spin_cache_index(local_b_col-1, local_row, TSize)] + 
                        spin_cache[get_spin_cache_index(local_b_col+1, local_row, TSize)]);
        float dE = 2.0f * J * s * nb_sum;
        int dE_idx = (nb_sum * s + 4) / 2;

        float r = uniform_01(MWC64X(&state1));
        if (dE <= 0.0f || r < boltz[dE_idx])
            spin_cache[index] = -s;

        // Ensure all black spins are updated
        barrier(CLK_LOCAL_MEM_FENCE);

        // Update the white spin
        index = get_spin_cache_index(local_w_col, local_row, TSize);
        s = spin_cache[index];
        nb_sum = (  spin_cache[get_spin_cache_index(local_w_col, local_row-1, TSize)] + 
                    spin_cache[get_spin_cache_index(local_w_col, local_row+1, TSize)] + 
                    spin_cache[get_spin_cache_index(local_w_col-1, local_row, TSize)] + 
                    spin_cache[get_spin_cache_index(local_w_col+1, local_row, TSize)]);
        dE = 2.0f * J * s * nb_sum;
        dE_idx = (nb_sum * s + 4) / 2;

        r = uniform_01(MWC64X(&state2));
        if (dE <= 0.0f || r < boltz[dE_idx])
            spin_cache[index] = -s;

        // Ensure all white spins are updated
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back
    spins[global_idx] = spin_cache[get_spin_cache_index(2*local_col, local_row, TSize)];
    spins[global_idx_p] = spin_cache[get_spin_cache_index(2*local_col+1, local_row, TSize)];

    rng_state[global_idx] = state1;
    rng_state[global_idx_p] = state2;
}



// Row implementation
__kernel void double_checkerboard_row(
        __global char *spins, __global mwc_state_t *rng_state,
        __local char *spin_cache, // (TSize + 2, TSize + 2)
        __constant const float boltz[5],
        const float beta, const float J, const int L, const int TSize, const int k_max, const int color) {
    
    // Figure out position, see utils.double_checkerboard_row_indices.py for details
    int local_row = get_local_id(1);
    // int local_col = get_local_id(0);

    int group_row = get_group_id(1);
    int group_col = get_group_id(0);

    int global_row = local_row + TSize*group_row;
    int global_col = (2*TSize*group_col + TSize*((group_row + color) % 2));
    int global_idx = global_row*L + global_col;

    int b_offset = (global_row + color) % 2;
    int w_offset = !b_offset;

    // Load all spins from the row + halo
    for (int x=-1; x<=TSize; x++) {
        int idx = global_row*L + (global_col + x + L) % L;
        spin_cache[get_spin_cache_index(x, local_row, TSize)] = spins[idx];
    }

    __private mwc_state_t state1 = rng_state[global_idx];

    // Vertical Halo - each row loads the colum of top and bottom spin correspodning to row number
    // This is done to equally distribute work a coalesce global memory reads
    int global_col_halo = (global_col + local_row) % L;

    int global_row_up = (TSize*group_row - 1 + L) % L;
    spin_cache[get_spin_cache_index(local_row, -1, TSize)] = spins[global_row_up*L + global_col_halo];

    int global_row_down = (TSize * (group_row + 1)) % L;
    spin_cache[get_spin_cache_index(local_row, TSize, TSize)] = spins[global_row_down*L + global_col_halo];

    barrier(CLK_LOCAL_MEM_FENCE);

    // Loop over multi-hit parameter
    for (int k=0; k<k_max; k++) {
        // Update the black spins
        for (int x=b_offset; x<TSize; x+=2) {
            int index = get_spin_cache_index(x, local_row, TSize);
            char s = spin_cache[index];
            char nb_sum = ( spin_cache[get_spin_cache_index(x, local_row-1, TSize)] + 
                            spin_cache[get_spin_cache_index(x, local_row+1, TSize)] + 
                            spin_cache[get_spin_cache_index(x-1, local_row, TSize)] + 
                            spin_cache[get_spin_cache_index(x+1, local_row, TSize)]);
            float dE = 2.0f * J * s * nb_sum;
            int dE_idx = (nb_sum * s + 4) / 2;

            float r = uniform_01(MWC64X(&state1));
            if (dE <= 0.0f || r < boltz[dE_idx])
                spin_cache[index] = -s;
        }

        // Ensure all black spins are updated
        barrier(CLK_LOCAL_MEM_FENCE);

        // Update the white spins
        for (int x=w_offset; x<TSize; x+=2) {
            int index = get_spin_cache_index(x, local_row, TSize);
            char s = spin_cache[index];
            char nb_sum = ( spin_cache[get_spin_cache_index(x, local_row-1, TSize)] + 
                            spin_cache[get_spin_cache_index(x, local_row+1, TSize)] + 
                            spin_cache[get_spin_cache_index(x-1, local_row, TSize)] + 
                            spin_cache[get_spin_cache_index(x+1, local_row, TSize)]);
            float dE = 2.0f * J * s * nb_sum;
            int dE_idx = (nb_sum * s + 4) / 2;

            float r = uniform_01(MWC64X(&state1));
            if (dE <= 0.0f || r < boltz[dE_idx])
                spin_cache[index] = -s;
        }

        // Ensure all the white spins are updated
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write back
    for (int x=0; x<TSize; x++) {
        spins[global_row*L + (global_col + x + L)%L] = spin_cache[get_spin_cache_index(x, local_row, TSize)];
    }

    rng_state[global_idx] = state1;
}