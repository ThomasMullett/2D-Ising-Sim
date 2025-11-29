"""
utils.double_checkerboard_indices.py

Debugging script to check double checkerboard indexes are correct
This script is not used by runner.py
"""


import numpy as np
import matplotlib.pyplot as plt

B = 4
T = 16
L = int(B*T)

color = 0

none_value = 0
black_value = 1
white_value = 2
boundary_value = 3

board = np.ones((L**2)) * none_value

def get_spin_cache_index(x, y):
    return (x+1) + (y+1)*(T + 2)

# Mirrors the OpenCL kernel
def kernel(group_id_0, group_id_1, local_id_0, local_id_1, cache):

    global_row = local_id_1 + T*group_id_1
    global_col = 2*local_id_0 + 2*T*group_id_0 + T*((group_id_1 + color) % 2)

    global_idx = global_row*L + global_col
    global_idx_p = global_row*L + (global_col + 1) % L

    global_idx_b = global_idx if (global_row + color) % 2 else global_idx_p
    global_idx_w = global_idx_p if (global_row + color) % 2 else global_idx

    b_offset = (global_row + color) % 2
    w_offset = not b_offset

    local_col_b = 2*local_id_0 + b_offset
    lcoal_col_w = 2*local_id_0 + w_offset

    # board[global_idx_b] = black_value
    # board[global_idx_w] = white_value
    cache[get_spin_cache_index(local_col_b, local_id_1)] = black_value
    cache[get_spin_cache_index(lcoal_col_w, local_id_1)] = white_value

    if (local_id_1 == 0):
        global_row_up = (global_row - 1 + L) % L
        # board[global_row_up*L + global_col] = boundary_value
        # board[global_row_up*L + global_col + 1] = boundary_value
        cache[get_spin_cache_index(2*local_id_0, -1)] = boundary_value
        cache[get_spin_cache_index(2*local_id_0+1, -1)] = boundary_value

    if (local_id_1 == T - 1):
        global_row_down = (global_row + 1) % L
        # board[global_row_down*L + global_col] = boundary_value
        # board[global_row_down*L + global_col + 1] = boundary_value
        cache[get_spin_cache_index(2*local_id_0, T)] = boundary_value
        cache[get_spin_cache_index(2*local_id_0+1, T)] = boundary_value

    if (local_id_0 == 0):
        global_col_left = (global_col - 1 + L) % L
        # board[global_row*L + global_col_left] = boundary_value
        cache[get_spin_cache_index(2*local_id_0-1, local_id_1)] = boundary_value

    if (local_id_0 == int(T/2) - 1):
        global_col_right = (global_col + 2) % L
        # board[global_row*L + global_col_right] = boundary_value
        cache[get_spin_cache_index(2*local_id_0+2, local_id_1)] = boundary_value

def work_group(group_id_0, group_id_1):
    
    # Local memory
    cache = np.ones(((T+2)*(T+2))) * 0
    # cache[get_spin_cache_index(-1, -1)] = 0
    # cache[get_spin_cache_index(T, -1)] = 0
    # cache[get_spin_cache_index(-1, T)] = 0
    # cache[get_spin_cache_index(T, T)] = 0

    # Each spin updated
    for x in range(int(T/2)):
        for y in range(T):
            kernel(group_id_0, group_id_1, x, y, cache)

    global_x = 2*T*group_id_0 + T*((group_id_1 + color) % 2)
    global_y = group_id_1*T

    # global_x = group_id_0*T
    # global_y = group_id_1*T

    # Loop over tile and halo
    for x in range(-1, T+1, 1):
        for y in range(-1, T+1, 1):
            local_x = (global_x + x) % L
            local_y = (global_y + y) % L
            board[local_y*L + local_x] += cache[get_spin_cache_index(x, y)]# * (group_id_0 + group_id_1*2) * 100
            # print(f"{x},{y}   {local_x},{local_y}   {local_y*L + local_x}   {board[local_y*L + local_x]}")

for x in range(int(B/2)):
    for y in range(B):
        work_group(x, y)

# cache = np.zeros(((T+2)**2, 2))
# for x in range(-1, T+1):
#     for y in range(-1, T+1):
#         cache[get_spin_cache_index(x, y)] = np.array([x, y])

# print(cache.reshape((T+2, T+2, 2)))

new_board = np.zeros((L, L))
for x in range(0, L):
    for y in range(0, L):
        new_board[y, x] = board[y*L + x]

plt.imshow(new_board, cmap='hot')
plt.show()