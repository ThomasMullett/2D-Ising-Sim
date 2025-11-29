"""
Docstring for utils.double_checkerboard_row_indices.py

Debugging script to check double checkerboard row indexes are correct
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

# Mirrors OpenCL kernel
def kernel(group_id_0, group_id_1, local_id_0, local_id_1, cache):

    local_row = local_id_1
    group_row = group_id_1
    group_col = group_id_0

    global_row = local_row + T*group_row
    global_col = 2*T*group_col + T*((group_row + color) % 2)
    global_idx = global_row*L + global_col

    b_offset = (global_row + color) % 2
    w_offset = not b_offset

    for x in range(-1, T+1):
        idx = global_row*L + (global_col + x + L) % L
        cache[get_spin_cache_index(x, local_row)] = black_value

        global_col_halo = (global_col + local_row + L) % L
        global_row_up = (T*group_row - 1 + L) % L
        global_row_down = (T * (group_row + 1)) % L

        cache[get_spin_cache_index(local_row, -1)] = boundary_value
        cache[get_spin_cache_index(local_row, T)] = boundary_value

def work_group(group_id_0, group_id_1):
    
    # Local memory
    cache = np.ones(((T+2)*(T+2))) * 0

    # Each spin updated
    for y in range(T):
        kernel(group_id_0, group_id_1, 0, y, cache)

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

new_board = np.zeros((L, L))
for x in range(0, L):
    for y in range(0, L):
        new_board[y, x] = board[y*L + x]

plt.imshow(new_board, cmap='hot')
plt.show()