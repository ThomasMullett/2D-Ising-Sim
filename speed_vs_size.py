"""
speed_vs_size.py

Plot runtime vs lattice size from JSON metadata files.

Searches for *.json files in a folder and groups runtimes
by algorithm label. Plots log-log (here log(L) vs log(runtime)).

This is not used for the paper's plot, that uses times_vs_size_from_csv.py
"""


import matplotlib.pyplot as plt
import glob
import json
import numpy as np

# metadatas = glob.glob("data/SpeedVsSizeRun1/*.json")
# metadatas = glob.glob("data/BigDump/*.json")
metadatas = glob.glob("data/*.json")

speeds = {}
sizes = {}


for file_name in metadatas:
    # Load each JSON file containing run metadata (algorithm, L, runtime, ...)
    with open(file_name, "r") as f:
        data = json.load(f)

    algo = data["algorithm"]
    if not (algo in speeds):
        speeds[algo] = []
        sizes[algo] = []

    # Store runtime and lattice size for the algorithm
    size = data["L"]
    speeds[algo].append(data["runtime"])
    sizes[algo].append(size)


for (algo, size_list), speed_list in zip(sizes.items(), speeds.values()):
    # Convert to arrays to make sorting easier
    size_list = np.array(size_list)
    speed_list = np.array(speed_list)

    # Sort by size
    sort_idx = np.argsort(size_list)
    size_list = size_list[sort_idx]
    speed_list = speed_list[sort_idx]
    
    plt.plot(np.log(size_list), np.log(speed_list), "", label=f"{algo}")

plt.title("Runtime vs lattice size for different algorithms")
plt.xlabel("Lattice size")
plt.ylabel("Runtime (s)")
plt.legend()
plt.show()