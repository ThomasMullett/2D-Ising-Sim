"""
times_vs_size_from_csv.py

Load timing table from times.csv and plot spin update time vs L for multiple algorithms.

Assumes times.csv has a header row and numeric columns; missing entries are
represented as '--' or empty and are converted to NaN.

Expects time.csv within the same folder and outputs scaling_plot.png,
and displays a matplotlib window
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

# Read entire CSV as strings to inspect headers (first two rows contain headers)
data = np.genfromtxt("times.csv", delimiter=",", dtype=str)

# Extract column headers (first row)
headers = data[0:2]

# Extract numeric data (convert '--' or '' to NaN)
table = np.genfromtxt(
    "times.csv",
    delimiter=",",
    dtype=float,
    skip_header=1,
    missing_values="--",
    filling_values=np.nan
)

# First column is lattice size L
L = table[:, 0]

# Columns and labels for the algorithms to plot (indices based on CSV layout)
algo_cols  = [4, 6, 8]
algo_names = ["Metro Numba", "C Numba", "C Numba Parallel"]

algo_cols  += [10, 12, 16]
algo_names += ["C CL", "DC CL Pair", "DC CL Row"]

algo_cols  += [14, 18]
algo_names += ["DC CL Pair k=100", "DC CL Row k=100"]

# Styling to look publication-ready
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Use Computer Modern
    "text.usetex": False,      # Avoid LaTeX dependency
})

fig, ax = plt.subplots()

ax.tick_params(
    direction="in",
    length=3,
    width=1,
    which="both",
    right=True, top=True,
    labelsize=11,
)


# Plot each algorithm's t_spin vs L (skipping missing entries)
for name, col in zip(algo_names, algo_cols):
    y = table[:, col]
    mask = ~np.isnan(y)

    if mask.sum() == 0:
        continue  # skip if no data available

    ax.plot(np.array(L[mask]), np.array(y[mask]), "-", label=name)


ax.set_xscale("log", base=2)
ax.set_yscale("log", base=10)

ax.set_xlabel("L", fontsize=12)
ax.set_ylabel("$t_{spin}$ [s]", fontsize=12)
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.set_ylim(1.1e-12, 9.9e-7)

# Legend formatting for clear printing
legend = plt.legend(
    frameon=True,
    fontsize=10,
    loc="lower left",           
    fancybox=False,        
    edgecolor="black",   
)
legend.get_frame().set_linewidth(1.0)   
legend.get_frame().set_facecolor("white")  


# Inset to zoom into a region with large L where tiny differences matter
axins = inset_axes(ax, width="25%", height="35%", loc="upper right", borderpad=1)

for name, col in zip(algo_names, algo_cols):
    y = table[:, col]
    mask = ~np.isnan(y)

    if mask.sum() == 0:
        continue  # skip if no data available

    axins.plot(
        np.array(L[mask]),
        np.array(y[mask]),
        "-"
    )

axins.set_xlim(21000, 34000)
axins.set_ylim(3.5e-11, 6e-11)
axins.set_yscale("log")
axins.tick_params(axis="y", which="minor", labelleft=False)

axins.tick_params(
    axis="both", which="both",
    labelsize=8,
    direction="in",
    right=True, top=True,
    length=2
)

# Draw a rectangle linking inset to the main axes
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.3")

# Save high-resolution figure and show
plt.savefig("scaling_plot.png", dpi=300, bbox_inches="tight")
plt.show()