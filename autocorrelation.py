"""
autocorrelation.py

Compute integrated autocorrelation times from simulation CSV output
and plot tau_int vs temperature for different algorithms.

Expects *.csv files and corresponding *_meta.json inside a folder
and outputs auto.png, , and a displayed matplotlib window.
"""


import os, glob, json
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import LogFormatterMathtext


# Estimate the integrated autocorrelation time of a 1D time series x with windowing parameter c
def autocorrelation_time(x, c=5):
    x = np.asarray(x, dtype=float)
    x -= np.mean(x)     # subtract mean to compute connected correlation
    n = len(x)
    var = np.var(x)
    if var == 0:
        return 0.0

    # Normalized autocorrelation (via FFT-based correlation)
    acf = correlate(x, x, mode='full')[n-1:] / (var * np.arange(n, 0, -1))
    acf /= acf[0]  # Normalize so that C(0)=1

    # Integrate until first zero crossing or Sokal window criterion t > c * tau_int
    tau_int = 0.5
    for t in range(1, n):
        if acf[t] <= 0:
            break
        tau_int += acf[t]
        if t > c * tau_int:
            break
    return tau_int


# Collect all data files matching the pattern
# datas = glob.glob("data/AutoRuns/*seed42*.csv")
datas = glob.glob("data/*.csv")

autos = {}
Ts = {}
Tc = 2.269


for data_file_name in datas:
    # Expect a metadata JSON file with the same base name + "_meta.json"
    meta_file = data_file_name.replace(".csv", "_meta.json")
    if not os.path.exists(meta_file):
        continue

    # Load metadata
    with open(meta_file) as f:
        meta = json.load(f)

    data = np.loadtxt(data_file_name, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Column indexing: adjust here if CSV layout changes
    energy = data[:, 2]
    mag = data[:, 3]

    tau_m = autocorrelation_time(mag)

    # Build a human-readable algorithm label from metadata
    algo = meta["algorithm"]
    if "k" in meta["params"]:
        algo = f"k={meta['params']['k']}"

    if algo == "WolffNumba":
        algo = "Wolff"
    elif algo == "CheckerboardOpenCL":
        algo = "k=1"

    if not (algo in autos):
        autos[algo] = []
        Ts[algo] = []

    autos[algo].append(tau_m)
    Ts[algo].append(meta["T"])


# Plot styling: use serif math font that resembles publication style
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",  # Use Computer Modern
    "text.usetex": False,      # Avoid LaTeX dependency
})

fig, ax = plt.subplots()

# Ticks and general axis styling for a cleaner look
ax.tick_params(
    direction="in",  # ticks point inward
    length=3,
    width=1,
    which="both",
    right=True, top=True,
    labelsize=11,
)

# Plot each algorithm's tau_int vs T
for algo, T_list in Ts.items():
    auto_list = autos[algo]
    ax.plot(T_list, auto_list, "-", label=f"{algo}")

ax.axvline(Tc, ls="--", color="black", label=f"Tc = {Tc}")    # Mark the critical temperature for reference
ax.set_xlabel("$k_B$T [J]", fontsize=12)
ax.set_ylabel(r"$\tau_{int}$", fontsize=14)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(LogFormatterMathtext())

# Legend styling to make it printer-friendly
legend = ax.legend(
    frameon=True,
    fontsize=10,
    loc="upper right",
    fancybox=False,
    edgecolor="black",
)
legend.get_frame().set_linewidth(1.0)
legend.get_frame().set_facecolor("white")


# Add an inset zoom around Tc to show critical slowing down region
axins = inset_axes(ax, width="35%", height="35%", loc="upper left", borderpad=3)

for algo, T_list in Ts.items():
    auto_list = autos[algo]
    axins.plot(T_list, auto_list, "-")

axins.set_xlim(2.25, 2.29)
axins.set_ylim(400, 4700)
axins.set_yscale("log")
axins.yaxis.set_major_formatter(LogFormatterMathtext())
axins.tick_params(axis="y", which="minor", labelleft=False)

axins.tick_params(
    axis="both", which="both",
    labelsize=8,
    direction="in",
    right=True, top=True,
    length=2
)

# Draw connecting box between main axes and inset
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.3")

# Save high-resolution figure and display
plt.savefig("auto.png", dpi=300, bbox_inches="tight")
plt.show()