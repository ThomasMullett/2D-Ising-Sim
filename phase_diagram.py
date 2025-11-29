"""
phase_diagram.py

Build phase diagram: read magnetisation from CSV runs and compare to Onsager solution.

Expects CSV runs *.csv within a folder with accompanying *_meta.json files that
contain algorithm and temperature 'T'. Outputs phase_diagram.png,
and displays a matplotlib window
"""


import matplotlib.pyplot as plt
import glob
import json
import numpy as np
import os

# datas = glob.glob("data/DCRun5/*.csv")
datas = glob.glob("data/*.csv")

mags = {}
mag_err = {}
Ts = {}
Tc = 2.269


for data_file_name in datas:
    # Expect metadata JSON next to each CSV file
    meta_file = data_file_name.replace(".csv", "_meta.json")
    if not os.path.exists(meta_file):
        continue

    # Load metadata
    with open(meta_file) as f:
        meta = json.load(f)

    # Load CSV numeric columns, skipping header
    data = np.loadtxt(data_file_name, delimiter=',', skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    # Column indices: energy and magnetisation (adjust if CSV format changes)
    energy = data[:, 2]
    mag = data[:, 3]

    algo = meta["algorithm"]
    if not (algo in mags):
        mags[algo] = []
        mag_err[algo] = []
        Ts[algo] = []

    # Store mean and std of magnetisation for plotting with errorbars
    mags[algo].append(float(np.mean(mag)))
    mag_err[algo].append(float(np.std(mag)))
    Ts[algo].append(meta["T"])


# Compute analytical Onsager magnetisation curve for 2D Ising (for T < Tc)
exact_Ts = np.linspace(1, 3.5, 10000)
exact_mags = np.pow(1 - np.pow(np.sinh(2/exact_Ts)*np.sinh(2/exact_Ts) , -2), 1/8)


# Plot setup: two stacked axes (main + residuals)
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

fig, (main_ax, res_ax) = plt.subplots(
    2, 1, figsize=(5, 4), sharex=True,
    gridspec_kw={"height_ratios": [1, 1], "hspace": 0}
)

# Apply consistent tick styling to both axes
for ax in [main_ax, res_ax]:
    ax.tick_params(
        direction="in",  # ticks point inward
        length=3,
        width=1,
        which="both",
        right=True, top=True,
        labelsize=11,
    )
    ax.xaxis.set_tick_params(labelbottom=True)
    ax.yaxis.set_tick_params(labelleft=True)

# Plot measured magnetisation with errorbars
for (algo, T_list), mags_list, mag_err_list in zip(Ts.items(), mags.values(), mag_err.values()):
    main_ax.errorbar(T_list, mags_list, yerr=mag_err_list, xerr=0, fmt="o", label=f"Data", markersize=5)

# Plot Onsager exact curve and a vertical line at Tc
main_ax.plot(exact_Ts, exact_mags, label="Onsager", color="black")
main_ax.axvline(Tc, ls="--", color="black", label=f"Tc = {Tc}")
main_ax.axhline(0, ls="-.", color="black", alpha=0.5)
main_ax.set_ylabel("Magnetisation", fontsize=12)

# Legend formatting
legend = main_ax.legend(
    frameon=True,          
    fontsize=10,
    loc="best",          
    fancybox=False,      
    edgecolor="black",    
)
legend.get_frame().set_linewidth(1.0) 
legend.get_frame().set_facecolor("white") 

# Compute normalized residuals (data - theory) / error and plot in lower panel
for algo, T_list in Ts.items():
    mags_list = mags[algo]

    # Interpolate exact mags for all T (zero above Tc)
    exact_interp = np.zeros_like(T_list)
    mask = np.array(T_list) < Tc
    exact_interp[mask] = (1 - np.sinh(2 / np.array(T_list)[mask]) ** -4) ** (1 / 8)

    # Normalized residuals relative to measurement uncertainty
    residuals = (mags_list - exact_interp) / np.array(mag_err[algo])

    res_ax.plot(T_list, residuals, "o", markersize=5)

res_ax.axhline(0, color="black", linestyle="-.", alpha=0.5)
res_ax.axvline(Tc, ls="--", color="black", label=f"Tc = {Tc}")
res_ax.set_xlabel("$k_B$T [J]", fontsize=12)
res_ax.set_ylabel("Norm. Residuals", labelpad=4, fontsize=12)

# Save and show final figure
plt.savefig("phase_diagram.png", dpi=300, bbox_inches="tight")
plt.tight_layout()
plt.show()