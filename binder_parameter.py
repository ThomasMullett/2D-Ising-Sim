"""
binder_parameter.py

Compute Binder cumulant, susceptibility and heat capacity from run CSVs.

analyze_simulation_data(folder, algorithm_name) returns a sorted list of dicts
with derived observables for each (L, T) run. The script then plots the Binder
cumulant versus temperature for different L values, and saves the plot to
binder.png, and a displays a matplotlib window. 
"""


import os, glob, json
import numpy as np
from scipy.signal import correlate
import matplotlib.pyplot as plt

Tc = 2.269


# Read CSV files in a folder and compute ensemble averages and observables
def analyze_simulation_data(folder, algorithm_name):
    files = sorted(glob.glob(os.path.join(folder, f"{algorithm_name}_L*_T*.csv")))
    results = []

    for csv_file in files:
        meta_file = csv_file.replace(".csv", "_meta.json")
        if not os.path.exists(meta_file):
            continue

        # Load metadata
        with open(meta_file) as f:
            meta = json.load(f)
        L = meta["L"]
        T = meta["T"]

        # Load CSV (skip header line)
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        if data.ndim == 1:
            data = data.reshape(1, -1)

        N = L * L     
        energy = data[:, 2]
        mag = data[:, 3]

        e_mean = np.mean(energy)
        e2_mean = np.mean(energy**2)
        m_mean = np.mean(mag)
        m2_mean = np.mean(mag**2)
        m4_mean = np.mean(mag**4)

        # Derived thermodynamic observables (with standard definitions)
        chi = N * (m2_mean - m_mean**2) / T           # susceptibility
        C = N * (e2_mean - e_mean**2) / (T * T)       # heat capacity
        binder = 1 - m4_mean / (3 * m2_mean**2)       # Binder cumulant

        results.append({
            "L": L, "T": T,
            "m_mean": m_mean, "e_mean": e_mean,
            "chi": chi, "C": C, "binder": binder,
            "runtime": meta.get("runtime", np.nan)
        })

    # Sort by lattice size then temperature for consistent plotting
    results.sort(key=lambda x: (x["L"], x["T"]))
    return results

# results = analyze_simulation_data("data/WolffRun6", "WolffNumba")
results = analyze_simulation_data("data", "WolffNumba")

Ts = {}
binders = {}

for result in results:
    L = result["L"]
    if not (L in Ts):
        binders[L] = []
        Ts[L] = []

    binders[L].append(result["binder"])
    Ts[L].append(result["T"])

# Plot styling
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "text.usetex": False,
})

fig, ax = plt.subplots()

plt.tick_params(
    direction="in",
    length=3,
    width=1,
    which="both",
    right=True, top=True,
    labelsize=11,
)

# Plot each L as a separate series
for (L, T_list), binder_list in zip(Ts.items(), binders.values()):
    ax.plot(T_list, binder_list, "", label=f"L={L}", markersize=5, markerfacecolor='none',)

# Mark critical temperature for reference
ax.axvline(Tc, ls="--", color="black", label=f"Tc = {Tc}")
ax.set_xlabel("$k_B$T [J]", fontsize=12)
ax.set_ylabel("Binder cumulant", fontsize=12)

# Legend formatting for clarity
legend = plt.legend(
    frameon=True,       
    fontsize=10,
    loc="best",           
    fancybox=False,       
    edgecolor="black",   
)
legend.get_frame().set_linewidth(1.0)  
legend.get_frame().set_facecolor("white")  

# Save and show figure
plt.savefig("binder.png", dpi=300, bbox_inches="tight")
plt.show()