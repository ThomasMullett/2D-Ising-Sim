"""
runner.py

This is the main script to run the simulations.
It expects a config.json file with simulation configuration, loads it,
performs the simulations and outputs a .csv and _meta.json file with
simulation output into the requested folder.
"""


import json
import os
from datetime import datetime
import time
import numpy as np
import algorithms as algos
from utils.lattice_animator import animate_lattice, show_lattice
from utils.config_parser import ConfigData, parse_temperatures_new, parse_config, ensure_dir
from dataclasses import dataclass
import csv


@dataclass
class SimulationData:
    sweep_indices: np.ndarray
    energies: np.ndarray
    magnetizations: np.ndarray
    mean_sweep_time: float
    total_runtime: float    


def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results_csv(path, sim_data: SimulationData):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "sweep", "energy", "magnetization"])
        for idx, (s, E, M) in enumerate(zip(sim_data.sweep_indices, sim_data.energies, sim_data.magnetizations)):
            writer.writerow([idx, s, E, M])


def run_simulation(UpdaterClass: type[algos.IsingUpdater], L: int, T: float,
                   eq_sweeps: int, meas_sweeps: int, meas_interval:int, seed:int,
                   params: dict, animate: bool=False, anim_interval: int=10, anim_save: str="animation.mp4") -> SimulationData:
    
    updater = UpdaterClass(L=L, T=T, seed=seed, **params)
    updater.initialize(hot=False) # Changes whether all spins start aligned or randomly distributed

    # Equilibration
    for _ in range(eq_sweeps):
        updater.sweep()

    animation_frames = []

    num_measurements = meas_sweeps // meas_interval
    energies = np.zeros(num_measurements)
    mags = np.zeros(num_measurements)
    sweeps = np.arange(0, meas_sweeps, meas_interval)
    sweep_times = np.zeros(meas_sweeps)

    meas_idx = 0
    for sweep_idx in range(meas_sweeps):
        # Animation update
        if animate and sweep_idx % anim_interval == 0:
            animation_frames.append(updater.copy_state())

        # Perform one algorithm sweep
        t0 = time.perf_counter()
        updater.sweep()
        t1 = time.perf_counter()
        sweep_times[sweep_idx] = t1 - t0

        # Write current lattice observables to arrays
        if sweep_idx % meas_interval == 0:
            data = updater.measure()
            energies[meas_idx] = data.energy
            mags[meas_idx] = data.magnetization
            meas_idx += 1

    # show_lattice(updater.copy_state())

    # Save animation if requested
    if animate:
        animate_lattice(animation_frames, anim_interval, anim_save, fps=5)
        
    return SimulationData(sweeps, energies, mags, float(np.mean(sweep_times)), float(np.sum(sweep_times)))


# Loads config.json, performs the simulations and saves the output
def main(config_path: str):

    with open(config_path, "r") as f:
        config = json.load(f)

    cd = parse_config(config)

    # Total runs the simulation will run for
    runs = len(cd.algorithms) * len(cd.lattice_sizes) * len(cd.temperatures) * len(cd.seeds)
    run = 0

    total_start_time = time.perf_counter()

    for algorithm in cd.algorithms:
        
        updater: type[algos.IsingUpdater] = algos.IsingUpdater # Needed for type checking

        if algorithm["name"] in algos.algos_dict:
            updater = algos.algos_dict[algorithm["name"]]
        else:
            print(f"[Warning] Algorithm {algorithm['name']} not found in algorithms module. Skipping.")
            continue

        # Algorithm specific parameters
        params = algorithm["params"]

        for L in cd.lattice_sizes:
            for T in cd.temperatures:
                for seed in cd.seeds:
                    run += 1
                    run_id = f"{algorithm['name']}_L{L}_T{T:.4f}_seed{seed}_{timestamp()}"
                    print(f"\n=== Running {run}/{runs} : {run_id} ===")

                    start_time = time.perf_counter()
                    results = run_simulation(
                        updater, L, T,
                        cd.eq_sweeps, cd.meas_sweeps,
                        cd.meas_interval, seed, params,
                        cd.animate, cd.animation_interval, os.path.join(cd.animation_save, f"{run_id}.mp4")
                    )
                    end_time = time.perf_counter()

                    # Output result .csv file
                    base = os.path.join(cd.output_dir, run_id)
                    save_results_csv(base + ".csv", results)

                    # Metadata json file
                    metadata = {
                        "algorithm": algorithm["name"],
                        "L": L,
                        "T": T,
                        "seed": seed,
                        "total_runtime": end_time - start_time,
                        "runtime": results.total_runtime,
                        "mean_sweep_time": results.mean_sweep_time,
                        "eq_sweeps": cd.eq_sweeps,
                        "meas_sweeps": cd.meas_sweeps,
                        "meas_interval": cd.meas_interval,
                        "average_mag": np.mean(results.magnetizations),
                        "average_abs_mag": np.mean(np.abs(results.magnetizations)),
                        "average_energy": np.mean(results.energies),
                        "params": params
                    }

                    with open(base + "_meta.json", "w") as f:
                        json.dump(metadata, f, indent=2)

                    # Output averages to check simulations are running corretly
                    print(f"Average energy {np.mean(results.energies):.4f}")
                    print(f"Average magnetisation {np.mean(results.magnetizations):.4f}")
                    print(f"Runtime {(end_time - start_time):.4f}")
    
    # Print total runtime. This is not spin-flip runtime
    total_end_time = time.perf_counter()
    runtime = total_end_time - total_start_time
    print(f"\nAll simulations complete in {runtime:.4f}, average runtime {(runtime / runs):.4f}")


if __name__ == "__main__":
    main("config.json")