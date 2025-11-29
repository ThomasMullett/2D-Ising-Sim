"""
utils.config_parser.py

Contains to helper functions for parsing the simualation
config.json file used by runner.py
"""


from dataclasses import dataclass
import numpy as np
import os


@dataclass
class ConfigData:
    output_dir: str

    eq_sweeps: int
    meas_sweeps: int
    meas_interval: int

    animate: bool
    animation_interval: int
    animation_save: str

    temperatures: list[float]
    lattice_sizes: list[int]
    seeds: list[int]
    algorithms: dict

# Same as os.makedirs(exist_ok=True)
def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


# Allows for temperature range or set values, use range if provided
def parse_temperatures_new(config: dict) -> list[float]:
    if "temperature_range" in config:
        tr = config["temperature_range"]
        return np.linspace(tr["start"], tr["stop"], tr["num"]).tolist()
    else:
        return config["temperatures"]
    

# Returns config file as dataclass and ensures dirs exist
def parse_config(config: dict) -> ConfigData:
    output_dir = config.get("output_dir", "data/")
    ensure_dir(output_dir)

    eq_sweeps = config["eq_sweeps"]
    meas_sweeps = config["meas_sweeps"]
    meas_interval = config["meas_interval"]

    animate = config["animate"]
    animation_interval = config["animation_interval"]
    animation_save = config.get("animation_save", "animations/")
    ensure_dir(animation_save)

    temperatures = parse_temperatures_new(config)
    lattice_sizes = config["lattice_sizes"]
    seeds = config["seeds"]
    algorithms = config["algorithms"]

    return ConfigData(output_dir, eq_sweeps, meas_sweeps, meas_interval,
                      animate, animation_interval, animation_save,
                      temperatures, lattice_sizes, seeds, algorithms)
