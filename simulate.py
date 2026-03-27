"""
simulate.py
-----------
Industrial production line simulator using SimPy.

Generates realistic multi-variate sensor data with:
- Stochastic degradation patterns
- Fault injection with configurable MTBF
- 8 sensor channels per machine
- 10,000+ timestamped data points over 30 simulated days

Usage:
    python src/simulate.py
    python src/simulate.py --machines 5 --days 60 --output data/raw/sensors.csv
"""

import simpy
import numpy as np
import pandas as pd
import argparse
import os
from dataclasses import dataclass, field
from typing import Generator


# ─── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class MachineConfig:
    """Configuration for a single production machine."""
    machine_id: str
    mtbf_hours: float = 200.0          # Mean Time Between Failures
    mttr_hours: float = 4.0            # Mean Time To Repair
    degradation_rate: float = 0.002    # Health decay per hour under load
    sensor_noise: float = 0.02         # Gaussian noise std on sensor readings
    sampling_interval: float = 0.1     # Hours between readings (~6 min)


@dataclass
class SimulationConfig:
    n_machines: int = 3
    sim_days: int = 30
    random_seed: int = 42
    output_path: str = "data/raw/sensors.csv"
    machines: list = field(default_factory=list)

    def __post_init__(self):
        if not self.machines:
            self.machines = [
                MachineConfig(machine_id=f"M{i+1:02d}")
                for i in range(self.n_machines)
            ]


# ─── Sensor Physics Model ──────────────────────────────────────────────────────

class SensorModel:
    """
    Physics-inspired sensor model.
    Each channel degrades differently as machine health decays.
    """

    CHANNELS = [
        "temperature",   # °C  — rises with degradation
        "vibration",     # g   — spikes near failure
        "pressure",      # bar — drops with wear
        "rpm",           # RPM — becomes unstable
        "current",       # A   — increases with friction
        "torque",        # Nm  — fluctuates
        "oil_level",     # %   — decreases slowly
        "acoustics",     # dB  — rises near failure
    ]

    # Nominal (healthy) operating values
    NOMINAL = {
        "temperature": 75.0,
        "vibration":    0.3,
        "pressure":    12.0,
        "rpm":        1480.0,
        "current":     18.0,
        "torque":      95.0,
        "oil_level":   90.0,
        "acoustics":   62.0,
    }

    # Degradation sensitivity per channel (multiplier on health loss)
    SENSITIVITY = {
        "temperature": +15.0,   # rises
        "vibration":   +0.8,    # rises sharply near failure
        "pressure":    -3.0,    # drops
        "rpm":         -50.0,   # drops
        "current":     +5.0,    # rises
        "torque":      +12.0,   # rises
        "oil_level":   -15.0,   # slow drain
        "acoustics":   +12.0,   # rises
    }

    @classmethod
    def read(cls, health: float, noise_std: float = 0.02, in_fault: bool = False) -> dict:
        """
        Compute sensor readings for a given health state.

        Args:
            health:    Machine health in [0, 1]. 1 = perfect, 0 = failed.
            noise_std: Gaussian noise standard deviation.
            in_fault:  Whether the machine is currently in a fault state.

        Returns:
            Dict of sensor channel -> reading value.
        """
        degradation = 1.0 - health
        readings = {}

        for ch in cls.CHANNELS:
            base = cls.NOMINAL[ch]
            delta = cls.SENSITIVITY[ch] * degradation

            # Non-linear spike near failure
            if health < 0.2:
                spike = cls.SENSITIVITY[ch] * 0.5 * np.random.exponential(1.0)
            else:
                spike = 0.0

            noise = np.random.normal(0, abs(base) * noise_std)

            # Fault state: extreme values
            if in_fault:
                fault_factor = np.random.uniform(1.5, 3.0) * np.sign(cls.SENSITIVITY[ch])
                readings[ch] = base + delta * fault_factor + noise
            else:
                readings[ch] = base + delta + spike + noise

        return readings


# ─── Machine Process ───────────────────────────────────────────────────────────

class Machine:
    """
    Simulates a single machine's lifecycle using SimPy.
    Tracks health, generates sensor readings, injects faults.
    """

    def __init__(self, env: simpy.Environment, config: MachineConfig, records: list):
        self.env = env
        self.config = config
        self.records = records
        self.health = 1.0
        self.in_fault = False
        self.fault_count = 0
        self.total_downtime = 0.0

    def run(self) -> Generator:
        """Main simulation process — runs for the entire simulation duration."""
        while True:
            yield self.env.process(self._operate())
            if self.in_fault:
                yield self.env.process(self._repair())

    def _operate(self) -> Generator:
        """Normal operation: health degrades, sensors sampled at each interval."""
        ttf = np.random.exponential(self.config.mtbf_hours)  # time to next fault

        elapsed = 0.0
        while elapsed < ttf and self.health > 0:
            # Sample sensors
            reading = SensorModel.read(
                health=self.health,
                noise_std=self.config.sensor_noise,
                in_fault=False
            )
            self._record(reading, label=0, rul=max(0, ttf - elapsed))

            # Degrade health
            self.health = max(0.0, self.health - self.config.degradation_rate)

            yield self.env.timeout(self.config.sampling_interval)
            elapsed += self.config.sampling_interval

        # Trigger fault
        self.in_fault = True
        self.fault_count += 1

        # Record fault moment
        reading = SensorModel.read(health=0.0, noise_std=self.config.sensor_noise, in_fault=True)
        self._record(reading, label=1, rul=0)

    def _repair(self) -> Generator:
        """Repair process: machine is down, health restored after MTR."""
        repair_time = np.random.exponential(self.config.mttr_hours)
        self.total_downtime += repair_time

        # Record downtime (no sensor readings during repair)
        yield self.env.timeout(repair_time)

        self.health = 1.0
        self.in_fault = False

    def _record(self, sensor_reading: dict, label: int, rul: float) -> None:
        """Append a sensor reading to the shared records list."""
        row = {
            "timestamp_h": round(self.env.now, 3),
            "machine_id": self.config.machine_id,
            "health": round(self.health, 4),
            "label": label,         # 0 = normal, 1 = fault
            "rul": round(rul, 3),   # remaining useful life in hours
            **{k: round(v, 4) for k, v in sensor_reading.items()}
        }
        self.records.append(row)


# ─── Simulation Runner ────────────────────────────────────────────────────────

def run_simulation(config: SimulationConfig) -> pd.DataFrame:
    """
    Run the full production line simulation.

    Args:
        config: SimulationConfig instance.

    Returns:
        DataFrame with all sensor readings and labels.
    """
    np.random.seed(config.random_seed)
    records = []

    env = simpy.Environment()
    machines = []

    for machine_config in config.machines:
        m = Machine(env, machine_config, records)
        env.process(m.run())
        machines.append(m)

    sim_duration_hours = config.sim_days * 24
    print(f"Running simulation: {config.n_machines} machines × {config.sim_days} days "
          f"({sim_duration_hours}h)...")

    env.run(until=sim_duration_hours)

    df = pd.DataFrame(records)
    df = df.sort_values(["machine_id", "timestamp_h"]).reset_index(drop=True)

    # Summary
    n_faults = df[df["label"] == 1].groupby("machine_id").size()
    print(f"\nSimulation complete:")
    print(f"  Total records  : {len(df):,}")
    print(f"  Fault events   : {df['label'].sum():,} ({df['label'].mean():.1%})")
    print(f"  Faults per machine:\n{n_faults.to_string()}")

    return df


# ─── Export ───────────────────────────────────────────────────────────────────

def save_data(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"\nSaved → {path} ({os.path.getsize(path) / 1024:.1f} KB)")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Industrial sensor data simulator")
    parser.add_argument("--machines", type=int, default=3, help="Number of machines")
    parser.add_argument("--days", type=int, default=30, help="Simulation duration in days")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/raw/sensors.csv", help="Output CSV path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = SimulationConfig(
        n_machines=args.machines,
        sim_days=args.days,
        random_seed=args.seed,
        output_path=args.output
    )

    df = run_simulation(config)
    save_data(df, config.output_path)
    print("\nNext step: run preprocess.py")
