"""Synthetic platoon dataset generator.

Creates a CSV of synthetic ego-vehicle samples with simple rule-based
labels for decisions (BRAKE, FOLLOW, CHANGE_LANE_LEFT, CHANGE_LANE_RIGHT).

Run as a script to produce `artifacts/synthetic_platoon.csv`.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

DECISION_BRAKE = 0
DECISION_FOLLOW = 1
DECISION_LANE_LEFT = 2
DECISION_LANE_RIGHT = 3

def generate_sample(rng):
    # lanes: 0=left,1=center,2=right
    lane = int(rng.integers(0, 3))

    # Ego speed (m/s), typical highway ~20-33 m/s
    ego_speed = float(rng.normal(25.0, 3.0))
    ego_speed = max(0.1, ego_speed)

    # Lead vehicle distance and speed
    lead_distance = float(rng.normal(20.0, 8.0))
    lead_distance = max(1.0, lead_distance)
    lead_speed = float(ego_speed + rng.normal(-2.0, 2.0))

    # Adjacent lane occupancy (simple Bernoulli)
    left_occupied = int(rng.random() < 0.35)
    right_occupied = int(rng.random() < 0.35)

    # Time gap feature (s)
    time_gap = lead_distance / ego_speed

    # Simple rule-based label generation (for synthetic ground truth)
    # Prefer changing right when gap is small and right lane is free.
    if time_gap < 1.0 and right_occupied == 0 and lane != 2:
        decision = DECISION_LANE_RIGHT
    elif time_gap < 0.9 and right_occupied == 1:
        decision = DECISION_BRAKE
    elif time_gap < 1.4:
        decision = DECISION_FOLLOW
    else:
        decision = DECISION_FOLLOW

    return {
        "ego_speed": round(ego_speed, 3),
        "lead_speed": round(lead_speed, 3),
        "lead_distance": round(lead_distance, 3),
        "time_gap": round(time_gap, 3),
        "lane": lane,
        "left_occupied": left_occupied,
        "right_occupied": right_occupied,
        "decision": int(decision),
    }

def generate_dataset(n_samples=5000, seed=42, out_path=None):
    rng = np.random.default_rng(seed)
    rows = [generate_sample(rng) for _ in range(n_samples)]
    df = pd.DataFrame(rows)

    if out_path is None:
        out_dir = Path(__file__).resolve().parents[1] / "artifacts"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "synthetic_platoon.csv"

    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate synthetic platoon dataset")
    parser.add_argument("--n", type=int, default=5000, help="number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default=None, help="output CSV path")

    args = parser.parse_args()
    out = generate_dataset(n_samples=args.n, seed=args.seed, out_path=args.out)
    print(f"Wrote synthetic dataset to: {out}")
