"""
Prepare NGSIM vehicle trajectory data for platoon behavior prediction.

Converts raw NGSIM trajectory columns into the project feature format:

ego_vel
front_dist
front_rel_vel
time_gap
left_front_dist
left_rear_dist
right_front_dist
right_rear_dist
decision

Decision labels:
0 = BRAKE
1 = FOLLOW
2 = CHANGE_LANE_LEFT
3 = CHANGE_LANE_RIGHT

Usage:
    python src/data/prepare_ngsim_dataset.py --input data/raw/ngsim.csv --sample 100000
    python src/data/prepare_ngsim_dataset.py --input data/raw/ngsim.csv --sample 500000
    python src/data/prepare_ngsim_dataset.py --input data/raw/ngsim.csv --output data/processed/platoon_data_ngsim.csv
"""

import argparse
import os

import numpy as np
import pandas as pd


DECISION_BRAKE = 0
DECISION_FOLLOW = 1
DECISION_LANE_LEFT = 2
DECISION_LANE_RIGHT = 3


REQUIRED_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
    "Following",
    "Space_Headway",
    "Time_Headway",
]


def load_ngsim(path, sample=None):
    print("Loading NGSIM data...")

    df = pd.read_csv(
        path,
        usecols=lambda col: col in REQUIRED_COLUMNS,
        low_memory=False,
    )

    print(f"Raw loaded shape: {df.shape}")

    if sample is not None and sample < len(df):
        df = df.head(sample).copy()
        print(f"Using first {sample} rows for sequence-preserving sample.")
        print(f"Sampled shape: {df.shape}")

    return df


def clean_ngsim(df):
    print("Cleaning data...")

    df = df.copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "Vehicle_ID",
            "Frame_ID",
            "v_Vel",
            "v_Acc",
            "Lane_ID",
            "Space_Headway",
            "Time_Headway",
        ]
    )

    df["Vehicle_ID"] = df["Vehicle_ID"].astype(int)
    df["Frame_ID"] = df["Frame_ID"].astype(int)
    df["Lane_ID"] = df["Lane_ID"].astype(int)
    df["Preceding"] = df["Preceding"].fillna(0).astype(int)
    df["Following"] = df["Following"].fillna(0).astype(int)

    df = df[df["v_Vel"] >= 0]
    df = df[df["Space_Headway"] >= 0]

    # NGSIM often uses 9999.99 to mean no valid time headway.
    df.loc[df["Time_Headway"] > 20, "Time_Headway"] = 20

    # Cap very large distances to keep scale reasonable.
    df.loc[df["Space_Headway"] > 300, "Space_Headway"] = 300

    print(f"After cleaning shape: {df.shape}")

    return df


def add_front_relative_velocity(df):
    print("Computing front relative velocity...")

    df = df.copy()

    front_lookup = df[["Vehicle_ID", "Frame_ID", "v_Vel"]].copy()

    front_lookup = front_lookup.rename(
        columns={
            "Vehicle_ID": "Preceding",
            "v_Vel": "front_vel",
        }
    )

    df = df.merge(
        front_lookup,
        on=["Preceding", "Frame_ID"],
        how="left",
    )

    # If no preceding vehicle is found, assume relative velocity is 0.
    df["front_vel"] = df["front_vel"].fillna(df["v_Vel"])
    df["front_rel_vel"] = df["front_vel"] - df["v_Vel"]

    return df


def add_future_state_and_labels(df, horizon_frames=10):
    print("Creating future labels...")

    df = df.copy()
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])

    future = df[["Vehicle_ID", "Frame_ID", "Lane_ID", "v_Vel"]].copy()

    # Shift future rows backward so current Frame_ID joins to future Frame_ID.
    future["Frame_ID"] = future["Frame_ID"] - horizon_frames

    future = future.rename(
        columns={
            "Lane_ID": "future_lane",
            "v_Vel": "future_vel",
        }
    )

    df = df.merge(
        future,
        on=["Vehicle_ID", "Frame_ID"],
        how="left",
    )

    df = df.dropna(subset=["future_lane", "future_vel"])
    df["future_lane"] = df["future_lane"].astype(int)

    speed_drop = df["v_Vel"] - df["future_vel"]

    conditions = [
        df["future_lane"] < df["Lane_ID"],
        df["future_lane"] > df["Lane_ID"],
        (df["v_Acc"] < -2.0) | (speed_drop > 5.0),
    ]

    choices = [
        DECISION_LANE_LEFT,
        DECISION_LANE_RIGHT,
        DECISION_BRAKE,
    ]

    df["decision"] = np.select(
        conditions,
        choices,
        default=DECISION_FOLLOW,
    )

    return df


def create_project_features(df):
    print("Creating project feature dataset...")

    output = pd.DataFrame()

    output["ego_vel"] = df["v_Vel"]
    output["front_dist"] = df["Space_Headway"]
    output["front_rel_vel"] = df["front_rel_vel"]
    output["time_gap"] = df["Time_Headway"]

    # NGSIM does not directly provide adjacent-lane front/rear distances
    # in the same simple form as the synthetic platoon dataset.
    # These placeholders keep the same feature schema for first real-data testing.
    output["left_front_dist"] = 150.0
    output["left_rear_dist"] = 150.0
    output["right_front_dist"] = 150.0
    output["right_rear_dist"] = 150.0

    output["decision"] = df["decision"].astype(int)

    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.dropna()

    print(f"Final processed shape: {output.shape}")

    print("\nDecision distribution:")
    print(output["decision"].value_counts().sort_index())

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
        help="Path to raw NGSIM CSV file.",
    )

    parser.add_argument(
        "--output",
        default="data/processed/platoon_data_ngsim.csv",
        help="Output path for processed NGSIM platoon dataset.",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use first N rows for a quick sequence-preserving test.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Number of frames ahead used to create behavior labels.",
    )

    args = parser.parse_args()

    df = load_ngsim(args.input, args.sample)
    df = clean_ngsim(df)
    df = add_front_relative_velocity(df)
    df = add_future_state_and_labels(df, horizon_frames=args.horizon)

    output = create_project_features(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"\nSaved processed NGSIM dataset to: {args.output}")


if __name__ == "__main__":
    main()