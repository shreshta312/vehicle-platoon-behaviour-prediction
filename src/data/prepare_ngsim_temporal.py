"""
Prepare temporal NGSIM dataset for platoon behavior prediction.

Creates t0-t5 historical features from real NGSIM trajectory data.

This version uses groupby-shift per Vehicle_ID instead of assuming
Frame_ID increments by exactly 1.

Output:
    data/processed/platoon_data_ngsim_temporal.csv

Usage:
    python src/data/prepare_ngsim_temporal.py --input data/raw/ngsim.csv --sample 500000
"""

import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd


DECISION_BRAKE = 0
DECISION_FOLLOW = 1
DECISION_LANE_LEFT = 2
DECISION_LANE_RIGHT = 3

MAX_SENSOR_DIST = 150.0
TIMESTEPS = 6

SENSOR_COLS = [
    "ego_vel",
    "front_dist",
    "front_rel_vel",
    "time_gap",
    "left_front_dist",
    "left_rear_dist",
    "right_front_dist",
    "right_rear_dist",
]

REQUIRED_COLUMNS = [
    "Vehicle_ID",
    "Frame_ID",
    "Local_Y",
    "v_Vel",
    "v_Acc",
    "Lane_ID",
    "Preceding",
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
    print("Cleaning NGSIM data...")

    df = df.copy()

    for col in REQUIRED_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(
        subset=[
            "Vehicle_ID",
            "Frame_ID",
            "Local_Y",
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

    df = df[df["v_Vel"] >= 0]
    df = df[df["Space_Headway"] >= 0]

    df.loc[df["Time_Headway"] > 20, "Time_Headway"] = 20
    df.loc[df["Space_Headway"] > MAX_SENSOR_DIST, "Space_Headway"] = MAX_SENSOR_DIST

    print(f"After cleaning shape: {df.shape}")

    return df


def add_front_relative_velocity(df):
    print("Computing front relative velocity...")

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

    df["front_vel"] = df["front_vel"].fillna(df["v_Vel"])
    df["front_rel_vel"] = df["front_vel"] - df["v_Vel"]

    return df


def build_lane_position_index(df):
    print("Building lane-position index...")

    lane_index = defaultdict(dict)

    grouped = df.groupby(["Frame_ID", "Lane_ID"])

    for (frame_id, lane_id), group in grouped:
        positions = np.sort(group["Local_Y"].values.astype(np.float32))
        lane_index[int(frame_id)][int(lane_id)] = positions

    return lane_index


def add_adjacent_lane_distances(df, lane_index):
    print("Computing adjacent-lane distances...")

    df = df.copy()

    left_front = np.full(len(df), MAX_SENSOR_DIST, dtype=np.float32)
    left_rear = np.full(len(df), MAX_SENSOR_DIST, dtype=np.float32)
    right_front = np.full(len(df), MAX_SENSOR_DIST, dtype=np.float32)
    right_rear = np.full(len(df), MAX_SENSOR_DIST, dtype=np.float32)

    frame_values = df["Frame_ID"].values
    lane_values = df["Lane_ID"].values
    y_values = df["Local_Y"].values.astype(np.float32)

    for i in range(len(df)):
        frame_id = int(frame_values[i])
        lane_id = int(lane_values[i])
        ego_y = y_values[i]

        adjacent_lanes = [
            (lane_id - 1, left_front, left_rear),
            (lane_id + 1, right_front, right_rear),
        ]

        for target_lane, front_arr, rear_arr in adjacent_lanes:
            positions = lane_index.get(frame_id, {}).get(target_lane)

            if positions is None or len(positions) == 0:
                continue

            insert_idx = np.searchsorted(positions, ego_y)

            if insert_idx < len(positions):
                dist_front = positions[insert_idx] - ego_y
                if 0 <= dist_front <= MAX_SENSOR_DIST:
                    front_arr[i] = dist_front

            if insert_idx > 0:
                dist_rear = ego_y - positions[insert_idx - 1]
                if 0 <= dist_rear <= MAX_SENSOR_DIST:
                    rear_arr[i] = dist_rear

    df["left_front_dist"] = left_front
    df["left_rear_dist"] = left_rear
    df["right_front_dist"] = right_front
    df["right_rear_dist"] = right_rear

    return df


def add_future_labels(df, horizon_rows=10):
    """
    Creates future labels using row shift per vehicle.

    This is more reliable than Frame_ID arithmetic because some files
    may have irregular frame spacing after cleaning/sampling.
    """
    print("Creating future behavior labels...")

    df = df.copy()
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])

    df["future_lane"] = df.groupby("Vehicle_ID")["Lane_ID"].shift(-horizon_rows)
    df["future_vel"] = df.groupby("Vehicle_ID")["v_Vel"].shift(-horizon_rows)

    df = df.dropna(subset=["future_lane", "future_vel"])

    df["future_lane"] = df["future_lane"].astype(int)

    speed_drop = df["v_Vel"] - df["future_vel"]

    lane_left = df["future_lane"] < df["Lane_ID"]
    lane_right = df["future_lane"] > df["Lane_ID"]
    brake = (df["v_Acc"] < -4.0) | (speed_drop > 5.0)

    conditions = [
        lane_left,
        lane_right,
        brake,
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


def create_base_features(df):
    print("Creating base frame-level features...")

    base = pd.DataFrame()

    base["Vehicle_ID"] = df["Vehicle_ID"]
    base["Frame_ID"] = df["Frame_ID"]

    base["ego_vel"] = df["v_Vel"]
    base["front_dist"] = df["Space_Headway"]
    base["front_rel_vel"] = df["front_rel_vel"]
    base["time_gap"] = df["Time_Headway"]

    base["left_front_dist"] = df["left_front_dist"]
    base["left_rear_dist"] = df["left_rear_dist"]
    base["right_front_dist"] = df["right_front_dist"]
    base["right_rear_dist"] = df["right_rear_dist"]

    base["decision"] = df["decision"].astype(int)

    base = base.replace([np.inf, -np.inf], np.nan)
    base = base.dropna()

    return base


def create_temporal_features(base):
    """
    Creates temporal features using previous rows within each Vehicle_ID.

    t0 = current row
    t1 = previous row
    t2 = 2 rows before
    ...
    t5 = 5 rows before
    """
    print("Creating temporal t0-t5 features using groupby-shift...")

    base = base.sort_values(["Vehicle_ID", "Frame_ID"]).copy()

    result = base[["Vehicle_ID", "Frame_ID", "decision"]].copy()

    grouped = base.groupby("Vehicle_ID", group_keys=False)

    for t in range(TIMESTEPS):
        for col in SENSOR_COLS:
            if t == 0:
                result[f"{col}_t{t}"] = base[col]
            else:
                result[f"{col}_t{t}"] = grouped[col].shift(t)

    result = result.dropna()

    result = result.drop(columns=["Vehicle_ID", "Frame_ID"])

    feature_cols = [col for col in result.columns if col != "decision"]
    result = result[feature_cols + ["decision"]]

    print(f"Temporal dataset shape: {result.shape}")

    print("\nDecision distribution:")
    print(result["decision"].value_counts().sort_index())

    return result


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)

    parser.add_argument(
        "--output",
        default="data/processed/platoon_data_ngsim_temporal.csv",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Use first N rows for quick testing.",
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Future row gap per vehicle used for label creation.",
    )

    args = parser.parse_args()

    df = load_ngsim(args.input, args.sample)
    df = clean_ngsim(df)
    df = add_front_relative_velocity(df)

    lane_index = build_lane_position_index(df)
    df = add_adjacent_lane_distances(df, lane_index)

    df = add_future_labels(df, horizon_rows=args.horizon)

    base = create_base_features(df)
    temporal = create_temporal_features(base)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    temporal.to_csv(args.output, index=False)

    print(f"\nSaved temporal NGSIM dataset to: {args.output}")


if __name__ == "__main__":
    main()