"""
Prepare NGSIM vehicle trajectory data for platoon behavior prediction.

V2 improvement:
- Uses Local_Y to compute real adjacent-lane distances.
- Computes:
    left_front_dist
    left_rear_dist
    right_front_dist
    right_rear_dist
- Derives behavior labels from future lane/speed changes.

Output format:
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
    python src/data/prepare_ngsim_dataset_v2.py --input data/raw/ngsim.csv --sample 500000
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

    # NGSIM often uses huge values when no valid headway exists.
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
    """
    Builds an index:
        frame_id -> lane_id -> sorted Local_Y positions

    This allows nearest front/rear vehicle lookup in adjacent lanes.
    """
    print("Building lane-position index...")

    lane_index = defaultdict(dict)

    grouped = df.groupby(["Frame_ID", "Lane_ID"])

    for (frame_id, lane_id), group in grouped:
        positions = np.sort(group["Local_Y"].values.astype(np.float32))
        lane_index[int(frame_id)][int(lane_id)] = positions

    return lane_index


def get_adjacent_lane_distances(df, lane_index):
    print("Computing adjacent-lane front/rear distances...")

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

        # In many NGSIM files, smaller lane number is left.
        # If your lane orientation is opposite, the model still learns left/right consistently.
        adjacent_specs = [
            (lane_id - 1, left_front, left_rear),
            (lane_id + 1, right_front, right_rear),
        ]

        for target_lane, front_arr, rear_arr in adjacent_specs:
            positions = lane_index.get(frame_id, {}).get(target_lane)

            if positions is None or len(positions) == 0:
                continue

            insert_idx = np.searchsorted(positions, ego_y)

            # Front vehicle in adjacent lane
            if insert_idx < len(positions):
                dist_front = positions[insert_idx] - ego_y
                if 0 <= dist_front <= MAX_SENSOR_DIST:
                    front_arr[i] = dist_front

            # Rear vehicle in adjacent lane
            if insert_idx > 0:
                dist_rear = ego_y - positions[insert_idx - 1]
                if 0 <= dist_rear <= MAX_SENSOR_DIST:
                    rear_arr[i] = dist_rear

    df["left_front_dist"] = left_front
    df["left_rear_dist"] = left_rear
    df["right_front_dist"] = right_front
    df["right_rear_dist"] = right_rear

    return df


def add_future_state_and_labels(df, horizon_frames=10):
    print("Creating future labels...")

    df = df.copy()
    df = df.sort_values(["Vehicle_ID", "Frame_ID"])

    future = df[["Vehicle_ID", "Frame_ID", "Lane_ID", "v_Vel"]].copy()

    # Current frame joins with future frame.
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

    lane_left = df["future_lane"] < df["Lane_ID"]
    lane_right = df["future_lane"] > df["Lane_ID"]

    # NGSIM velocity is often in ft/s.
    # A speed drop of 5 ft/s over 1 second is meaningful braking.
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


def create_project_features(df):
    print("Creating final project feature dataset...")

    output = pd.DataFrame()

    output["ego_vel"] = df["v_Vel"]
    output["front_dist"] = df["Space_Headway"]
    output["front_rel_vel"] = df["front_rel_vel"]
    output["time_gap"] = df["Time_Headway"]

    output["left_front_dist"] = df["left_front_dist"]
    output["left_rear_dist"] = df["left_rear_dist"]
    output["right_front_dist"] = df["right_front_dist"]
    output["right_rear_dist"] = df["right_rear_dist"]

    output["decision"] = df["decision"].astype(int)

    output = output.replace([np.inf, -np.inf], np.nan)
    output = output.dropna()

    print(f"Final processed shape: {output.shape}")
    print("\nDecision distribution:")
    print(output["decision"].value_counts().sort_index())

    return output


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--output",
        default="data/processed/platoon_data_ngsim_v2.csv",
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
        help="Future frame gap for label creation.",
    )

    args = parser.parse_args()

    df = load_ngsim(args.input, args.sample)
    df = clean_ngsim(df)
    df = add_front_relative_velocity(df)

    lane_index = build_lane_position_index(df)
    df = get_adjacent_lane_distances(df, lane_index)

    df = add_future_state_and_labels(df, horizon_frames=args.horizon)
    output = create_project_features(df)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output.to_csv(args.output, index=False)

    print(f"\nSaved processed NGSIM v2 dataset to: {args.output}")


if __name__ == "__main__":
    main()