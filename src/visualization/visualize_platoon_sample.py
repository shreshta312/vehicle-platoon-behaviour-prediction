"""Visualize a single platoon sample from a CSV.

Shows lanes and vehicle positions; annotates the ego vehicle and the
decision label (e.g., CHANGE_LANE_RIGHT).
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

decision_map = {
    0: "BRAKE",
    1: "FOLLOW",
    2: "CHANGE_LANE_LEFT",
    3: "CHANGE_LANE_RIGHT",
}

def plot_sample(row, show=True, save_path=None):
    # Lanes y positions
    lane_ys = {0: 1.0, 1: 0.0, 2: -1.0}

    fig, ax = plt.subplots(figsize=(8, 4))

    # Draw lanes
    for y in lane_ys.values():
        ax.hlines(y, -10, 100, colors="#ddd", linewidth=3)

    # Ego at x=0
    ego_x = 0.0
    ego_y = lane_ys.get(int(row['lane']), 0.0)
    ax.scatter([ego_x], [ego_y], c="blue", s=200, label="Ego")
    ax.annotate("EGO", (ego_x, ego_y), textcoords="offset points", xytext=(5, 5))

    # Lead vehicle ahead at lead_distance
    lead_x = float(row['lead_distance'])
    lead_y = ego_y
    ax.scatter([lead_x], [lead_y], c="red", s=160, label="Lead")
    ax.annotate("Lead", (lead_x, lead_y), textcoords="offset points", xytext=(5, 5))

    # Show adjacent occupancy markers
    if int(row.get('left_occupied', 0)):
        ax.scatter([10], [lane_ys.get(max(0, int(row['lane'])-1), 1.0)], c="#555", s=100)
    if int(row.get('right_occupied', 0)):
        ax.scatter([10], [lane_ys.get(min(2, int(row['lane'])+1), -1.0)], c="#555", s=100)

    # Annotate features and decision
    decision = int(row['decision'])
    txt = (
        f"speed={row['ego_speed']} m/s, gap={row['time_gap']} s\n"
        f"decision={decision_map.get(decision, decision)}"
    )
    ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlim(-5, max(40, lead_x + 10))
    ax.set_ylim(-2, 2)
    ax.set_xlabel("Distance (m)")
    ax.set_yticks([])
    ax.set_title("Platoon sample visualization")
    ax.legend(loc="lower right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)

def visualize_from_csv(csv_path, index=0):
    df = pd.read_csv(csv_path)
    if index < 0 or index >= len(df):
        raise IndexError("index out of range")
    plot_sample(df.iloc[index])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize platoon CSV sample")
    parser.add_argument("csv", type=str, nargs="?", default=None, help="path to CSV (defaults to artifacts/synthetic_platoon.csv)")
    parser.add_argument("--index", type=int, default=0)

    args = parser.parse_args()
    csv = args.csv
    if csv is None:
        csv = Path(__file__).resolve().parents[1] / "artifacts" / "synthetic_platoon.csv"
    visualize_from_csv(csv, index=args.index)
