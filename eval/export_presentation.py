from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "presentation"

TYPE_COLORS = [
    "#b8bdc2",  # road
    "#b56a5a",  # building
    "#3f8f4f",  # green
    "#edf1df",  # yard
    "#e25d5d",  # no_fly
    "#f4b942",  # pickup
    "#2878bd",  # dropoff
]

POLICY_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
    "ppo_100k_radius30": "PPO 100k",
    "ppo_preliminary": "PPO 50k",
}

GOAL_LABELS = {
    "drop_t45": "Hội trường T45",
    "drop_library": "Thư viện",
    "drop_k1": "K1",
    "drop_c1": "C1",
    "drop_dorm4": "KTX số 4",
}


def load_pois() -> dict:
    with (PROCESSED_DIR / "pois.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def save_map_overview() -> None:
    type_map = np.load(PROCESSED_DIR / "type_map.npy")
    pois = load_pois()

    fig, ax = plt.subplots(figsize=(14, 8), dpi=180)
    ax.imshow(type_map, cmap=ListedColormap(TYPE_COLORS), origin="upper", interpolation="nearest")

    for p in pois["pickup_points"]:
        ax.scatter(p["x"], p["y"], s=110, marker="o", edgecolors="black", color="#f4b942", label="Cổng sau")
        ax.text(p["x"] + 12, p["y"] - 10, p["label"], fontsize=8, weight="bold")

    for p in pois["dropoff_points"]:
        ax.scatter(p["x"], p["y"], s=120, marker="X", edgecolors="black", color="#2878bd")
        ax.text(p["x"] + 12, p["y"] - 10, p["label"], fontsize=8, weight="bold")

    for p in pois.get("landmarks", []):
        if not str(p["id"]).endswith("_door"):
            continue
        ax.scatter(p["x"], p["y"], s=105, marker="X", edgecolors="black", color="#2878bd")
        ax.text(p["x"] + 12, p["y"] - 10, p["label"], fontsize=8, weight="bold")

    ax.set_title("Bản đồ mô phỏng khuôn viên cho bài toán UAV", fontsize=13, weight="bold")
    ax.set_axis_off()
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "map_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_raster_layers() -> None:
    layers = [
        ("Lưới vật cản", np.load(PROCESSED_DIR / "occupancy_grid.npy"), "gray_r", 0, 1),
        ("Lớp semantic", np.load(PROCESSED_DIR / "type_map.npy"), ListedColormap(TYPE_COLORS), 0, len(TYPE_COLORS) - 1),
        ("Lớp độ cao tương đối", np.load(PROCESSED_DIR / "height_map.npy"), "viridis", 0, 3),
        ("Lớp rủi ro", np.load(PROCESSED_DIR / "risk_map.npy"), "magma", 0, 1),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), dpi=180)
    for ax, (title, data, cmap, vmin, vmax) in zip(axes.ravel(), layers):
        im = ax.imshow(data, cmap=cmap, origin="upper", interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=11, weight="bold")
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)

    fig.suptitle("Các lớp raster dùng cho môi trường mô phỏng", fontsize=13, weight="bold")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "raster_layers.png", bbox_inches="tight")
    plt.close(fig)


def read_policy_summary() -> list[dict[str, str]]:
    csv_path = RESULTS_DIR / "policy_eval_summary.csv"
    if not csv_path.exists():
        return []

    with csv_path.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_baseline_comparison() -> None:
    rows = read_policy_summary()
    if not rows:
        return

    labels = []
    success = []
    collision = []
    timeout = []

    for row in rows:
        policy = POLICY_LABELS.get(row["policy"], row["policy"])
        goal = GOAL_LABELS.get(row["goal_id"], row["goal_id"])
        labels.append(f"{policy}\n{goal}")
        success.append(float(row["success_rate"]))
        collision.append(float(row["collision_rate"]))
        timeout.append(float(row["timeout_rate"]))

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6), dpi=180)
    ax.bar(x - width, success, width, label="Thành công", color="#2e7d32")
    ax.bar(x, collision, width, label="Va chạm", color="#c62828")
    ax.bar(x + width, timeout, width, label="Quá thời gian", color="#f9a825")

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Tỷ lệ")
    ax.set_title("So sánh baseline trên bản đồ mô phỏng", fontsize=13, weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "baseline_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_map_overview()
    save_raster_layers()
    save_baseline_comparison()
    print(f"Presentation figures saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
