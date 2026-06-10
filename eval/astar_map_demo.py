from __future__ import annotations

import heapq
import json

import csv
import math

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PREVIEW_DIR = PROCESSED_DIR / "previews"
RESULT_DIR = PROJECT_ROOT / "results" / "trajectories"

OCCUPANCY_PATH = PROCESSED_DIR / "occupancy_grid.npy"
TYPE_MAP_PATH = PROCESSED_DIR / "type_map.npy"
POIS_PATH = PROCESSED_DIR / "pois.json"

TYPE_COLORS = [
    "#b8bdc2",  # road
    "#b56a5a",  # building
    "#3f8f4f",  # green
    "#edf1df",  # yard
    "#e25d5d",  # no_fly
    "#f4b942",  # pickup
    "#2878bd",  # dropoff
]

GOAL_LABELS = {
    "drop_t45": "Hội trường T45",
    "drop_library": "Thư viện",
    "drop_k1": "K1",
    "drop_c1": "C1",
    "drop_dorm4": "KTX số 4",
}


def load_pois():
    with POIS_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def find_point(points, point_id: str):
    for p in points:
        if p["id"] == point_id:
            return int(p["x"]), int(p["y"])
    raise ValueError(f"Không tìm thấy POI id={point_id}")


def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def path_length(path):
    """Tính tổng độ dài đường đi theo pixel."""
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(1, len(path)):
        x1, y1 = path[i - 1]
        x2, y2 = path[i]
        total += math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return total

def euclidean_distance(a, b):
    """Khoảng cách thẳng từ start tới goal."""
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def astar(occupancy_grid: np.ndarray, start, goal, step: int = 8):
    """
    A* đơn giản trên occupancy grid.
    step lớn hơn 1 để chạy nhanh trên map 1400x900.
    """
    height, width = occupancy_grid.shape

    def is_free(x, y):
        if x < 0 or x >= width or y < 0 or y >= height:
            return False
        return occupancy_grid[y, x] == 0

    def is_segment_free(a, b):
        dist = euclidean_distance(a, b)
        sample_count = max(2, int(math.ceil(dist / 2.0)))

        for i in range(sample_count + 1):
            t = i / sample_count
            x = int(round(a[0] * (1.0 - t) + b[0] * t))
            y = int(round(a[1] * (1.0 - t) + b[1] * t))
            if not is_free(x, y):
                return False

        return True

    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    open_set = []
    heapq.heappush(open_set, (0, start))

    came_from = {}
    g_score = {start: 0}

    directions = [
        (step, 0),
        (-step, 0),
        (0, step),
        (0, -step),
        (step, step),
        (step, -step),
        (-step, step),
        (-step, -step),
    ]

    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in visited:
            continue
        visited.add(current)

        if heuristic(current, goal) <= step * 2 and is_segment_free(current, goal):
            came_from[goal] = current
            return reconstruct_path(came_from, goal)

        for dx, dy in directions:
            nx = current[0] + dx
            ny = current[1] + dy
            neighbor = (nx, ny)

            if not is_segment_free(current, neighbor):
                continue

            move_cost = np.sqrt(dx * dx + dy * dy)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f, neighbor))

    return []


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def render_path(type_map, pois, start, goal, path, title, output_path):
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = ListedColormap(TYPE_COLORS)
    ax.imshow(type_map, cmap=cmap, origin="upper", vmin=0, vmax=len(TYPE_COLORS) - 1)

    ax.scatter(start[0], start[1], s=140, marker="o", edgecolors="black", label="Cổng sau")
    ax.scatter(goal[0], goal[1], s=160, marker="X", edgecolors="black", label="Điểm giao")

    if path:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        ax.plot(xs, ys, linewidth=2.5, label="Đường đi A*")
        ax.scatter(xs, ys, s=4)

    for p in pois["pickup_points"]:
        ax.text(p["x"] + 10, p["y"] - 10, p["label"], fontsize=8, weight="bold")

    for p in pois["dropoff_points"]:
        ax.text(p["x"] + 10, p["y"] - 10, p["label"], fontsize=8, weight="bold")

    ax.set_title(title)
    ax.set_xlim(0, type_map.shape[1])
    ax.set_ylim(type_map.shape[0], 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    occupancy_grid = np.load(OCCUPANCY_PATH)
    type_map = np.load(TYPE_MAP_PATH)
    pois = load_pois()

    pickup_id = "pickup_back_gate"
    dropoff_ids = [
        "drop_t45",
        "drop_library",
        "drop_k1",
        "drop_c1",
        "drop_dorm4",
    ]

    start = find_point(pois["pickup_points"], pickup_id)

    print("A* BASELINE MAP INTEGRATION DEMO")
    print("=" * 60)
    print(f"Start: {pickup_id} at {start}")

    metrics = []

    for dropoff_id in dropoff_ids:
        goal = find_point(pois["dropoff_points"], dropoff_id)
        path = astar(occupancy_grid, start, goal, step=8)

        print("-" * 60)
        print(f"Route: {pickup_id} -> {dropoff_id}")
        print(f"Goal : {goal}")

        found = bool(path)
        path_len = path_length(path) if found else 0.0
        straight_len = euclidean_distance(start, goal)
        efficiency_ratio = path_len / straight_len if found and straight_len > 0 else None

        row = {
            "route": f"{pickup_id}_to_{dropoff_id}",
            "start_id": pickup_id,
            "goal_id": dropoff_id,
            "start_x": start[0],
            "start_y": start[1],
            "goal_x": goal[0],
            "goal_y": goal[1],
            "path_found": found,
            "path_points": len(path) if found else 0,
            "path_length_px": round(path_len, 2),
            "straight_distance_px": round(straight_len, 2),
            "efficiency_ratio": round(efficiency_ratio, 4) if efficiency_ratio is not None else "",
            "status": "pass" if found else "fail",
        }
        metrics.append(row)

        if path:
            print(f"Path found. Points: {len(path)}")
            print(f"Path length: {path_len:.2f} px")
            print(f"Straight distance: {straight_len:.2f} px")
            print(f"Efficiency ratio: {efficiency_ratio:.4f}")

            output_path = RESULT_DIR / f"astar_{pickup_id}_to_{dropoff_id}.png"
            render_path(
                type_map=type_map,
                pois=pois,
                start=start,
                goal=goal,
                path=path,
                title=f"A*: Cổng sau → {GOAL_LABELS.get(dropoff_id, dropoff_id)}",
                output_path=output_path,
            )
            print(f"Saved: {output_path}")
        else:
            print("No path found.")

    # Save metrics as CSV
    csv_path = PROJECT_ROOT / "results" / "astar_metrics.csv"
    json_path = PROJECT_ROOT / "results" / "astar_metrics.json"

    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "route",
        "start_id",
        "goal_id",
        "start_x",
        "start_y",
        "goal_x",
        "goal_y",
        "path_found",
        "path_points",
        "path_length_px",
        "straight_distance_px",
        "efficiency_ratio",
        "status",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("A* metrics saved:")
    print(f"- {csv_path}")
    print(f"- {json_path}")


if __name__ == "__main__":
    main()
