from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLE_DIR = RESULTS_DIR / "report_tables"
REPORT_PATH = RESULTS_DIR / "report_summary.md"

GOAL_LABELS = {
    "drop_t45": "Hoi truong T45",
    "drop_library": "Thu vien",
    "drop_k1": "K1",
    "drop_c1": "C1",
    "drop_dorm4": "KTX so 4",
}

POLICY_LABELS = {
    "random": "Random baseline",
    "greedy": "Greedy baseline",
    "ppo_100k_radius30": "PPO 100k radius30",
    "ppo_preliminary": "PPO 50k preliminary",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def percent(value: object) -> str:
    return f"{as_float(value) * 100:.1f}%"


def load_policy_rows() -> list[dict[str, str]]:
    rows = []
    rows.extend(read_csv(RESULTS_DIR / "policy_eval_summary.csv"))
    rows.extend(read_csv(RESULTS_DIR / "ppo_eval_summary_100k_radius30.csv"))
    return rows


def build_astar_table(astar_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    table = []
    for row in astar_rows:
        goal_id = row["goal_id"]
        table.append(
            {
                "goal": GOAL_LABELS.get(goal_id, goal_id),
                "path_found": row["path_found"],
                "path_points": row["path_points"],
                "path_length_px": row["path_length_px"],
                "straight_distance_px": row["straight_distance_px"],
                "efficiency_ratio": row["efficiency_ratio"],
            }
        )
    return table


def build_policy_table(policy_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    table = []
    for row in policy_rows:
        policy = row["policy"]
        goal_id = row["goal_id"]
        table.append(
            {
                "policy": POLICY_LABELS.get(policy, policy),
                "goal": GOAL_LABELS.get(goal_id, goal_id),
                "episodes": row["episodes"],
                "success_rate": percent(row["success_rate"]),
                "collision_rate": percent(row["collision_rate"]),
                "timeout_rate": percent(row["timeout_rate"]),
                "avg_steps": row["avg_steps"],
                "avg_reward": row["avg_reward"],
                "avg_final_distance_px": row["avg_final_distance_px"],
            }
        )
    return table


def build_goal_summary(policy_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    by_goal: dict[str, list[dict[str, str]]] = {}
    for row in policy_rows:
        by_goal.setdefault(row["goal_id"], []).append(row)

    summary = []
    for goal_id, rows in by_goal.items():
        best = max(
            rows,
            key=lambda r: (
                as_float(r["success_rate"]),
                -as_float(r["avg_steps"], default=999999.0),
                as_float(r["avg_reward"]),
            ),
        )
        summary.append(
            {
                "goal": GOAL_LABELS.get(goal_id, goal_id),
                "best_policy": POLICY_LABELS.get(best["policy"], best["policy"]),
                "best_success_rate": percent(best["success_rate"]),
                "best_avg_steps": best["avg_steps"],
                "best_avg_reward": best["avg_reward"],
            }
        )
    return summary


def nearest_obstacle_distance(occupancy_grid: np.ndarray, x: int, y: int, radius: int = 80) -> float | None:
    height, width = occupancy_grid.shape
    x1 = max(0, x - radius)
    x2 = min(width - 1, x + radius)
    y1 = max(0, y - radius)
    y2 = min(height - 1, y + radius)

    obstacle_points = np.argwhere(occupancy_grid[y1:y2 + 1, x1:x2 + 1] == 1)
    if obstacle_points.size == 0:
        return None

    # argwhere returns local [y, x] coordinates.
    dy = obstacle_points[:, 0] + y1 - y
    dx = obstacle_points[:, 1] + x1 - x
    distances = np.sqrt(dx * dx + dy * dy)
    return float(np.min(distances))


def build_poi_validation_table() -> list[dict[str, object]]:
    with (PROCESSED_DIR / "pois.json").open("r", encoding="utf-8") as f:
        pois = json.load(f)
    occupancy_grid = np.load(PROCESSED_DIR / "occupancy_grid.npy")
    height, width = occupancy_grid.shape

    rows: list[dict[str, object]] = []
    for group_name in ("pickup_points", "dropoff_points"):
        for point in pois[group_name]:
            x = int(round(float(point["x"])))
            y = int(round(float(point["y"])))
            inside_map = 0 <= x < width and 0 <= y < height
            blocked = True
            clearance = None
            if inside_map:
                blocked = bool(occupancy_grid[y, x] == 1)
                clearance = nearest_obstacle_distance(occupancy_grid, x, y)

            rows.append(
                {
                    "group": group_name,
                    "id": point["id"],
                    "label": point["label"],
                    "x": x,
                    "y": y,
                    "inside_map": inside_map,
                    "occupancy_at_point": int(blocked),
                    "is_free": inside_map and not blocked,
                    "nearest_obstacle_px": "" if clearance is None else round(clearance, 2),
                }
            )
    return rows


def markdown_table(rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    if not rows:
        return "_Khong co du lieu._"

    header = "| " + " | ".join(fieldnames) + " |"
    separator = "| " + " | ".join(["---"] * len(fieldnames)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(name, "")) for name in fieldnames) + " |")
    return "\n".join([header, separator, *body])


def main() -> None:
    astar_rows = read_csv(RESULTS_DIR / "astar_metrics.csv")
    policy_rows = load_policy_rows()

    astar_table = build_astar_table(astar_rows)
    policy_table = build_policy_table(policy_rows)
    goal_summary = build_goal_summary(policy_rows)
    poi_validation = build_poi_validation_table()

    write_csv(
        TABLE_DIR / "astar_routes.csv",
        astar_table,
        ["goal", "path_found", "path_points", "path_length_px", "straight_distance_px", "efficiency_ratio"],
    )
    write_csv(
        TABLE_DIR / "policy_comparison.csv",
        policy_table,
        [
            "policy",
            "goal",
            "episodes",
            "success_rate",
            "collision_rate",
            "timeout_rate",
            "avg_steps",
            "avg_reward",
            "avg_final_distance_px",
        ],
    )
    write_csv(
        TABLE_DIR / "goal_summary.csv",
        goal_summary,
        ["goal", "best_policy", "best_success_rate", "best_avg_steps", "best_avg_reward"],
    )
    write_csv(
        TABLE_DIR / "poi_validation.csv",
        poi_validation,
        [
            "group",
            "id",
            "label",
            "x",
            "y",
            "inside_map",
            "occupancy_at_point",
            "is_free",
            "nearest_obstacle_px",
        ],
    )

    astar_success = sum(1 for row in astar_rows if row.get("path_found") == "True")
    ppo_current = [row for row in policy_rows if row.get("policy") == "ppo_100k_radius30"]
    ppo_success = sum(as_float(row.get("success_rate")) for row in ppo_current)
    ppo_goal_count = len(ppo_current)

    report = f"""# TLU UAV PPO - Tom tat ket qua cho bao cao

File nay duoc sinh tu `eval/generate_report_summary.py` dua tren cac file trong `results/`.

## Diem chinh

- Ban do hien tai co 5 diem giao: T45, Thu vien, K1, C1, KTX so 4.
- A* tim duoc duong cho {astar_success}/{len(astar_rows)} muc tieu tren occupancy grid.
- PPO 100k tren ban do hien tai dat trung binh {ppo_success / ppo_goal_count * 100:.1f}% success neu tinh theo 5 muc tieu.
- Random baseline dung de chung minh hanh dong ngau nhien de va cham.
- Greedy baseline dung de chung minh chien luoc tham lam co the thanh cong o mot so diem, nhung de timeout o cac diem can di vong.

## Bang A* Baseline

{markdown_table(astar_table, ["goal", "path_found", "path_points", "path_length_px", "straight_distance_px", "efficiency_ratio"])}

## Bang So Sanh Policy

{markdown_table(policy_table, ["policy", "goal", "episodes", "success_rate", "collision_rate", "timeout_rate", "avg_steps", "avg_reward", "avg_final_distance_px"])}

## Policy Tot Nhat Theo Tung Muc Tieu

{markdown_table(goal_summary, ["goal", "best_policy", "best_success_rate", "best_avg_steps", "best_avg_reward"])}

## Kiem Tra Pickup/Dropoff

Bang nay xac nhan cac diem pickup/dropoff nam trong ban do va khong nam tren obstacle.

{markdown_table(poi_validation, ["group", "id", "label", "x", "y", "inside_map", "occupancy_at_point", "is_free", "nearest_obstacle_px"])}

## Cach doc ket qua

- `success_rate`: ty le episode ket thuc bang viec UAV vao vung goal.
- `collision_rate`: ty le episode va cham voi obstacle/no-fly.
- `timeout_rate`: ty le episode het `max_steps` nhung chua toi goal.
- `avg_steps`: so buoc trung binh; cang thap cang tot neu van thanh cong.
- `avg_reward`: tong reward trung binh; cao hon thuong la policy tot hon.
- `avg_final_distance_px`: khoang cach con lai toi goal khi episode ket thuc.
- `efficiency_ratio`: do dai duong A* / khoang cach thang; gan 1 nghia la duong gan toi uu ve hinh hoc.

## Goi y viet bao cao

- Dung A* lam baseline quy hoach duong di tren ban do tinh.
- Dung Random va Greedy lam baseline chinh sach don gian.
- Dung PPO de trinh bay huong hoc tang cuong: policy hoc tu reward thay vi duoc lap trinh quy tac duong di.
- Neu PPO that bai o mot goal, trinh bay nhu han che thuc nghiem va ly do can fine-tune/retrain tren ban do cuoi.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Generated report assets:")
    print(f"- {REPORT_PATH}")
    print(f"- {TABLE_DIR / 'astar_routes.csv'}")
    print(f"- {TABLE_DIR / 'policy_comparison.csv'}")
    print(f"- {TABLE_DIR / 'goal_summary.csv'}")
    print(f"- {TABLE_DIR / 'poi_validation.csv'}")


if __name__ == "__main__":
    main()
