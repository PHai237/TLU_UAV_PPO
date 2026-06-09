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
    "drop_t45": "Hội trường T45",
    "drop_library": "Thư viện",
    "drop_k1": "K1",
    "drop_c1": "C1",
    "drop_dorm4": "KTX số 4",
}

POLICY_LABELS = {
    "random": "Random baseline",
    "greedy": "Greedy baseline",
    "ppo_100k_radius30": "PPO 100k bán kính 30",
    "ppo_preliminary": "PPO 50k sơ bộ",
}

REPORT_HEADERS = {
    "goal": "Mục tiêu",
    "path_found": "Tìm được đường",
    "path_points": "Số điểm đường đi",
    "path_length_px": "Độ dài đường đi (px)",
    "straight_distance_px": "Khoảng cách thẳng (px)",
    "efficiency_ratio": "Tỷ lệ hiệu quả",
    "policy": "Chính sách",
    "episodes": "Số episode",
    "success_rate": "Tỷ lệ thành công",
    "collision_rate": "Tỷ lệ va chạm",
    "timeout_rate": "Tỷ lệ quá thời gian",
    "avg_steps": "Số bước TB",
    "avg_reward": "Reward TB",
    "avg_final_distance_px": "Khoảng cách cuối TB (px)",
    "best_policy": "Chính sách tốt nhất",
    "best_success_rate": "Tỷ lệ thành công tốt nhất",
    "best_avg_steps": "Số bước TB tốt nhất",
    "best_avg_reward": "Reward TB tốt nhất",
    "group": "Nhóm",
    "id": "Mã điểm",
    "label": "Tên hiển thị",
    "x": "x",
    "y": "y",
    "inside_map": "Trong bản đồ",
    "occupancy_at_point": "Obstacle tại điểm",
    "is_free": "Điểm hợp lệ",
    "nearest_obstacle_px": "Obstacle gần nhất (px)",
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


def yes_no(value: object) -> str:
    if isinstance(value, str):
        return "Có" if value.lower() == "true" else "Không"
    return "Có" if bool(value) else "Không"


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
                "path_found": yes_no(row["path_found"]),
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
                    "inside_map": yes_no(inside_map),
                    "occupancy_at_point": int(blocked),
                    "is_free": yes_no(inside_map and not blocked),
                    "nearest_obstacle_px": "" if clearance is None else round(clearance, 2),
                }
            )
    return rows


def markdown_table(rows: list[dict[str, object]], fieldnames: list[str]) -> str:
    if not rows:
        return "_Không có dữ liệu._"

    header = "| " + " | ".join(REPORT_HEADERS.get(name, name) for name in fieldnames) + " |"
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

    report = f"""# TLU UAV PPO - Tóm tắt kết quả cho báo cáo

File này được sinh từ `eval/generate_report_summary.py` dựa trên các file trong `results/`.

## Điểm chính

- Bản đồ hiện tại có 5 điểm giao: T45, Thư viện, K1, C1, KTX số 4.
- A* tìm được đường cho {astar_success}/{len(astar_rows)} mục tiêu trên occupancy grid.
- PPO 100k trên bản đồ hiện tại đạt trung bình {ppo_success / ppo_goal_count * 100:.1f}% success nếu tính theo 5 mục tiêu.
- Random baseline dùng để chứng minh hành động ngẫu nhiên dễ va chạm.
- Greedy baseline dùng để chứng minh chiến lược tham lam có thể thành công ở một số điểm, nhưng dễ timeout ở các điểm cần đi vòng.
- PPO evaluation dùng `deterministic=True` với điểm xuất phát cố định; các episode cùng goal kiểm tra tính nhất quán của policy.

## Bảng A* Baseline

{markdown_table(astar_table, ["goal", "path_found", "path_points", "path_length_px", "straight_distance_px", "efficiency_ratio"])}

## Bảng so sánh policy

{markdown_table(policy_table, ["policy", "goal", "episodes", "success_rate", "collision_rate", "timeout_rate", "avg_steps", "avg_reward", "avg_final_distance_px"])}

## Policy tốt nhất theo từng mục tiêu

{markdown_table(goal_summary, ["goal", "best_policy", "best_success_rate", "best_avg_steps", "best_avg_reward"])}

## Kiểm tra pickup/dropoff

Bảng này xác nhận các điểm pickup/dropoff nằm trong bản đồ và không nằm trên obstacle.

{markdown_table(poi_validation, ["group", "id", "label", "x", "y", "inside_map", "occupancy_at_point", "is_free", "nearest_obstacle_px"])}

## Cách đọc kết quả

- `success_rate`: tỷ lệ episode kết thúc bằng việc UAV vào vùng goal.
- `collision_rate`: tỷ lệ episode va chạm với obstacle/no-fly.
- `timeout_rate`: tỷ lệ episode hết `max_steps` nhưng chưa tới goal.
- `avg_steps`: số bước trung bình; càng thấp càng tốt nếu vẫn thành công.
- `avg_reward`: tổng reward trung bình; cao hơn thường là policy tốt hơn.
- `avg_final_distance_px`: khoảng cách còn lại tới goal khi episode kết thúc.
- `efficiency_ratio`: độ dài đường A* / khoảng cách thẳng; gần 1 nghĩa là đường gần tối ưu về hình học.

## Gợi ý viết báo cáo

- Dùng A* làm baseline quy hoạch đường đi trên bản đồ tĩnh.
- Dùng Random và Greedy làm baseline chính sách đơn giản.
- Dùng PPO để trình bày hướng học tăng cường: policy học từ reward thay vì được lập trình quy tắc đường đi.
- Nếu PPO thất bại ở một goal, trình bày như hạn chế thực nghiệm và lý do cần fine-tune/retrain trên bản đồ cuối.
- Với Thư viện, quỹ đạo kẹt gần biên trên cho thấy policy chưa học được chiến lược tiếp cận goal; không phải do goal nằm trên obstacle.
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
