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

from envs.tlu_uav_env import TluUavEnv  # noqa: E402


RESULTS_DIR = PROJECT_ROOT / "results"
TRAJECTORY_DIR = RESULTS_DIR / "trajectories"
SUMMARY_CSV = RESULTS_DIR / "policy_eval_summary.csv"
SUMMARY_JSON = RESULTS_DIR / "policy_eval_summary.json"

TYPE_COLORS = [
    "#b8bdc2",  # road
    "#b56a5a",  # building
    "#3f8f4f",  # green
    "#edf1df",  # yard
    "#e25d5d",  # no_fly
    "#f4b942",  # pickup
    "#2878bd",  # dropoff
]

GOAL_IDS = [
    "drop_t45",
    "drop_library",
    "drop_k1",
    "drop_c1",
    "drop_dorm4",
]

GOAL_LABELS = {
    "drop_t45": "Hội trường T45",
    "drop_library": "Thư viện",
    "drop_k1": "K1",
    "drop_c1": "C1",
    "drop_dorm4": "KTX số 4",
}

POLICY_LABELS = {
    "random": "Random",
    "greedy": "Greedy",
}

EVENT_LABELS = {
    "goal_reached": "Hoàn thành",
    "timeout": "Quá thời gian",
    "collision": "Va chạm",
    "move": "Đang di chuyển",
    "unknown": "Không rõ",
}


def unpack_reset(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, {}


def unpack_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = terminated or truncated
        return obs, reward, done, info

    obs, reward, done, info = result
    return obs, reward, done, info


def run_episode(
    policy_name: str,
    goal_id: str,
    seed: int,
    max_steps: int = 500,
):
    env = TluUavEnv(goal_id=goal_id, max_steps=max_steps, seed=seed)
    unpack_reset(env.reset(seed=seed, options={"goal_id": goal_id}))

    positions = [env.position.copy()]
    total_reward = 0.0
    done = False
    info = {}

    while not done:
        if policy_name == "random":
            action = env.sample_random_action()
        elif policy_name == "greedy":
            action = env.greedy_action()
        else:
            raise ValueError(f"Unsupported policy: {policy_name}")

        result = env.step(action)
        _, reward, done, info = unpack_step(result)

        total_reward += float(reward)
        positions.append(env.position.copy())

        if env.steps >= max_steps:
            break

    return {
        "policy": policy_name,
        "goal_id": goal_id,
        "success": bool(info.get("success", False)),
        "collision": bool(info.get("collision", False)),
        "timeout": bool(info.get("timeout", False)),
        "event": str(info.get("event", "unknown")),
        "steps": int(info.get("steps", env.steps)),
        "total_reward": float(total_reward),
        "final_distance": float(info.get("distance_to_goal", env.prev_distance)),
        "positions": positions,
        "env": env,
    }


def find_point(points, point_id: str):
    for p in points:
        if p["id"] == point_id:
            return float(p["x"]), float(p["y"]), p["label"]
    raise ValueError(f"POI not found: {point_id}")


def render_trajectory(env: TluUavEnv, positions, title: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14, 9))
    cmap = ListedColormap(TYPE_COLORS)
    ax.imshow(
        env.type_map,
        cmap=cmap,
        origin="upper",
        vmin=0,
        vmax=len(TYPE_COLORS) - 1,
    )

    start_x, start_y, start_label = find_point(
        env.pois["pickup_points"],
        "pickup_back_gate",
    )
    goal_x, goal_y, goal_label = find_point(
        env.pois["dropoff_points"],
        env.goal_id,
    )

    xs = [float(p[0]) for p in positions]
    ys = [float(p[1]) for p in positions]

    ax.plot(xs, ys, linewidth=2.2, label="Quỹ đạo")
    ax.scatter(xs, ys, s=5)

    ax.scatter(start_x, start_y, s=140, marker="o", edgecolors="black", label="Cổng sau")
    ax.scatter(goal_x, goal_y, s=160, marker="X", edgecolors="black", label="Điểm giao")

    ax.text(start_x + 10, start_y - 10, start_label, fontsize=8, weight="bold")
    ax.text(goal_x + 10, goal_y - 10, goal_label, fontsize=8, weight="bold")

    ax.set_title(title)
    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def summarize_results(policy_name: str, goal_id: str, episode_results):
    total = len(episode_results)

    success_count = sum(1 for r in episode_results if r["success"])
    collision_count = sum(1 for r in episode_results if r["collision"])
    timeout_count = sum(1 for r in episode_results if r["timeout"])

    avg_steps = float(np.mean([r["steps"] for r in episode_results]))
    avg_reward = float(np.mean([r["total_reward"] for r in episode_results]))
    avg_final_distance = float(np.mean([r["final_distance"] for r in episode_results]))

    return {
        "policy": policy_name,
        "goal_id": goal_id,
        "episodes": total,
        "success_count": success_count,
        "collision_count": collision_count,
        "timeout_count": timeout_count,
        "success_rate": round(success_count / total, 4),
        "collision_rate": round(collision_count / total, 4),
        "timeout_rate": round(timeout_count / total, 4),
        "avg_steps": round(avg_steps, 2),
        "avg_reward": round(avg_reward, 2),
        "avg_final_distance_px": round(avg_final_distance, 2),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    TRAJECTORY_DIR.mkdir(parents=True, exist_ok=True)

    policies = {
        "random": 20,
        # Greedy is deterministic for a fixed map/start/goal, but using 10
        # episodes keeps the baseline table aligned with PPO evaluation.
        "greedy": 10,
    }

    all_summaries = []

    print("POLICY EVALUATION ON TLU UAV ENV")
    print("=" * 70)

    for policy_name, episodes_per_goal in policies.items():
        for goal_id in GOAL_IDS:
            episode_results = []

            for ep in range(episodes_per_goal):
                seed = 1000 + ep
                result = run_episode(
                    policy_name=policy_name,
                    goal_id=goal_id,
                    seed=seed,
                    max_steps=500,
                )
                episode_results.append(result)

            summary = summarize_results(policy_name, goal_id, episode_results)
            all_summaries.append(summary)

            # Render the best episode if success exists, otherwise first episode.
            success_episodes = [r for r in episode_results if r["success"]]
            if success_episodes:
                sample = min(success_episodes, key=lambda r: r["steps"])
            else:
                sample = episode_results[0]

            output_path = TRAJECTORY_DIR / f"{policy_name}_{goal_id}.png"
            render_trajectory(
                env=sample["env"],
                positions=sample["positions"],
                title=(
                    f"{POLICY_LABELS[policy_name]}: Cổng sau → {GOAL_LABELS[goal_id]} | "
                    f"{EVENT_LABELS.get(sample['event'], sample['event'])}"
                ),
                output_path=output_path,
            )

            print("-" * 70)
            print(f"Policy : {policy_name}")
            print(f"Goal   : {goal_id}")
            print(f"Success rate : {summary['success_rate']:.2%}")
            print(f"Collision rate: {summary['collision_rate']:.2%}")
            print(f"Timeout rate  : {summary['timeout_rate']:.2%}")
            print(f"Avg steps     : {summary['avg_steps']}")
            print(f"Avg reward    : {summary['avg_reward']}")
            print(f"Avg final dist: {summary['avg_final_distance_px']} px")
            print(f"Saved trajectory: {output_path}")

    fieldnames = [
        "policy",
        "goal_id",
        "episodes",
        "success_count",
        "collision_count",
        "timeout_count",
        "success_rate",
        "collision_rate",
        "timeout_rate",
        "avg_steps",
        "avg_reward",
        "avg_final_distance_px",
    ]

    with SUMMARY_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_summaries)

    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    print("=" * 70)
    print("Policy evaluation saved:")
    print(f"- {SUMMARY_CSV}")
    print(f"- {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
