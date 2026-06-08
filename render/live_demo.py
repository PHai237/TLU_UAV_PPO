from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

try:
    from stable_baselines3 import PPO
except ImportError:
    PPO = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.tlu_uav_env import TluUavEnv  # noqa: E402


TYPE_COLORS = [
    "#b8bdc2",  # road
    "#b56a5a",  # building
    "#3f8f4f",  # green
    "#edf1df",  # yard
    "#e25d5d",  # no_fly
    "#f4b942",  # pickup
    "#2878bd",  # dropoff
]

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "ppo_tlu_uav_100k_radius30.zip"

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
    "ppo": "PPO",
    "greedy": "Greedy",
    "random": "Random",
}

EVENT_LABELS = {
    "goal_reached": "Hoàn thành",
    "timeout": "Quá thời gian",
    "collision": "Va chạm",
    "move": "Đang di chuyển",
    "reset": "Khởi tạo",
}


def unpack_reset(result):
    if isinstance(result, tuple):
        return result[0], result[1]
    return result, {}


def unpack_step(result):
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, reward, terminated or truncated, info

    obs, reward, done, info = result
    return obs, reward, done, info


def find_point(points, point_id: str):
    for p in points:
        if p["id"] == point_id:
            return float(p["x"]), float(p["y"]), p["label"]
    raise ValueError(f"POI not found: {point_id}")


def draw_static_labels(ax, env: TluUavEnv) -> None:
    for p in env.pois["pickup_points"]:
        ax.scatter(
            float(p["x"]),
            float(p["y"]),
            s=120,
            marker="o",
            edgecolors="black",
            label="Cổng sau",
        )
        ax.text(
            float(p["x"]) + 10,
            float(p["y"]) - 10,
            p["label"],
            fontsize=8,
            weight="bold",
        )

    for p in env.pois["dropoff_points"]:
        ax.scatter(
            float(p["x"]),
            float(p["y"]),
            s=130,
            marker="X",
            edgecolors="black",
        )
        ax.text(
            float(p["x"]) + 10,
            float(p["y"]) - 10,
            p["label"],
            fontsize=8,
            weight="bold",
        )


def choose_action(policy: str, env: TluUavEnv, obs, model):
    if policy == "ppo":
        if model is None:
            raise RuntimeError("PPO model is not loaded.")
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    if policy == "greedy":
        return env.greedy_action()

    if policy == "random":
        return env.sample_random_action()

    raise ValueError(f"Unsupported policy: {policy}")


def draw_sensor_lines(ax, env: TluUavEnv, sensor_artists: list):
    for artist in sensor_artists:
        artist.remove()
    sensor_artists.clear()

    if not hasattr(env, "_ray_cast"):
        return

    ray_values = env._ray_cast()
    ray_distances = ray_values * env.ray_range

    for i, dist in enumerate(ray_distances):
        angle = 2.0 * np.pi * i / env.ray_count
        x2 = env.position[0] + np.cos(angle) * dist
        y2 = env.position[1] + np.sin(angle) * dist

        line = ax.plot(
            [env.position[0], x2],
            [env.position[1], y2],
            linewidth=0.8,
            alpha=0.65,
            zorder=2,
        )[0]
        sensor_artists.append(line)


def run_live_demo(
    goal_id: str,
    policy: str,
    model_path: Path,
    max_steps: int,
    delay: float,
    show_sensors: bool,
):
    env = TluUavEnv(goal_id=goal_id, max_steps=max_steps)

    model = None
    if policy == "ppo":
        if PPO is None:
            raise ImportError("stable-baselines3 is not installed.")
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        model = PPO.load(str(model_path))

    obs, info = unpack_reset(env.reset(options={"goal_id": goal_id}))

    cmap = ListedColormap(TYPE_COLORS)

    plt.ion()
    fig, ax = plt.subplots(figsize=(14, 9))

    ax.imshow(
        env.type_map,
        cmap=cmap,
        origin="upper",
        vmin=0,
        vmax=len(TYPE_COLORS) - 1,
        interpolation="nearest",
    )

    draw_static_labels(ax, env)

    start_x, start_y, start_label = find_point(
        env.pois["pickup_points"],
        "pickup_back_gate",
    )
    goal_x, goal_y, goal_label = find_point(
        env.pois["dropoff_points"],
        goal_id,
    )

    path_x = [float(env.position[0])]
    path_y = [float(env.position[1])]

    path_artist, = ax.plot(
        path_x,
        path_y,
        linewidth=2.2,
        label="Quỹ đạo",
        zorder=4,
    )

    drone_artist, = ax.plot(
        [env.position[0]],
        [env.position[1]],
        marker="o",
        markersize=11,
        markeredgecolor="black",
        linestyle="None",
        label="UAV",
        zorder=6,
    )

    goal_artist, = ax.plot(
        [goal_x],
        [goal_y],
        marker="X",
        markersize=13,
        markeredgecolor="black",
        linestyle="None",
        label="Điểm giao",
        zorder=6,
    )

    sensor_artists = []

    ax.set_xlim(0, env.width)
    ax.set_ylim(env.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(labelsize=8, colors="#666666")
    ax.legend(loc="upper right")
    status_box = ax.text(
        0.02,
        0.02,
        "",
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999", "alpha": 0.9},
        zorder=50,
    )

    done = False
    total_reward = 0.0
    final_info = info

    while not done:
        action = choose_action(policy, env, obs, model)

        result = env.step(action)
        obs, reward, done, final_info = unpack_step(result)
        total_reward += float(reward)

        path_x.append(float(env.position[0]))
        path_y.append(float(env.position[1]))

        path_artist.set_data(path_x, path_y)
        drone_artist.set_data([env.position[0]], [env.position[1]])
        goal_artist.set_data([goal_x], [goal_y])

        if show_sensors:
            draw_sensor_lines(ax, env, sensor_artists)

        event = str(final_info.get("event", "-"))
        event_label = EVENT_LABELS.get(event, event)
        goal_label = GOAL_LABELS.get(goal_id, goal_id)
        policy_label = POLICY_LABELS.get(policy, policy.upper())
        title = f"{policy_label}: Cổng sau → {goal_label} | {event_label}"
        ax.set_title(title)
        status_box.set_text(
            f"Mục tiêu: {goal_label}\n"
            f"Bước: {final_info.get('steps', env.steps)}/{max_steps}\n"
            f"Khoảng cách: {final_info.get('distance_to_goal', env.prev_distance):.2f} px\n"
            f"Trạng thái: {event_label}"
        )

        fig.canvas.draw_idle()
        plt.pause(delay)

    print("=" * 70)
    print("LIVE DEMO FINISHED")
    print("=" * 70)
    print(f"Policy        : {POLICY_LABELS.get(policy, policy)}")
    print(f"Goal          : {GOAL_LABELS.get(goal_id, goal_id)}")
    print(f"Event         : {EVENT_LABELS.get(str(final_info.get('event')), final_info.get('event'))}")
    print(f"Success       : {final_info.get('success')}")
    print(f"Collision     : {final_info.get('collision')}")
    print(f"Timeout       : {final_info.get('timeout')}")
    print(f"Steps         : {final_info.get('steps')}")
    print(f"Total reward  : {total_reward:.2f}")
    print(f"Final distance: {final_info.get('distance_to_goal'):.2f}")

    plt.ioff()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Live 2D render demo for the current TLU UAV PPO project."
    )

    parser.add_argument(
        "--goal",
        choices=GOAL_IDS,
        default="drop_t45",
        help="Dropoff goal to demo.",
    )

    parser.add_argument(
        "--policy",
        choices=["ppo", "greedy", "random"],
        default="ppo",
        help="Policy used in the live demo.",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to PPO model.",
    )

    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum number of environment steps.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.03,
        help="Pause time between rendered frames.",
    )

    parser.add_argument(
        "--hide-sensors",
        action="store_true",
        help="Hide ray-casting sensor lines. This is the default presentation mode.",
    )

    parser.add_argument(
        "--show-sensors",
        action="store_true",
        help="Show ray-casting sensor lines for technical explanation.",
    )

    args = parser.parse_args()

    run_live_demo(
        goal_id=args.goal,
        policy=args.policy,
        model_path=args.model,
        max_steps=args.max_steps,
        delay=args.delay,
        show_sensors=args.show_sensors and not args.hide_sensors,
    )


if __name__ == "__main__":
    main()
