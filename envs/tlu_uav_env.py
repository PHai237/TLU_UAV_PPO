from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None
    spaces = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OCCUPANCY_PATH = PROCESSED_DIR / "occupancy_grid.npy"
TYPE_MAP_PATH = PROCESSED_DIR / "type_map.npy"
HEIGHT_MAP_PATH = PROCESSED_DIR / "height_map.npy"
RISK_MAP_PATH = PROCESSED_DIR / "risk_map.npy"
POIS_PATH = PROCESSED_DIR / "pois.json"


ACTION_DIRECTIONS = np.array(
    [
        [0.0, -1.0],   # 0 up
        [0.0, 1.0],    # 1 down
        [-1.0, 0.0],   # 2 left
        [1.0, 0.0],    # 3 right
        [1.0, -1.0],   # 4 up-right
        [-1.0, -1.0],  # 5 up-left
        [1.0, 1.0],    # 6 down-right
        [-1.0, 1.0],   # 7 down-left
    ],
    dtype=np.float32,
)

# Normalize diagonal actions.
for i in range(len(ACTION_DIRECTIONS)):
    norm = np.linalg.norm(ACTION_DIRECTIONS[i])
    if norm > 0:
        ACTION_DIRECTIONS[i] /= norm


class TluUavEnv(gym.Env if gym is not None else object):
    """
    PPO-ready 2D UAV navigation environment for the TLU semantic campus map.

    Observation vector:
    [
        x_norm,
        y_norm,
        goal_dx_norm,
        goal_dy_norm,
        distance_to_goal_norm,
        local_risk,
        ray_0, ray_1, ..., ray_7
    ]

    Action space:
        Discrete(8): 8 movement directions.

    Termination:
        - reaches goal
        - collides with obstacle/no-fly area
        - goes out of map
        - timeout
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        goal_id: str = "drop_t45",
        step_size: float = 8.0,
        goal_radius: float = 14.0,
        max_steps: int = 500,
        ray_count: int = 8,
        ray_range: float = 120.0,
        ray_step: float = 4.0,
        seed: int | None = None,
    ) -> None:
        self.occupancy_grid = np.load(OCCUPANCY_PATH)
        self.type_map = np.load(TYPE_MAP_PATH)
        self.height_map = np.load(HEIGHT_MAP_PATH)
        self.risk_map = np.load(RISK_MAP_PATH)

        with POIS_PATH.open("r", encoding="utf-8") as f:
            self.pois = json.load(f)

        self.height, self.width = self.occupancy_grid.shape

        self.goal_id = goal_id
        self.step_size = float(step_size)
        self.goal_radius = float(goal_radius)
        self.max_steps = int(max_steps)
        self.ray_count = int(ray_count)
        self.ray_range = float(ray_range)
        self.ray_step = float(ray_step)

        self.rng = np.random.default_rng(seed)

        self.start_xy = self._find_point(self.pois["pickup_points"], "pickup_back_gate")
        self.goal_xy = self._find_point(self.pois["dropoff_points"], self.goal_id)

        self.position = self.start_xy.astype(np.float32).copy()
        self.prev_distance = self._distance_to_goal()
        self.steps = 0
        self.done = False
        self.last_info: dict[str, Any] = {}

        # Gymnasium compatibility.
        if spaces is not None:
            self.action_space = spaces.Discrete(8)
            obs_dim = 6 + self.ray_count
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(obs_dim,),
                dtype=np.float32,
            )

    def _find_point(self, points: list[dict[str, Any]], point_id: str) -> np.ndarray:
        for p in points:
            if p["id"] == point_id:
                return np.array([float(p["x"]), float(p["y"])], dtype=np.float32)
        raise ValueError(f"POI not found: {point_id}")

    def _distance_to_goal(self) -> float:
        return float(np.linalg.norm(self.goal_xy - self.position))

    def _is_inside_map(self, x: float, y: float) -> bool:
        return 0 <= int(round(x)) < self.width and 0 <= int(round(y)) < self.height

    def _is_blocked(self, x: float, y: float) -> bool:
        if not self._is_inside_map(x, y):
            return True

        ix = int(round(x))
        iy = int(round(y))
        return bool(self.occupancy_grid[iy, ix] == 1)

    def _segment_collides(self, start: np.ndarray, end: np.ndarray) -> bool:
        dist = float(np.linalg.norm(end - start))
        sample_count = max(2, int(math.ceil(dist / 2.0)))

        for i in range(sample_count + 1):
            t = i / sample_count
            p = start * (1.0 - t) + end * t
            if self._is_blocked(float(p[0]), float(p[1])):
                return True

        return False

    def _local_risk(self, radius: int = 6) -> float:
        x = int(round(self.position[0]))
        y = int(round(self.position[1]))

        x1 = max(0, x - radius)
        x2 = min(self.width, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(self.height, y + radius + 1)

        patch = self.risk_map[y1:y2, x1:x2]
        if patch.size == 0:
            return 1.0

        return float(np.max(patch))

    def _ray_cast(self) -> np.ndarray:
        rays = []

        for i in range(self.ray_count):
            angle = 2.0 * math.pi * i / self.ray_count
            direction = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)

            hit_distance = self.ray_range

            d = 0.0
            while d <= self.ray_range:
                p = self.position + direction * d
                if self._is_blocked(float(p[0]), float(p[1])):
                    hit_distance = d
                    break
                d += self.ray_step

            # Normalize: 1.0 means no obstacle nearby, 0.0 means obstacle very close.
            rays.append(hit_distance / self.ray_range)

        return np.array(rays, dtype=np.float32)

    def _get_obs(self) -> np.ndarray:
        x_norm = (self.position[0] / self.width) * 2.0 - 1.0
        y_norm = (self.position[1] / self.height) * 2.0 - 1.0

        goal_delta = self.goal_xy - self.position
        goal_dx_norm = goal_delta[0] / self.width
        goal_dy_norm = goal_delta[1] / self.height

        distance = self._distance_to_goal()
        max_distance = math.sqrt(self.width**2 + self.height**2)
        distance_norm = distance / max_distance

        local_risk = self._local_risk()
        rays = self._ray_cast()

        obs = np.concatenate(
            [
                np.array(
                    [
                        x_norm,
                        y_norm,
                        goal_dx_norm,
                        goal_dy_norm,
                        distance_norm,
                        local_risk,
                    ],
                    dtype=np.float32,
                ),
                rays,
            ]
        )

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if options is not None and "goal_id" in options:
            self.goal_id = str(options["goal_id"])
            self.goal_xy = self._find_point(self.pois["dropoff_points"], self.goal_id)

        self.position = self.start_xy.astype(np.float32).copy()
        self.prev_distance = self._distance_to_goal()
        self.steps = 0
        self.done = False
        self.last_info = {
            "event": "reset",
            "goal_id": self.goal_id,
            "distance_to_goal": self.prev_distance,
        }

        obs = self._get_obs()

        if gym is not None:
            return obs, self.last_info

        return obs

    def step(self, action: int):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        action = int(action)
        if action < 0 or action >= len(ACTION_DIRECTIONS):
            raise ValueError(f"Invalid action: {action}")

        self.steps += 1

        old_pos = self.position.copy()
        old_distance = self._distance_to_goal()

        direction = ACTION_DIRECTIONS[action]
        new_pos = self.position + direction * self.step_size

        collision = self._segment_collides(old_pos, new_pos)

        reward = -0.10  # step penalty
        terminated = False
        truncated = False
        event = "move"

        if collision:
            reward -= 100.0
            terminated = True
            event = "collision"
        else:
            self.position = new_pos.astype(np.float32)

            new_distance = self._distance_to_goal()
            progress = old_distance - new_distance

            # Progress reward: positive if moving closer, negative if moving away.
            reward += progress * 0.20

            # Risk penalty.
            local_risk = self._local_risk()
            reward -= local_risk * 2.0

            if new_distance <= self.goal_radius:
                reward += 100.0
                terminated = True
                event = "goal_reached"

        if self.steps >= self.max_steps and not terminated:
            reward -= 30.0
            truncated = True
            event = "timeout"

        self.done = terminated or truncated
        self.prev_distance = self._distance_to_goal()

        info = {
            "event": event,
            "goal_id": self.goal_id,
            "position_x": float(self.position[0]),
            "position_y": float(self.position[1]),
            "distance_to_goal": float(self.prev_distance),
            "steps": self.steps,
            "reward": float(reward),
            "collision": bool(collision),
            "success": event == "goal_reached",
            "timeout": event == "timeout",
        }
        self.last_info = info

        obs = self._get_obs()

        if gym is not None:
            return obs, float(reward), terminated, truncated, info

        # Fallback old-style API when Gymnasium is not installed.
        done = terminated or truncated
        return obs, float(reward), done, info

    def sample_random_action(self) -> int:
        return int(self.rng.integers(0, len(ACTION_DIRECTIONS)))

    def greedy_action(self) -> int:
        """
        Select action that minimizes Euclidean distance to goal while avoiding
        immediate collision when possible.
        """
        best_action = 0
        best_score = float("inf")

        for action, direction in enumerate(ACTION_DIRECTIONS):
            candidate = self.position + direction * self.step_size

            if self._segment_collides(self.position, candidate):
                continue

            distance = float(np.linalg.norm(self.goal_xy - candidate))
            risk_penalty = self._risk_at(candidate) * 20.0
            score = distance + risk_penalty

            if score < best_score:
                best_score = score
                best_action = action

        return best_action

    def _risk_at(self, position: np.ndarray) -> float:
        x = int(round(float(position[0])))
        y = int(round(float(position[1])))

        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return 1.0

        return float(self.risk_map[y, x])


def smoke_test() -> None:
    env = TluUavEnv(goal_id="drop_t45")
    obs = env.reset()
    print("Smoke test: TluUavEnv")
    print(f"Observation shape: {np.asarray(obs[0] if isinstance(obs, tuple) else obs).shape}")

    total_reward = 0.0
    for _ in range(20):
        action = env.greedy_action()
        result = env.step(action)

        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            done = terminated or truncated
        else:
            obs, reward, done, info = result

        total_reward += reward

        if done:
            break

    print(f"Last info: {env.last_info}")
    print(f"Total reward after smoke run: {total_reward:.2f}")


if __name__ == "__main__":
    smoke_test()