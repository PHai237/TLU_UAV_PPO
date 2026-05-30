from __future__ import annotations

import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from envs.tlu_uav_env import TluUavEnv  # noqa: E402


MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

GOAL_IDS = [
    "drop_t45",
    "drop_library",
    "drop_k1",
    "drop_c1",
    "drop_dorm4",
]


class MultiGoalTluUavEnv(TluUavEnv):
    """
    Training wrapper: each reset cycles through multiple dropoff goals.
    This makes PPO learn a more general navigation policy instead of memorizing
    only one route.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(goal_id="drop_t45", *args, **kwargs)
        self.goal_index = 0

    def reset(self, *, seed=None, options=None):
        goal_id = GOAL_IDS[self.goal_index % len(GOAL_IDS)]
        self.goal_index += 1

        if options is None:
            options = {}
        options["goal_id"] = goal_id

        return super().reset(seed=seed, options=options)


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = MultiGoalTluUavEnv(max_steps=500)
    env = Monitor(env)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        tensorboard_log=str(LOGS_DIR),
    )

    total_timesteps = 50_000

    print("=" * 70)
    print("Training PPO on TLU UAV environment")
    print(f"Total timesteps: {total_timesteps}")
    print("=" * 70)

    model.learn(total_timesteps=total_timesteps)

    output_path = MODELS_DIR / "ppo_tlu_uav_preliminary"
    model.save(output_path)

    print("=" * 70)
    print(f"Saved PPO model: {output_path}.zip")


if __name__ == "__main__":
    main()