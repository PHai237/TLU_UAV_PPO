from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train.train_ppo import MultiGoalTluUavEnv  # noqa: E402


MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune an existing PPO model on the final TLU map.")
    parser.add_argument(
        "--input-model",
        type=Path,
        default=MODELS_DIR / "ppo_tlu_uav_100k_radius30.zip",
        help="Existing PPO model to continue training from.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=MODELS_DIR / "ppo_tlu_uav_200k_finetuned_radius30",
        help="Output model path, with or without .zip suffix.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100_000,
        help="Additional training timesteps.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Fine-tuning learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=237,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_model.exists():
        raise FileNotFoundError(f"Input model not found: {args.input_model}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    env = MultiGoalTluUavEnv(max_steps=500, seed=args.seed)
    env = Monitor(env)

    model = PPO.load(args.input_model, env=env)
    model.learning_rate = args.learning_rate
    # PPO stores a learning-rate schedule when loading a saved model. Replace
    # it explicitly so fine-tuning really uses the requested rate on SB3
    # versions where assigning model.learning_rate alone is not sufficient.
    model.lr_schedule = lambda _: args.learning_rate

    print("=" * 70)
    print("Fine-tuning PPO on final TLU UAV map")
    print(f"Input model : {args.input_model}")
    print(f"Timesteps   : {args.timesteps}")
    print(f"LR          : {args.learning_rate}")
    print(f"Output model: {args.output_model}.zip")
    print("=" * 70)

    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=False)
    model.save(args.output_model)

    print("=" * 70)
    print(f"Saved fine-tuned PPO model: {args.output_model}.zip")


if __name__ == "__main__":
    main()
