"""Train a Stable-Baselines3 SAC agent on Pendulum-v1.

Run in an isolated conda environment so the shared cybernetic_env is not
modified:

    conda create -n sac_sb3_demo --clone cybernetic_env
    conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
    conda run -n sac_sb3_demo python sac_sb3_pendulum_demo.py

For a quick smoke run:

    conda run -n sac_sb3_demo python sac_sb3_pendulum_demo.py \
        --timesteps 1000 --eval-episodes 2 --eval-freq 500

See SAC_SB3_TRICKS.md for the SB3/RL-Zoo tricks used by this demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


ENV_ID = "Pendulum-v1"
DEFAULT_TIMESTEPS = 20_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_FREQ = 1_000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("outputs/sac_pendulum")
DEFAULT_TENSORBOARD_LOG = Path("runs/sac_pendulum")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Stable-Baselines3 SAC agent on Pendulum-v1."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps. Default: {DEFAULT_TIMESTEPS}",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Number of episodes for each evaluation. Default: {DEFAULT_EVAL_EPISODES}",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=DEFAULT_EVAL_FREQ,
        help=f"Evaluate and checkpoint every N environment steps. Default: {DEFAULT_EVAL_FREQ}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device passed to SB3, for example 'auto', 'cpu', or 'cuda'. Default: auto",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for models, checkpoints, and summaries. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=Path,
        default=DEFAULT_TENSORBOARD_LOG,
        help=f"TensorBoard log directory. Default: {DEFAULT_TENSORBOARD_LOG}",
    )
    return parser.parse_args()


def make_monitored_env(seed: int, monitor_dir: Path | None = None) -> Monitor:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_dir / "monitor.csv") if monitor_dir else None)


def evaluate(
    model: SAC,
    eval_env: Monitor,
    n_eval_episodes: int,
    label: str,
) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        warn=False,
    )
    print(f"{label}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return float(mean_reward), float(std_reward)


def build_summary(
    args: argparse.Namespace,
    before_eval: tuple[float, float],
    after_eval: tuple[float, float],
    final_model_path: Path,
    best_model_path: Path,
) -> dict[str, Any]:
    return {
        "env_id": ENV_ID,
        "algorithm": "SAC",
        "policy": "MlpPolicy",
        "seed": args.seed,
        "timesteps": args.timesteps,
        "learning_rate": 1e-3,
        "ent_coef": "auto",
        "eval_episodes": args.eval_episodes,
        "eval_freq": args.eval_freq,
        "before_training": {
            "mean_reward": before_eval[0],
            "std_reward": before_eval[1],
        },
        "after_training": {
            "mean_reward": after_eval[0],
            "std_reward": after_eval[1],
        },
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "tensorboard_log": str(args.tensorboard_log),
    }


def main() -> None:
    args = parse_args()

    output_dir = args.output_dir
    monitor_dir = output_dir / "monitor"
    best_model_dir = output_dir / "best_model"
    checkpoint_dir = output_dir / "checkpoints"
    final_model_path = output_dir / "final_model"
    best_model_path = best_model_dir / "best_model.zip"
    summary_path = output_dir / "eval_summary.json"

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.tensorboard_log.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    train_env = make_monitored_env(args.seed, monitor_dir / "train")
    eval_env = make_monitored_env(args.seed + 1, monitor_dir / "eval")

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=1e-3,
        ent_coef="auto",
        tensorboard_log=str(args.tensorboard_log),
        seed=args.seed,
        device=args.device,
        verbose=1,
    )

    print("Evaluating initial policy...")
    before_eval = evaluate(model, eval_env, args.eval_episodes, "Before training")

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(best_model_dir),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        deterministic=True,
        render=False,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=args.eval_freq,
        save_path=str(checkpoint_dir),
        name_prefix="sac_pendulum",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    callbacks = CallbackList([eval_callback, checkpoint_callback])

    print(f"Training SAC on {ENV_ID} for {args.timesteps} timesteps...")
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        log_interval=4,
        tb_log_name="SAC",
        progress_bar=True,
    )

    model.save(final_model_path)
    print(f"Saved final model to {final_model_path.with_suffix('.zip')}")

    loaded_model = SAC.load(final_model_path, env=eval_env, device=args.device)

    print("Evaluating final policy...")
    after_eval = evaluate(loaded_model, eval_env, args.eval_episodes, "After training")

    summary = build_summary(
        args=args,
        before_eval=before_eval,
        after_eval=after_eval,
        final_model_path=final_model_path.with_suffix(".zip"),
        best_model_path=best_model_path,
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Best model:  {best_model_path}")
    print(f"Final model: {final_model_path.with_suffix('.zip')}")
    print(f"Summary:     {summary_path}")
    print(
        "Reward delta: "
        f"{after_eval[0] - before_eval[0]:.2f} "
        f"({before_eval[0]:.2f} -> {after_eval[0]:.2f})"
    )

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
