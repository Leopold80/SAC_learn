"""Standalone, non-stacked LunarLander SAC baseline workflow.

This remains a small teaching baseline. The YAML-driven comparison workflow is
implemented separately in :mod:`sac_experiments.lunarlander_compare`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from sac_experiments.lunarlander_common import (
    DEFAULT_DEVICE,
    DEFAULT_EVAL_EPISODES,
    DEFAULT_EVAL_FREQ,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LEARNING_RATE_NAME,
    DEFAULT_POLICY_NET_ARCH,
    DEFAULT_SEED,
    DEFAULT_TIMESTEPS,
    ENV_ID,
    SAC_CONFIG,
    configure_torch,
    evaluate,
    linear_schedule,
    make_lunarlander_env,
)


DEFAULT_OUTPUT_DIR = Path("outputs/sac_lunarlander")
DEFAULT_TENSORBOARD_LOG = Path("runs/sac_lunarlander")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a non-stacked Stable-Baselines3 SAC baseline on "
            "LunarLanderContinuous-v3."
        )
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--eval-episodes", type=int, default=DEFAULT_EVAL_EPISODES)
    parser.add_argument("--eval-freq", type=int, default=DEFAULT_EVAL_FREQ)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        help="Torch device passed to SB3. Default: cuda",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU fallback for debugging only. Normal training should use CUDA.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tensorboard-log", type=Path, default=DEFAULT_TENSORBOARD_LOG)
    return parser.parse_args()


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
        "frame_stack": 1,
        "learning_rate": DEFAULT_LEARNING_RATE_NAME,
        **SAC_CONFIG,
        "policy_kwargs": {"net_arch": list(DEFAULT_POLICY_NET_ARCH)},
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
    device = configure_torch(args.device, args.allow_cpu)

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
    train_env = make_lunarlander_env(args.seed, frame_stack=1, monitor_dir=monitor_dir / "train")
    eval_env = make_lunarlander_env(args.seed + 1, frame_stack=1, monitor_dir=monitor_dir / "eval")

    try:
        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=linear_schedule(DEFAULT_LEARNING_RATE),
            **SAC_CONFIG,
            policy_kwargs={"net_arch": list(DEFAULT_POLICY_NET_ARCH)},
            tensorboard_log=str(args.tensorboard_log),
            seed=args.seed,
            device=device,
            verbose=1,
        )

        print("Evaluating initial policy...")
        before_eval = evaluate(model, eval_env, args.eval_episodes, "Before training")

        callbacks = CallbackList(
            [
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_model_dir),
                    log_path=str(output_dir / "eval_logs"),
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.eval_episodes,
                    deterministic=True,
                    render=False,
                ),
                CheckpointCallback(
                    save_freq=args.eval_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix="sac_lunarlander",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
            ]
        )

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

        loaded_model = SAC.load(final_model_path, env=eval_env, device=device)
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
    finally:
        train_env.close()
        eval_env.close()
