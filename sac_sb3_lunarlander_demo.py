"""Train a Stable-Baselines3 SAC agent on LunarLanderContinuous-v3.

This is a more visual and longer-running SAC demo than Pendulum-v1.

Recommended setup:

    conda create -n sac_sb3_demo --clone cybernetic_env
    conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt

Quick smoke run:

    conda run -n sac_sb3_demo python sac_sb3_lunarlander_demo.py \
        --timesteps 2000 --eval-episodes 2 --eval-freq 1000

Fuller RL-Zoo style run:

    conda run -n sac_sb3_demo python sac_sb3_lunarlander_demo.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import gymnasium as gym
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed


ENV_ID = "LunarLanderContinuous-v3"
DEFAULT_TIMESTEPS = 500_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_FREQ = 10_000
DEFAULT_SEED = 42
DEFAULT_OUTPUT_DIR = Path("outputs/sac_lunarlander")
DEFAULT_TENSORBOARD_LOG = Path("runs/sac_lunarlander")
DEFAULT_DEVICE = "cuda"


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Stable-Baselines3 SAC agent on LunarLanderContinuous-v3."
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


def configure_torch(args: argparse.Namespace) -> str:
    cuda_available = torch.cuda.is_available()
    requested_cuda = str(args.device).startswith("cuda")

    if requested_cuda and not cuda_available:
        if args.allow_cpu:
            print("CUDA is not available; falling back to CPU because --allow-cpu was set.")
            return "cpu"
        raise RuntimeError(
            "CUDA was requested for LunarLanderContinuous training, but PyTorch cannot "
            "see a GPU in this session. Move to a GPU-enabled SSH node/session or run "
            "with --device cpu --allow-cpu for debugging only."
        )

    if args.device == "auto" and not cuda_available and not args.allow_cpu:
        raise RuntimeError(
            "No CUDA device is available and CPU fallback is disabled. Use a GPU-enabled "
            "session for normal training, or pass --allow-cpu for a debug run."
        )

    if cuda_available:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    return args.device


def make_monitored_env(seed: int, monitor_dir: Path | None = None) -> Monitor:
    env = gym.make(ENV_ID)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_dir / "monitor.csv") if monitor_dir else None)


def evaluate(model: SAC, eval_env: Monitor, episodes: int, label: str) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
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
        "learning_rate": "linear_7.3e-4",
        "buffer_size": 1_000_000,
        "batch_size": 256,
        "ent_coef": "auto",
        "gamma": 0.99,
        "tau": 0.01,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 10_000,
        "policy_kwargs": {"net_arch": [400, 300]},
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
    device = configure_torch(args)
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
        learning_rate=linear_schedule(7.3e-4),
        buffer_size=1_000_000,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.01,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        policy_kwargs={"net_arch": [400, 300]},
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

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
