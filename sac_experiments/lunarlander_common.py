"""Common LunarLander SAC experiment utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import gymnasium as gym
import torch
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


ENV_ID = "LunarLanderContinuous-v3"
DEFAULT_TIMESTEPS = 500_000
DEFAULT_EVAL_EPISODES = 10
DEFAULT_EVAL_FREQ = 10_000
DEFAULT_SEED = 42
DEFAULT_DEVICE = "cuda"
DEFAULT_FRAME_STACK = 4
DEFAULT_OUTPUT_DIR = Path("outputs/sac_lunarlander_compare")
DEFAULT_TENSORBOARD_LOG = Path("runs/sac_lunarlander_compare")

SAC_CONFIG = {
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "ent_coef": "auto",
    "gamma": 0.99,
    "tau": 0.01,
    "train_freq": 1,
    "gradient_steps": 1,
    "learning_starts": 10_000,
}


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return schedule


def configure_torch(device: str, allow_cpu: bool) -> str:
    cuda_available = torch.cuda.is_available()
    requested_cuda = str(device).startswith("cuda")

    if requested_cuda and not cuda_available:
        if allow_cpu:
            print("CUDA is not available; falling back to CPU because --allow-cpu was set.")
            return "cpu"
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a GPU in this session. "
            "Run outside the Codex sandbox or pass --allow-cpu for a short debug run only."
        )

    if device == "auto" and not cuda_available and not allow_cpu:
        raise RuntimeError(
            "No CUDA device is available and CPU fallback is disabled. Use a GPU-enabled "
            "session, or pass --allow-cpu for a debug run."
        )

    if cuda_available:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")

    return device


def make_lunarlander_env(
    seed: int,
    frame_stack: int = DEFAULT_FRAME_STACK,
    monitor_dir: Path | None = None,
    render_mode: str | None = None,
) -> Monitor:
    env = gym.make(ENV_ID, render_mode=render_mode)
    if frame_stack > 1:
        env = FrameStackObservation(env, stack_size=frame_stack)
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_dir / "monitor.csv") if monitor_dir else None)


def evaluate(model, eval_env: Monitor, episodes: int, label: str) -> tuple[float, float]:
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=episodes,
        deterministic=True,
        warn=False,
    )
    print(f"{label}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
    return float(mean_reward), float(std_reward)


def best_eval_reward(eval_log_path: Path) -> float | None:
    if not eval_log_path.exists():
        return None

    import numpy as np

    data = np.load(eval_log_path)
    if "results" not in data.files or len(data["results"]) == 0:
        return None
    return float(data["results"].mean(axis=1).max())

