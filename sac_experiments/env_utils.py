"""Common environment utilities — CUDA, evaluation, vectorised env factory."""

from __future__ import annotations

import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv


# ── CUDA ──────────────────────────────────────────────────────────────────────

def configure_torch(device: str, allow_cpu: bool = False) -> str:
    """Resolve the requested device and return the best available option."""
    requested = device.strip().lower()
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "cuda":
        if allow_cpu:
            print("[env_utils] CUDA not available — falling back to CPU.")
            return "cpu"
        raise RuntimeError("CUDA is not available and allow_cpu is False.")
    return requested


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    model,
    eval_env: VecEnv,
    n_episodes: int,
    label: str = "",
) -> tuple[float, float]:
    """Deterministic evaluation returning (mean_reward, std_reward)."""
    prefix = f"[{label}] " if label else ""
    all_rewards: list[float] = []
    obs = eval_env.reset()

    while len(all_rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = eval_env.step(action)
        for idx, done in enumerate(dones):
            if done:
                episode_reward = infos[idx].get("episode", {}).get("r", 0.0)
                all_rewards.append(float(episode_reward))

    rewards = np.array(all_rewards[:n_episodes])
    mean, std = float(np.mean(rewards)), float(np.std(rewards))
    print(f"{prefix}mean_reward={mean:.2f} ± {std:.2f}  (n={len(rewards)})")
    return mean, std


def best_eval_reward(eval_log_path) -> float | None:
    """Read the best evaluation mean reward from an EvalCallback .npz log."""
    try:
        data = np.load(eval_log_path)
        results = data["results"]
        mean_rewards = results[:, 0].astype(float)
        return float(np.max(mean_rewards))
    except (FileNotFoundError, KeyError, IndexError):
        return None


# ── Learning rate schedule ───────────────────────────────────────────────────

def linear_schedule(initial_value: float):
    """SB3-compatible linear decay from ``initial_value`` to 0."""
    def _schedule(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return _schedule


# ── Vectorised environment factory ────────────────────────────────────────────

def make_vec_env(
    env_id: str,
    n_envs: int,
    seed: int,
    *,
    env_kwargs: dict | None = None,
    monitor_dir: str | None = None,
) -> VecEnv:
    """Create a SubprocVecEnv (n_envs > 1) or DummyVecEnv (n_envs == 1)."""
    vec_env = sb3_make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
        monitor_dir=monitor_dir,
        env_kwargs=env_kwargs or {},
    )
    return vec_env
