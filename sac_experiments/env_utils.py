"""Common environment utilities — CUDA, evaluation, vectorised env factory."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gymnasium as gym
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


class FixedSeedResetWrapper(gym.Wrapper):
    """Cycle through the same reset seeds for every checkpoint evaluation."""

    def __init__(self, env: gym.Env, reset_seeds: Sequence[int]):
        super().__init__(env)
        if not reset_seeds:
            raise ValueError("reset_seeds must not be empty.")
        self._reset_seeds = tuple(int(seed) for seed in reset_seeds)
        self._seed_index = 0

    def rewind_seed_cycle(self) -> None:
        self._seed_index = 0

    def set_reset_seeds(self, reset_seeds: Sequence[int]) -> None:
        if not reset_seeds:
            raise ValueError("reset_seeds must not be empty.")
        self._reset_seeds = tuple(int(seed) for seed in reset_seeds)
        self._seed_index = 0

    def reset(self, **kwargs: Any):
        kwargs["seed"] = self._reset_seeds[
            self._seed_index % len(self._reset_seeds)
        ]
        self._seed_index += 1
        return self.env.reset(**kwargs)


def rewind_fixed_seed_cycle(eval_env: VecEnv) -> None:
    """Rewind all fixed-seed wrappers in a vector environment."""
    try:
        eval_env.env_method("rewind_seed_cycle")
    except AttributeError:
        pass

def evaluate(
    model,
    eval_env: VecEnv,
    n_episodes: int,
    label: str = "",
) -> tuple[float, float]:
    """Deterministic evaluation returning (mean_reward, std_reward)."""
    prefix = f"[{label}] " if label else ""
    all_rewards: list[float] = []
    rewind_fixed_seed_cycle(eval_env)
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


def evaluate_detailed(
    model,
    eval_env: VecEnv,
    n_episodes: int,
    diagnostic_keys: Sequence[str],
    label: str = "",
) -> dict[str, Any]:
    """Deterministic evaluation with per-episode Go2 diagnostics."""
    prefix = f"[{label}] " if label else ""
    rewards: list[float] = []
    diagnostics: list[list[float]] = []
    rewind_fixed_seed_cycle(eval_env)
    obs = eval_env.reset()

    while len(rewards) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = eval_env.step(action)
        for index, done in enumerate(dones):
            if not done:
                continue
            info = infos[index]
            rewards.append(float(info.get("episode", {}).get("r", 0.0)))
            diagnostics.append([
                float(info.get(key, np.nan))
                for key in diagnostic_keys
            ])

    reward_array = np.asarray(rewards[:n_episodes], dtype=float)
    diagnostic_array = np.asarray(diagnostics[:n_episodes], dtype=float)
    mean = float(np.mean(reward_array))
    std = float(np.std(reward_array))
    print(f"{prefix}mean_reward={mean:.2f} ± {std:.2f}  (n={len(reward_array)})")
    return {
        "mean_reward": mean,
        "std_reward": std,
        "episode_rewards": reward_array,
        "diagnostic_names": np.asarray(tuple(diagnostic_keys)),
        "diagnostics": diagnostic_array,
        "metric_means": {
            key: float(np.nanmean(diagnostic_array[:, index]))
            for index, key in enumerate(diagnostic_keys)
        },
    }


def best_eval_reward(eval_log_path) -> float | None:
    """Read the best evaluation mean reward from an EvalCallback .npz log."""
    try:
        data = np.load(eval_log_path)
        results = data["results"]
        if results.ndim != 2 or results.shape[1] == 0:
            return None
        mean_rewards = results.mean(axis=1, dtype=float)
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
    monitor_info_keys: Sequence[str] = (),
    fixed_reset_seeds: Sequence[int] | None = None,
) -> VecEnv:
    """Create a SubprocVecEnv (n_envs > 1) or DummyVecEnv (n_envs == 1)."""
    vec_env = sb3_make_vec_env(
        env_id,
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=SubprocVecEnv if n_envs > 1 else None,
        monitor_dir=monitor_dir,
        env_kwargs=env_kwargs or {},
        monitor_kwargs={"info_keywords": tuple(monitor_info_keys)},
        wrapper_class=FixedSeedResetWrapper if fixed_reset_seeds else None,
        wrapper_kwargs=(
            {"reset_seeds": tuple(fixed_reset_seeds)}
            if fixed_reset_seeds
            else None
        ),
    )
    return vec_env
