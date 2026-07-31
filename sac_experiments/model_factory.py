"""Construct Stable-Baselines3 models from validated experiment settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC

from sac_experiments.config import ExperimentConfig
from sac_experiments.env_utils import linear_schedule
from sac_experiments.policies import TanhActorCriticPolicy


def algorithm_class(algorithm: str) -> type[SAC] | type[PPO]:
    return SAC if algorithm == "SAC" else PPO


def build_model(
    config: ExperimentConfig,
    train_env: Any,
    device: str,
    tensorboard_log: Path | None,
) -> SAC | PPO:
    """Build one untrained SAC/PPO model for Go2 locomotion."""

    learning_rate = (
        linear_schedule(config.learning_rate)
        if config.learning_rate_schedule == "linear"
        else config.learning_rate
    )
    algorithm_kwargs = dict(config.sac if config.algorithm == "SAC" else config.ppo)
    policy_kwargs = dict(
        net_arch=list(config.policy_net_arch),
    )
    policy: str | type[TanhActorCriticPolicy] = config.policy
    if config.algorithm == "PPO":
        policy_kwargs["log_std_init"] = config.ppo_policy_kwargs["log_std_init"]
        if config.policy == "TanhMlpPolicy":
            policy = TanhActorCriticPolicy
            policy_kwargs.update({
                "log_std_min": config.ppo_policy_kwargs["log_std_min"],
                "log_std_max": config.ppo_policy_kwargs["log_std_max"],
            })

    model_class = algorithm_class(config.algorithm)
    return model_class(
        policy,
        train_env,
        learning_rate=learning_rate,
        **algorithm_kwargs,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        seed=config.seed,
        device=device,
        verbose=1,
    )
