"""Construct Stable-Baselines3 models from validated experiment settings.

This module is deliberately small: configuration decides *what* to build,
``training.py`` decides *when* to train it, and the policy implementations stay
in their own LTC/RBF modules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from stable_baselines3 import PPO, SAC

from sac_experiments.config import ExperimentConfig
from sac_experiments.lunarlander_common import linear_schedule
from sac_experiments.variants import Variant, variant_policy, variant_policy_kwargs


def algorithm_class(algorithm: str) -> type[SAC] | type[PPO]:
    """Return the SB3 algorithm class selected by a validated config."""

    return SAC if algorithm == "SAC" else PPO


def build_model(
    config: ExperimentConfig,
    variant: Variant,
    train_env: Any,
    raw_obs_dim: int,
    device: str,
    tensorboard_log: Path | None,
) -> SAC | PPO:
    """Build one untrained SAC/PPO model for a configured variant."""

    learning_rate = (
        linear_schedule(config.learning_rate)
        if config.learning_rate_schedule == "linear"
        else config.learning_rate
    )
    model_class = algorithm_class(config.algorithm)
    algorithm_kwargs = dict(config.sac if config.algorithm == "SAC" else config.ppo)
    return model_class(
        variant_policy(config, variant),
        train_env,
        learning_rate=learning_rate,
        **algorithm_kwargs,
        policy_kwargs=variant_policy_kwargs(
            config,
            variant,
            raw_obs_dim=raw_obs_dim,
        ),
        tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        seed=config.seed,
        device=device,
        verbose=1,
    )
