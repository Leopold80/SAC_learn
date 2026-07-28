"""Build stable JSON summaries for LunarLander training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import stable_baselines3 as sb3

from sac_experiments.config import ExperimentConfig
from sac_experiments.lunarlander_common import best_eval_reward
from sac_experiments.variants import (
    Variant,
    feature_extractor_name,
    uses_action_history,
    variant_ltc_summary,
    variant_network_summary,
    variant_policy_kwargs,
    variant_rbf_summary,
)


def _module_parameter_count(module: Any) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _optimizer_parameter_count(*optimizers: Any) -> int:
    unique_parameters: dict[int, Any] = {}
    for optimizer in optimizers:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                unique_parameters[id(parameter)] = parameter
    return sum(parameter.numel() for parameter in unique_parameters.values())


def model_parameter_counts(model: Any, algorithm: str) -> dict[str, int]:
    """Record module and optimizer capacity without counting target critics as optimized."""

    policy_total = _module_parameter_count(model.policy)
    if algorithm != "SAC":
        return {
            "policy_total": policy_total,
            "optimized_total": _optimizer_parameter_count(model.policy.optimizer),
        }
    return {
        "policy_total": policy_total,
        "actor": _module_parameter_count(model.actor),
        "online_critic": _module_parameter_count(model.critic),
        "target_critic": _module_parameter_count(model.critic_target),
        "optimized_total": _optimizer_parameter_count(
            model.actor.optimizer,
            model.critic.optimizer,
        ),
    }


def build_variant_summary(
    config: ExperimentConfig,
    variant: Variant,
    before_eval: tuple[float, float],
    after_eval: tuple[float, float],
    final_model_path: Path | None,
    best_model_path: Path | None,
    eval_log_path: Path,
    tensorboard_log: Path | None,
    raw_obs_dim: int,
    action_dim: int,
    training_wall_time_seconds: float,
    sampled_transitions: int,
    policy_class: str,
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    learning_rate = {
        "initial": config.learning_rate,
        "schedule": config.learning_rate_schedule,
    }
    algorithm_kwargs = dict(config.sac if config.algorithm == "SAC" else config.ppo)
    if config.algorithm == "SAC":
        algorithm_metrics = {
            "gradient_updates_per_transition": (
                config.sac["gradient_steps"]
                / (config.sac["train_freq"] * config.n_envs)
            ),
        }
    else:
        rollout_size = config.n_envs * config.ppo["n_steps"]
        minibatches_per_epoch = rollout_size // config.ppo["batch_size"]
        algorithm_metrics = {
            "rollout_size_transitions": rollout_size,
            "minibatches_per_epoch": minibatches_per_epoch,
            "optimizer_steps_per_rollout": (
                minibatches_per_epoch * config.ppo["n_epochs"]
            ),
            "sample_reuse_epochs": config.ppo["n_epochs"],
        }
    resolved_policy_kwargs = variant_policy_kwargs(
        config,
        variant,
        raw_obs_dim=raw_obs_dim,
    )
    return {
        "config": str(config.config_path),
        "variant": variant,
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "algorithm_source": f"stable-baselines3=={sb3.__version__}",
        "policy": config.policy,
        "policy_class": policy_class,
        "seed": config.seed,
        "timesteps": config.timesteps,
        "frame_stack": config.frame_stack,
        "n_envs": config.n_envs,
        "vec_env": "SubprocVecEnv" if config.n_envs > 1 else "DummyVecEnv",
        "worker_seeds": [config.seed + index for index in range(config.n_envs)],
        "eval_seed": config.seed + config.n_envs,
        "uses_action_history": uses_action_history(variant),
        "raw_obs_dim": raw_obs_dim,
        "action_dim": action_dim,
        "learning_rate": learning_rate,
        **algorithm_kwargs,
        "policy_kwargs": {
            "net_arch": resolved_policy_kwargs["net_arch"],
            "features_extractor": feature_extractor_name(variant),
        },
        "network": variant_network_summary(config, variant),
        "trainable_parameter_count": parameter_counts["policy_total"],
        "parameter_counts": parameter_counts,
        "ltc": variant_ltc_summary(config, variant),
        "rbf": variant_rbf_summary(config, variant),
        "eval_episodes": config.eval_episodes,
        "eval_freq": config.eval_freq,
        "callback_freq_vec_steps": config.eval_freq // config.n_envs,
        "training_wall_time_seconds": training_wall_time_seconds,
        "sampled_transitions": sampled_transitions,
        "sample_throughput_transitions_per_second": (
            sampled_transitions / training_wall_time_seconds
        ),
        **algorithm_metrics,
        "before_training": {
            "mean_reward": before_eval[0],
            "std_reward": before_eval[1],
        },
        "after_training": {
            "mean_reward": after_eval[0],
            "std_reward": after_eval[1],
        },
        "best_eval_mean_reward": best_eval_reward(eval_log_path),
        "final_model_path": str(final_model_path) if final_model_path else None,
        "best_model_path": str(best_model_path) if best_model_path else None,
        "tensorboard_log": str(tensorboard_log) if tensorboard_log else None,
    }


def write_experiment_summary(
    config: ExperimentConfig,
    summaries: Sequence[dict[str, Any]],
) -> Path:
    """Persist the top-level summary without changing the established JSON shape."""

    experiment_summary = {
        "config": str(config.config_path),
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "timesteps": config.timesteps,
        "frame_stack": config.frame_stack,
        "n_envs": config.n_envs,
        "variants": list(summaries),
    }
    filename = (
        f"experiment_summary_{config.variants[0]}.json"
        if len(config.variants) == 1
        else "experiment_summary.json"
    )
    summary_path = config.output_dir / filename
    summary_path.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    return summary_path
