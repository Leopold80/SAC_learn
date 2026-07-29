"""Build stable JSON summaries for Go2 locomotion training runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import stable_baselines3 as sb3

from sac_experiments.config import ExperimentConfig
from sac_experiments.env_utils import best_eval_reward


def _module_parameter_count(module: Any) -> int:
    return sum(p.numel() for p in module.parameters())


def _optimizer_parameter_count(*optimizers: Any) -> int:
    unique: dict[int, Any] = {}
    for opt in optimizers:
        for group in opt.param_groups:
            for p in group["params"]:
                unique[id(p)] = p
    return sum(p.numel() for p in unique.values())


def model_parameter_counts(model: Any, algorithm: str) -> dict[str, int]:
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
    before_eval: tuple[float, float],
    after_eval: tuple[float, float],
    final_model_path: Path | None,
    best_model_path: Path | None,
    eval_log_path: Path,
    tensorboard_log: Path | None,
    training_wall_time_seconds: float,
    sampled_transitions: int,
    policy_class: str,
    parameter_counts: dict[str, int],
) -> dict[str, Any]:
    learning_rate = {"initial": config.learning_rate, "schedule": config.learning_rate_schedule}
    algorithm_kwargs = dict(config.sac if config.algorithm == "SAC" else config.ppo)
    if config.algorithm == "SAC":
        algo_metrics = {
            "gradient_updates_per_transition": (
                config.sac["gradient_steps"] / (config.sac["train_freq"] * config.n_envs)
            ),
        }
    else:
        rollout_size = config.n_envs * config.ppo["n_steps"]
        minibatches = rollout_size // config.ppo["batch_size"]
        algo_metrics = {
            "rollout_size_transitions": rollout_size,
            "minibatches_per_epoch": minibatches,
            "optimizer_steps_per_rollout": minibatches * config.ppo["n_epochs"],
            "sample_reuse_epochs": config.ppo["n_epochs"],
        }

    return {
        "config": str(config.config_path),
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
        "worker_seeds": [config.seed + i for i in range(config.n_envs)],
        "eval_seed": config.seed + config.n_envs,
        "raw_obs_dim": config.raw_obs_dim,
        "action_dim": config.action_dim,
        "learning_rate": learning_rate,
        **algorithm_kwargs,
        "policy_kwargs": {"net_arch": list(config.policy_net_arch)},
        "trainable_parameter_count": parameter_counts["policy_total"],
        "parameter_counts": parameter_counts,
        "eval_episodes": config.eval_episodes,
        "eval_freq": config.eval_freq,
        "callback_freq_vec_steps": config.eval_freq // config.n_envs,
        "training_wall_time_seconds": training_wall_time_seconds,
        "sampled_transitions": sampled_transitions,
        "sample_throughput_transitions_per_second": (
            sampled_transitions / training_wall_time_seconds
        ),
        **algo_metrics,
        "before_training": {"mean_reward": before_eval[0], "std_reward": before_eval[1]},
        "after_training": {"mean_reward": after_eval[0], "std_reward": after_eval[1]},
        "best_eval_mean_reward": best_eval_reward(eval_log_path),
        "final_model_path": str(final_model_path) if final_model_path else None,
        "best_model_path": str(best_model_path) if best_model_path else None,
        "tensorboard_log": str(tensorboard_log) if tensorboard_log else None,
    }


def write_experiment_summary(
    config: ExperimentConfig,
    summaries: Sequence[dict[str, Any]],
) -> Path:
    experiment_summary = {
        "config": str(config.config_path),
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "timesteps": config.timesteps,
        "frame_stack": config.frame_stack,
        "n_envs": config.n_envs,
        "run": summaries[0],
    }
    summary_path = config.output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")
    return summary_path
