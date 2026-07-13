"""Unified LunarLander SAC training workflow."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from collections.abc import Sequence
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any

import stable_baselines3 as sb3
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed

from sac_experiments.config import DEFAULT_CONFIG_PATH, ExperimentConfig, load_config
from sac_experiments.lunarlander_common import (
    best_eval_reward,
    configure_torch,
    evaluate,
    linear_schedule,
    lunarlander_dimensions,
    make_lunarlander_env,
    make_lunarlander_vec_env,
)
from sac_experiments.variants import (
    Variant,
    feature_extractor_name,
    tensorboard_run_name,
    uses_action_history,
    variant_policy_kwargs,
)


def parse_args(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train configured SAC variants on LunarLanderContinuous-v3."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=default_config_path,
        help=f"YAML experiment config. Default: {default_config_path}",
    )
    return parser.parse_args(argv)


def ltc_config_for_summary(
    config: ExperimentConfig,
    variant: Variant,
) -> dict[str, Any] | None:
    if variant not in {"ltc_simple", "ltc", "ltc_residual", "ltc_residual_action"}:
        return None

    ltc = config.ltc
    summary = {
        "liquid_hidden_dim": ltc["liquid_hidden_dim"],
        "features_dim": ltc["features_dim"],
        "dt": ltc["dt"],
    }
    if variant == "ltc":
        summary |= {
            "tau_min": ltc["tau_min"],
            "ode_unfolds": ltc["ode_unfolds"],
            "reversal_init_scale": ltc["reversal_init_scale"],
            "ode": "dx_i = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)",
        }
    if variant in {"ltc_residual", "ltc_residual_action"}:
        summary |= {
            "raw_features_dim": ltc["raw_features_dim"],
            "fusion_hidden_dim": ltc["fusion_hidden_dim"],
            "tau_min": ltc["tau_min"],
            "ode_unfolds": ltc["ode_unfolds"],
            "reversal_init_scale": ltc["reversal_init_scale"],
            "ode": "dx_i = -x_i/tau_i + sum_j f_ij(x_j,u;theta) * (A_ij - x_i)",
            "residual": "raw stacked observation projection + LTC feature concat",
        }
    return summary


def build_variant_summary(
    config: ExperimentConfig,
    variant: Variant,
    before_eval: tuple[float, float],
    after_eval: tuple[float, float],
    final_model_path: Path,
    best_model_path: Path,
    eval_log_path: Path,
    tensorboard_log: Path,
    raw_obs_dim: int,
    action_dim: int,
    training_wall_time_seconds: float,
    sampled_transitions: int,
) -> dict[str, Any]:
    learning_rate = {
        "initial": config.learning_rate,
        "schedule": config.learning_rate_schedule,
    }
    return {
        "config": str(config.config_path),
        "variant": variant,
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "algorithm_source": f"stable-baselines3=={sb3.__version__}",
        "policy": config.policy,
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
        **dict(config.sac),
        "policy_kwargs": {
            "net_arch": list(config.policy_net_arch),
            "features_extractor": feature_extractor_name(variant),
        },
        "ltc": ltc_config_for_summary(config, variant),
        "eval_episodes": config.eval_episodes,
        "eval_freq": config.eval_freq,
        "callback_freq_vec_steps": config.eval_freq // config.n_envs,
        "training_wall_time_seconds": training_wall_time_seconds,
        "sampled_transitions": sampled_transitions,
        "sample_throughput_transitions_per_second": (
            sampled_transitions / training_wall_time_seconds
        ),
        "gradient_updates_per_transition": (
            config.sac["gradient_steps"]
            / (config.sac["train_freq"] * config.n_envs)
        ),
        "before_training": {
            "mean_reward": before_eval[0],
            "std_reward": before_eval[1],
        },
        "after_training": {
            "mean_reward": after_eval[0],
            "std_reward": after_eval[1],
        },
        "best_eval_mean_reward": best_eval_reward(eval_log_path),
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path),
        "tensorboard_log": str(tensorboard_log),
    }


def train_variant(
    config: ExperimentConfig,
    variant: Variant,
    device: str,
) -> dict[str, Any]:
    print(f"\n=== Training variant: {variant} ===")

    variant_output_dir = config.output_dir / variant
    monitor_dir = variant_output_dir / "monitor"
    best_model_dir = variant_output_dir / "best_model"
    checkpoint_dir = variant_output_dir / "checkpoints"
    final_model_path = variant_output_dir / "final_model"
    best_model_path = best_model_dir / "best_model.zip"
    summary_path = variant_output_dir / "eval_summary.json"
    eval_log_path = variant_output_dir / "eval_logs" / "evaluations.npz"
    tb_run_name = tensorboard_run_name(variant)

    for directory in (
        variant_output_dir,
        config.tensorboard_log,
        best_model_dir,
        checkpoint_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    use_action_history = uses_action_history(variant)
    with ExitStack() as env_stack:
        train_env = env_stack.enter_context(
            closing(
                make_lunarlander_vec_env(
                    n_envs=config.n_envs,
                    seed=config.seed,
                    frame_stack=config.frame_stack,
                    monitor_dir=monitor_dir / "train",
                    use_action_history=use_action_history,
                )
            )
        )
        eval_env = env_stack.enter_context(
            closing(
                make_lunarlander_env(
                    config.seed + config.n_envs,
                    config.frame_stack,
                    monitor_dir / "eval",
                    use_action_history=use_action_history,
                )
            )
        )
        raw_obs_dim, action_dim = lunarlander_dimensions(eval_env)
        learning_rate = (
            linear_schedule(config.learning_rate)
            if config.learning_rate_schedule == "linear"
            else config.learning_rate
        )
        model = SAC(
            config.policy,
            train_env,
            learning_rate=learning_rate,
            **dict(config.sac),
            policy_kwargs=variant_policy_kwargs(config, variant, raw_obs_dim=raw_obs_dim),
            tensorboard_log=str(config.tensorboard_log),
            seed=config.seed,
            device=device,
            verbose=1,
        )

        before_eval = evaluate(
            model,
            eval_env,
            config.eval_episodes,
            f"{variant} before training",
        )
        callback_freq = config.eval_freq // config.n_envs
        callbacks = CallbackList(
            [
                EvalCallback(
                    eval_env,
                    best_model_save_path=str(best_model_dir),
                    log_path=str(variant_output_dir / "eval_logs"),
                    eval_freq=callback_freq,
                    n_eval_episodes=config.eval_episodes,
                    deterministic=True,
                    render=False,
                ),
                CheckpointCallback(
                    save_freq=callback_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"sac_lunarlander_{variant}",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                ),
            ]
        )

        training_started_at = perf_counter()
        model.learn(
            total_timesteps=config.timesteps,
            callback=callbacks,
            log_interval=4,
            tb_log_name=tb_run_name,
            progress_bar=config.progress_bar,
        )
        training_wall_time_seconds = perf_counter() - training_started_at
        tensorboard_run_dir = (
            Path(model.logger.dir) if model.logger.dir else config.tensorboard_log
        )
        model.save(final_model_path)
        loaded_model = SAC.load(final_model_path, env=eval_env, device=device)
        after_eval = evaluate(
            loaded_model,
            eval_env,
            config.eval_episodes,
            f"{variant} after training",
        )

        summary = build_variant_summary(
            config=config,
            variant=variant,
            before_eval=before_eval,
            after_eval=after_eval,
            final_model_path=final_model_path.with_suffix(".zip"),
            best_model_path=best_model_path,
            eval_log_path=eval_log_path,
            tensorboard_log=tensorboard_run_dir,
            raw_obs_dim=raw_obs_dim,
            action_dim=action_dim,
            training_wall_time_seconds=training_wall_time_seconds,
            sampled_transitions=model.num_timesteps,
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def ensure_fresh_run_paths(config: ExperimentConfig) -> None:
    for label, path in (
        ("output directory", config.output_dir),
        ("TensorBoard directory", config.tensorboard_log),
    ):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(
                f"Refusing to overwrite a non-empty {label}: {path}. "
                "Set a unique output.run_tag in the YAML config."
            )


def run_experiment(config: ExperimentConfig) -> Path:
    ensure_fresh_run_paths(config)
    device = configure_torch(config.device, config.allow_cpu)
    set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.tensorboard_log.mkdir(parents=True, exist_ok=True)

    summaries = [train_variant(config, variant, device) for variant in config.variants]
    experiment_summary = {
        "config": str(config.config_path),
        "env_id": config.env_id,
        "algorithm": config.algorithm,
        "seed": config.seed,
        "timesteps": config.timesteps,
        "frame_stack": config.frame_stack,
        "n_envs": config.n_envs,
        "variants": summaries,
    }
    if len(config.variants) == 1:
        summary_path = config.output_dir / f"experiment_summary_{config.variants[0]}.json"
    else:
        summary_path = config.output_dir / "experiment_summary.json"
    summary_path.write_text(json.dumps(experiment_summary, indent=2), encoding="utf-8")

    print("\n=== Experiment complete ===")
    print(f"Summary: {summary_path}")
    for summary in summaries:
        print(
            f"{summary['variant']}: final={summary['after_training']['mean_reward']:.2f}, "
            f"best_eval={summary['best_eval_mean_reward']}"
        )
    return summary_path


def main(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    args = parse_args(argv, default_config_path)
    run_experiment(load_config(args.config))
