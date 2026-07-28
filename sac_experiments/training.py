"""Unified LunarLander SAC and PPO training workflow."""

from __future__ import annotations

import argparse
import json
from time import perf_counter
from collections.abc import Sequence
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import stable_baselines3 as sb3
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
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
    is_rbf_variant,
    rbf_num_centers,
    tensorboard_run_name,
    uses_action_history,
    variant_network_summary,
    variant_policy,
    variant_policy_kwargs,
)


EvaluationCallbackFactory = Callable[[ExperimentConfig, Variant], BaseCallback | None]


@dataclass(frozen=True)
class TrainingRunOptions:
    """Optional runtime behavior used by searches without changing normal training."""

    evaluation_callback_factory: EvaluationCallbackFactory | None = None
    persist_models: bool = True
    persist_checkpoints: bool = True
    persist_tensorboard: bool = True
    persist_monitor: bool = True
    final_eval_episodes: int | None = None
    final_eval_seed: int | None = None


def parse_args(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train configured SAC or PPO variants on LunarLanderContinuous-v3."
    )
    config_group = parser.add_mutually_exclusive_group()
    config_group.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML experiment config. Default: {default_config_path}",
    )
    config_group.add_argument(
        "--search-config",
        type=Path,
        default=None,
        help="YAML hyperparameter-search config.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume an existing study selected by --search-config.",
    )
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="Statistically revalidate completed trials from --search-config.",
    )
    args = parser.parse_args(argv)
    if (args.resume or args.revalidate) and args.search_config is None:
        parser.error("--resume and --revalidate require --search-config.")
    if args.resume and args.revalidate:
        parser.error("--resume cannot be combined with --revalidate.")
    return args


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


def rbf_config_for_summary(
    config: ExperimentConfig,
    variant: Variant,
) -> dict[str, Any] | None:
    if not is_rbf_variant(variant):
        return None

    rbf = config.rbf
    summary = {
        "num_centers": rbf_num_centers(config, variant),
        "center_init_range": rbf["center_init_range"],
        "initial_bandwidth": rbf["initial_bandwidth"],
        "min_bandwidth": rbf["min_bandwidth"],
        "centers": "learnable",
        "bandwidths": "learnable diagonal per-center widths",
        "basis": "exp(-0.5 * sum(((x-c)/sigma)^2))",
    }
    if variant == "sac_rbf_matched":
        summary["capacity_match"] = "SAC actor + online twin-Q optimizer parameters"
    if variant == "sac_rbf_actor_mlp_critic_matched":
        summary["capacity_match"] = "SAC actor optimizer parameters"
    return summary


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
    """Record module and optimizer capacity without treating target critics as optimized."""

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
    algorithm_metrics: dict[str, Any]
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
        # Preserved for existing analysis scripts. For SAC it includes the target
        # critic, so use parameter_counts.optimized_total for capacity matching.
        "trainable_parameter_count": parameter_counts["policy_total"],
        "parameter_counts": parameter_counts,
        "ltc": ltc_config_for_summary(config, variant),
        "rbf": rbf_config_for_summary(config, variant),
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


def train_variant(
    config: ExperimentConfig,
    variant: Variant,
    device: str,
    options: TrainingRunOptions | None = None,
) -> dict[str, Any]:
    print(f"\n=== Training variant: {variant} ===")

    options = options or TrainingRunOptions()

    variant_output_dir = config.output_dir / variant
    monitor_dir = variant_output_dir / "monitor" if options.persist_monitor else None
    best_model_dir = variant_output_dir / "best_model"
    checkpoint_dir = variant_output_dir / "checkpoints"
    final_model_path = variant_output_dir / "final_model"
    best_model_path = best_model_dir / "best_model.zip"
    summary_path = variant_output_dir / "eval_summary.json"
    eval_log_path = variant_output_dir / "eval_logs" / "evaluations.npz"
    tb_run_name = tensorboard_run_name(variant)

    for directory in (
        variant_output_dir,
        config.tensorboard_log if options.persist_tensorboard else None,
        best_model_dir if options.persist_models else None,
        checkpoint_dir if options.persist_checkpoints and options.persist_models else None,
    ):
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

    use_action_history = uses_action_history(variant)
    with ExitStack() as env_stack:
        train_env = env_stack.enter_context(
            closing(
                make_lunarlander_vec_env(
                    n_envs=config.n_envs,
                    seed=config.seed,
                    frame_stack=config.frame_stack,
                    monitor_dir=monitor_dir / "train" if monitor_dir else None,
                    use_action_history=use_action_history,
                )
            )
        )
        eval_env = env_stack.enter_context(
            closing(
                make_lunarlander_env(
                    config.seed + config.n_envs,
                    config.frame_stack,
                    monitor_dir / "eval" if monitor_dir else None,
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
        model_class = SAC if config.algorithm == "SAC" else PPO
        algorithm_kwargs = dict(config.sac if config.algorithm == "SAC" else config.ppo)
        selected_policy = variant_policy(config, variant)
        policy_kwargs = variant_policy_kwargs(config, variant, raw_obs_dim=raw_obs_dim)
        model = model_class(
            selected_policy,
            train_env,
            learning_rate=learning_rate,
            **algorithm_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=(
                str(config.tensorboard_log) if options.persist_tensorboard else None
            ),
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
        after_eval_callback = (
            options.evaluation_callback_factory(config, variant)
            if options.evaluation_callback_factory
            else None
        )
        callbacks: list[BaseCallback] = [
            EvalCallback(
                eval_env,
                best_model_save_path=(str(best_model_dir) if options.persist_models else None),
                log_path=str(variant_output_dir / "eval_logs"),
                eval_freq=callback_freq,
                n_eval_episodes=config.eval_episodes,
                deterministic=True,
                render=False,
                callback_after_eval=after_eval_callback,
            )
        ]
        if options.persist_checkpoints and options.persist_models:
            callbacks.append(
                CheckpointCallback(
                    save_freq=callback_freq,
                    save_path=str(checkpoint_dir),
                    name_prefix=f"{config.algorithm.lower()}_lunarlander_{variant}",
                    save_replay_buffer=False,
                    save_vecnormalize=False,
                )
            )

        training_started_at = perf_counter()
        model.learn(
            total_timesteps=config.timesteps,
            callback=CallbackList(callbacks),
            log_interval=4,
            tb_log_name=tb_run_name,
            progress_bar=config.progress_bar,
        )
        training_wall_time_seconds = perf_counter() - training_started_at
        tensorboard_run_dir = (
            Path(model.logger.dir)
            if options.persist_tensorboard and model.logger.dir
            else (config.tensorboard_log if options.persist_tensorboard else None)
        )
        if options.persist_models:
            model.save(final_model_path)
            evaluation_model = model_class.load(final_model_path, env=eval_env, device=device)
        else:
            evaluation_model = model
        final_eval_episodes = options.final_eval_episodes or config.eval_episodes
        final_eval_env = eval_env
        if options.final_eval_seed is not None:
            final_eval_env = env_stack.enter_context(
                closing(
                    make_lunarlander_env(
                        options.final_eval_seed,
                        config.frame_stack,
                        None,
                        use_action_history=use_action_history,
                    )
                )
            )
        after_eval = evaluate(
            evaluation_model,
            final_eval_env,
            final_eval_episodes,
            f"{variant} after training",
        )

        summary = build_variant_summary(
            config=config,
            variant=variant,
            before_eval=before_eval,
            after_eval=after_eval,
            final_model_path=(
                final_model_path.with_suffix(".zip") if options.persist_models else None
            ),
            best_model_path=best_model_path if options.persist_models else None,
            eval_log_path=eval_log_path,
            tensorboard_log=tensorboard_run_dir,
            raw_obs_dim=raw_obs_dim,
            action_dim=action_dim,
            training_wall_time_seconds=training_wall_time_seconds,
            sampled_transitions=model.num_timesteps,
            policy_class=model.policy.__class__.__name__,
            parameter_counts=model_parameter_counts(model, config.algorithm),
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def ensure_fresh_run_paths(
    config: ExperimentConfig,
    options: TrainingRunOptions | None = None,
) -> None:
    options = options or TrainingRunOptions()
    paths: list[tuple[str, Path]] = [("output directory", config.output_dir)]
    if options.persist_tensorboard:
        paths.append(("TensorBoard directory", config.tensorboard_log))
    for label, path in paths:
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(
                f"Refusing to overwrite a non-empty {label}: {path}. "
                "Set a unique output.run_tag in the YAML config."
            )


def run_experiment(
    config: ExperimentConfig,
    options: TrainingRunOptions | None = None,
) -> Path:
    options = options or TrainingRunOptions()
    ensure_fresh_run_paths(config, options)
    device = configure_torch(config.device, config.allow_cpu)
    set_random_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if options.persist_tensorboard:
        config.tensorboard_log.mkdir(parents=True, exist_ok=True)

    summaries = [
        train_variant(config, variant, device, options) for variant in config.variants
    ]
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
    if args.search_config is not None:
        from sac_experiments.hyperparameter_search import (
            load_search_config,
            run_revalidation,
            run_search,
        )

        search_config = load_search_config(args.search_config)
        if args.revalidate:
            run_revalidation(search_config)
        else:
            run_search(search_config, resume=args.resume)
        return
    run_experiment(load_config(args.config or default_config_path))
