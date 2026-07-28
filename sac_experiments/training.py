"""Top-to-bottom orchestration for LunarLander training runs."""

from __future__ import annotations

import json
from time import perf_counter
from contextlib import ExitStack, closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed

from sac_experiments.config import ExperimentConfig
from sac_experiments.lunarlander_common import (
    configure_torch,
    evaluate,
    lunarlander_dimensions,
    make_lunarlander_env,
    make_lunarlander_vec_env,
)
from sac_experiments.model_factory import algorithm_class, build_model
from sac_experiments.reporting import (
    build_variant_summary,
    model_parameter_counts,
    write_experiment_summary,
)
from sac_experiments.variants import (
    Variant,
    tensorboard_run_name,
    uses_action_history,
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


@dataclass(frozen=True)
class VariantRunPaths:
    """Resolved artifact paths for one variant."""

    output_dir: Path
    monitor_dir: Path | None
    best_model_dir: Path
    checkpoint_dir: Path
    final_model_path: Path
    best_model_path: Path
    summary_path: Path
    eval_log_path: Path


def prepare_variant_paths(
    config: ExperimentConfig,
    variant: Variant,
    options: TrainingRunOptions,
) -> VariantRunPaths:
    """Resolve and create only the artifact directories enabled for this run."""

    output_dir = config.output_dir / variant
    paths = VariantRunPaths(
        output_dir=output_dir,
        monitor_dir=output_dir / "monitor" if options.persist_monitor else None,
        best_model_dir=output_dir / "best_model",
        checkpoint_dir=output_dir / "checkpoints",
        final_model_path=output_dir / "final_model",
        best_model_path=output_dir / "best_model" / "best_model.zip",
        summary_path=output_dir / "eval_summary.json",
        eval_log_path=output_dir / "eval_logs" / "evaluations.npz",
    )
    directories = [
        paths.output_dir,
        config.tensorboard_log if options.persist_tensorboard else None,
        paths.best_model_dir if options.persist_models else None,
        (
            paths.checkpoint_dir
            if options.persist_checkpoints and options.persist_models
            else None
        ),
    ]
    for directory in directories:
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
    return paths


def build_training_callbacks(
    config: ExperimentConfig,
    variant: Variant,
    eval_env: Any,
    paths: VariantRunPaths,
    options: TrainingRunOptions,
) -> CallbackList:
    """Build evaluation and optional checkpoint callbacks for one variant."""

    callback_freq = config.eval_freq // config.n_envs
    after_eval_callback = (
        options.evaluation_callback_factory(config, variant)
        if options.evaluation_callback_factory
        else None
    )
    callbacks: list[BaseCallback] = [
        EvalCallback(
            eval_env,
            best_model_save_path=(
                str(paths.best_model_dir) if options.persist_models else None
            ),
            log_path=str(paths.output_dir / "eval_logs"),
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
                save_path=str(paths.checkpoint_dir),
                name_prefix=f"{config.algorithm.lower()}_lunarlander_{variant}",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    return CallbackList(callbacks)


def train_variant(
    config: ExperimentConfig,
    variant: Variant,
    device: str,
    options: TrainingRunOptions | None = None,
) -> dict[str, Any]:
    print(f"\n=== Training variant: {variant} ===")

    options = options or TrainingRunOptions()
    paths = prepare_variant_paths(config, variant, options)
    use_action_history = uses_action_history(variant)
    with ExitStack() as env_stack:
        train_env = env_stack.enter_context(
            closing(
                make_lunarlander_vec_env(
                    n_envs=config.n_envs,
                    seed=config.seed,
                    frame_stack=config.frame_stack,
                    monitor_dir=(
                        paths.monitor_dir / "train" if paths.monitor_dir else None
                    ),
                    use_action_history=use_action_history,
                )
            )
        )
        eval_env = env_stack.enter_context(
            closing(
                make_lunarlander_env(
                    config.seed + config.n_envs,
                    config.frame_stack,
                    paths.monitor_dir / "eval" if paths.monitor_dir else None,
                    use_action_history=use_action_history,
                )
            )
        )
        raw_obs_dim, action_dim = lunarlander_dimensions(eval_env)
        model = build_model(
            config=config,
            variant=variant,
            train_env=train_env,
            raw_obs_dim=raw_obs_dim,
            device=device,
            tensorboard_log=(
                config.tensorboard_log if options.persist_tensorboard else None
            ),
        )

        before_eval = evaluate(
            model,
            eval_env,
            config.eval_episodes,
            f"{variant} before training",
        )

        training_started_at = perf_counter()
        model.learn(
            total_timesteps=config.timesteps,
            callback=build_training_callbacks(
                config,
                variant,
                eval_env,
                paths,
                options,
            ),
            log_interval=4,
            tb_log_name=tensorboard_run_name(variant),
            progress_bar=config.progress_bar,
        )
        training_wall_time_seconds = perf_counter() - training_started_at
        tensorboard_run_dir = (
            Path(model.logger.dir)
            if options.persist_tensorboard and model.logger.dir
            else (config.tensorboard_log if options.persist_tensorboard else None)
        )
        if options.persist_models:
            model.save(paths.final_model_path)
            model_class = algorithm_class(config.algorithm)
            evaluation_model = model_class.load(
                paths.final_model_path,
                env=eval_env,
                device=device,
            )
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
                paths.final_model_path.with_suffix(".zip")
                if options.persist_models
                else None
            ),
            best_model_path=paths.best_model_path if options.persist_models else None,
            eval_log_path=paths.eval_log_path,
            tensorboard_log=tensorboard_run_dir,
            raw_obs_dim=raw_obs_dim,
            action_dim=action_dim,
            training_wall_time_seconds=training_wall_time_seconds,
            sampled_transitions=model.num_timesteps,
            policy_class=model.policy.__class__.__name__,
            parameter_counts=model_parameter_counts(model, config.algorithm),
        )
        paths.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
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
    summary_path = write_experiment_summary(config, summaries)

    print("\n=== Experiment complete ===")
    print(f"Summary: {summary_path}")
    for summary in summaries:
        print(
            f"{summary['variant']}: final={summary['after_training']['mean_reward']:.2f}, "
            f"best_eval={summary['best_eval_mean_reward']}"
        )
    return summary_path
