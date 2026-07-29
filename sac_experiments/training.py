"""Top-to-bottom orchestration for Go2 locomotion training runs."""

from __future__ import annotations

import json
from time import perf_counter
from contextlib import closing
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
from sac_experiments.env_utils import configure_torch, evaluate, make_vec_env
from sac_experiments.model_factory import algorithm_class, build_model
from sac_experiments.reporting import (
    build_variant_summary,
    model_parameter_counts,
    write_experiment_summary,
)

# Register Go2 environment on import
import sac_experiments.go2_env  # noqa: F401 — triggers gymnasium registration

EvaluationCallbackFactory = Callable[[ExperimentConfig], BaseCallback | None]


@dataclass(frozen=True)
class TrainingRunOptions:
    evaluation_callback_factory: EvaluationCallbackFactory | None = None
    persist_models: bool = True
    persist_checkpoints: bool = True
    persist_tensorboard: bool = True
    persist_monitor: bool = True
    final_eval_episodes: int | None = None
    final_eval_seed: int | None = None


@dataclass(frozen=True)
class VariantRunPaths:
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
    options: TrainingRunOptions,
) -> VariantRunPaths:
    output_dir = config.output_dir
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
        paths.checkpoint_dir if options.persist_checkpoints and options.persist_models else None,
    ]
    for d in directories:
        if d is not None:
            d.mkdir(parents=True, exist_ok=True)
    return paths


def build_training_callbacks(
    config: ExperimentConfig,
    eval_env: Any,
    paths: VariantRunPaths,
    options: TrainingRunOptions,
) -> CallbackList:
    callback_freq = config.eval_freq // config.n_envs
    after_eval_callback = (
        options.evaluation_callback_factory(config)
        if options.evaluation_callback_factory
        else None
    )
    callbacks: list[BaseCallback] = [
        EvalCallback(
            eval_env,
            best_model_save_path=str(paths.best_model_dir) if options.persist_models else None,
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
                name_prefix=f"{config.algorithm.lower()}_go2",
                save_replay_buffer=False,
                save_vecnormalize=False,
            )
        )
    return CallbackList(callbacks)


def train_variant(
    config: ExperimentConfig,
    device: str,
    options: TrainingRunOptions | None = None,
) -> dict[str, Any]:
    print(f"\n=== Training {config.algorithm} on {config.env_id} ===")
    options = options or TrainingRunOptions()
    paths = prepare_variant_paths(config, options)

    with closing(make_vec_env(
        config.env_id,
        n_envs=config.n_envs,
        seed=config.seed,
        monitor_dir=str(paths.monitor_dir / "train") if paths.monitor_dir else None,
    )) as train_env:
        with closing(make_vec_env(
            config.env_id,
            n_envs=1,
            seed=config.seed + config.n_envs,
            monitor_dir=str(paths.monitor_dir / "eval") if paths.monitor_dir else None,
        )) as eval_env:
            model = build_model(
                config=config,
                train_env=train_env,
                device=device,
                tensorboard_log=config.tensorboard_log if options.persist_tensorboard else None,
            )

            before_eval = evaluate(model, eval_env, config.eval_episodes, "before training")

            training_started_at = perf_counter()
            model.learn(
                total_timesteps=config.timesteps,
                callback=build_training_callbacks(config, eval_env, paths, options),
                log_interval=4,
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
                    paths.final_model_path, env=eval_env, device=device
                )
            else:
                evaluation_model = model

            after_eval = evaluate(evaluation_model, eval_env, config.eval_episodes, "after training")

            summary = build_variant_summary(
                config=config,
                before_eval=before_eval,
                after_eval=after_eval,
                final_model_path=paths.final_model_path.with_suffix(".zip") if options.persist_models else None,
                best_model_path=paths.best_model_path if options.persist_models else None,
                eval_log_path=paths.eval_log_path,
                tensorboard_log=tensorboard_run_dir,
                training_wall_time_seconds=training_wall_time_seconds,
                sampled_transitions=model.num_timesteps,
                policy_class=model.policy.__class__.__name__,
                parameter_counts=model_parameter_counts(model, config.algorithm),
            )
            paths.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary


def ensure_fresh_run_paths(config: ExperimentConfig, options: TrainingRunOptions | None = None) -> None:
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

    summary = train_variant(config, device, options)
    summary_path = write_experiment_summary(config, [summary])

    print("\n=== Experiment complete ===")
    print(f"Summary: {summary_path}")
    print(
        f"final={summary['after_training']['mean_reward']:.2f}, "
        f"best_eval={summary['best_eval_mean_reward']}"
    )
    return summary_path
