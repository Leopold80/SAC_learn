"""Top-to-bottom orchestration for Go2 locomotion training runs."""

from __future__ import annotations

import json
import os
from time import perf_counter
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed

from sac_experiments.config import ExperimentConfig
from sac_experiments.env_utils import (
    configure_torch,
    evaluate,
    evaluate_detailed,
    make_vec_env,
    rewind_fixed_seed_cycle,
)
from sac_experiments.go2_env import EPISODE_INFO_KEYS
from sac_experiments.model_factory import algorithm_class, build_model
from sac_experiments.reporting import (
    build_variant_summary,
    model_parameter_counts,
    write_experiment_summary,
)

# Register Go2 environment on import
import sac_experiments.go2_env  # noqa: F401 — triggers gymnasium registration

EvaluationCallbackFactory = Callable[[ExperimentConfig], BaseCallback | None]

EVAL_DIAGNOSTIC_KEYS = (
    *EPISODE_INFO_KEYS,
    "episode_reward_tracking_lin_vel",
    "episode_reward_tracking_ang_vel",
    "episode_reward_orientation",
    "episode_reward_action_rate",
    "episode_reward_torque",
    "episode_reward_total",
)


class ZeroActionPolicy:
    """Minimal predict interface for a paired standstill baseline."""

    def __init__(self, action_dim: int) -> None:
        self._action_dim = action_dim

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        batch_size = observation.shape[0]
        return np.zeros((batch_size, self._action_dim), dtype=np.float32), None


def paired_bootstrap_interval(
    differences: np.ndarray,
    *,
    seed: int = 0,
    samples: int = 10_000,
) -> tuple[float, float]:
    """Return a deterministic 95% bootstrap CI for paired episode differences."""
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("differences must be a non-empty one-dimensional array.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(upper)


class LocomotionMetricsCallback(BaseCallback):
    """Aggregate Go2-specific rollout metrics into TensorBoard."""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self._step_metrics: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            for key in (
                "velocity_x",
                "velocity_y",
                "yaw_rate",
                "action_saturation",
                "torque_sq",
                "reward_tracking_lin_vel",
                "reward_tracking_ang_vel",
                "reward_orientation",
                "reward_action_rate",
                "reward_torque",
                "reward_total",
            ):
                if key in info:
                    self._step_metrics.setdefault(key, []).append(float(info[key]))
        return True

    def _on_rollout_end(self) -> None:
        for key, values in self._step_metrics.items():
            if values:
                self.logger.record(f"go2/{key}", float(np.mean(values)))
        self._step_metrics.clear()

        policy = self.model.policy
        if hasattr(policy, "effective_log_std"):
            with torch.no_grad():
                effective_log_std = policy.effective_log_std()
                effective_std = torch.exp(effective_log_std)
                self.logger.record(
                    "go2/effective_std_mean",
                    float(effective_std.mean().cpu()),
                )
                self.logger.record(
                    "go2/effective_std_max",
                    float(effective_std.max().cpu()),
                )

                observations = torch.as_tensor(
                    self.model.rollout_buffer.observations.reshape(
                        -1,
                        self.model.observation_space.shape[0],
                    ),
                    device=self.model.device,
                )
                distribution = policy.get_distribution(observations)
                gaussian_mean = distribution.distribution.mean
                deterministic_actions = torch.tanh(gaussian_mean)
                self.logger.record(
                    "go2/gaussian_mean_abs",
                    float(gaussian_mean.abs().mean().cpu()),
                )
                self.logger.record(
                    "go2/deterministic_action_saturation",
                    float((deterministic_actions.abs() > 0.95).float().mean().cpu()),
                )


class FixedScenarioEvalCallback(EvalCallback):
    """EvalCallback with repeatable seeds and per-episode diagnostics."""

    def __init__(self, *args: Any, diagnostics_path: Path, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._diagnostics_path = diagnostics_path
        self._current_diagnostics: list[list[float]] = []
        self._diagnostics_history: list[list[list[float]]] = []

    def _log_success_callback(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super()._log_success_callback(locals_, globals_)
        if locals_["done"]:
            info = locals_["info"]
            self._current_diagnostics.append([
                float(info.get(key, np.nan))
                for key in EVAL_DIAGNOSTIC_KEYS
            ])

    def _on_step(self) -> bool:
        evaluation_due = self.eval_freq > 0 and self.n_calls % self.eval_freq == 0
        if evaluation_due:
            rewind_fixed_seed_cycle(self.eval_env)
            self._current_diagnostics = []

        continue_training = super()._on_step()

        if evaluation_due:
            self._diagnostics_history.append(self._current_diagnostics[:])
            self._write_diagnostics()
            diagnostics = np.asarray(self._current_diagnostics, dtype=float)
            for index, key in enumerate(EVAL_DIAGNOSTIC_KEYS):
                self.logger.record(
                    f"eval/{key}",
                    float(np.nanmean(diagnostics[:, index])),
                )
            self.logger.dump(self.num_timesteps)
        return continue_training

    def _write_diagnostics(self) -> None:
        if not self._diagnostics_path.is_file():
            return
        with np.load(self._diagnostics_path) as data:
            arrays = {key: data[key] for key in data.files}
        arrays.update({
            "diagnostic_names": np.asarray(EVAL_DIAGNOSTIC_KEYS),
            "diagnostics": np.asarray(self._diagnostics_history, dtype=float),
        })
        np.savez(self._diagnostics_path, **arrays)


class RollingCheckpointCallback(CheckpointCallback):
    """Keep only the newest checkpoints from the current run."""

    def __init__(self, *args: Any, keep: int = 5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._keep = keep

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            checkpoints = sorted(
                Path(self.save_path).glob(f"{self.name_prefix}_*_steps.zip"),
                key=lambda path: path.stat().st_mtime_ns,
            )
            for path in checkpoints[:-self._keep]:
                path.unlink()
        return continue_training


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
    holdout_log_path: Path


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
        holdout_log_path=output_dir / "eval_logs" / "holdout.npz",
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
        FixedScenarioEvalCallback(
            eval_env,
            best_model_save_path=str(paths.best_model_dir) if options.persist_models else None,
            log_path=str(paths.output_dir / "eval_logs"),
            eval_freq=callback_freq,
            n_eval_episodes=config.eval_episodes,
            deterministic=True,
            render=False,
            callback_after_eval=after_eval_callback,
            diagnostics_path=paths.eval_log_path,
        )
    ]
    callbacks.append(LocomotionMetricsCallback())
    if options.persist_checkpoints and options.persist_models:
        callbacks.append(
            RollingCheckpointCallback(
                save_freq=callback_freq,
                save_path=str(paths.checkpoint_dir),
                name_prefix=f"{config.algorithm.lower()}_go2",
                save_replay_buffer=False,
                save_vecnormalize=False,
                keep=5,
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
        env_kwargs=config.env_kwargs,
        monitor_dir=str(paths.monitor_dir / "train") if paths.monitor_dir else None,
        monitor_info_keys=EPISODE_INFO_KEYS,
    )) as train_env:
        with closing(make_vec_env(
            config.env_id,
            n_envs=1,
            seed=config.eval_seed,
            env_kwargs=config.eval_env_kwargs,
            monitor_dir=str(paths.monitor_dir / "eval") if paths.monitor_dir else None,
            monitor_info_keys=EPISODE_INFO_KEYS,
            fixed_reset_seeds=tuple(
                range(config.eval_seed, config.eval_seed + config.eval_episodes)
            ),
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

            final_eval_episodes = (
                options.final_eval_episodes or config.holdout_episodes
            )
            final_eval_seed = (
                options.final_eval_seed
                if options.final_eval_seed is not None
                else config.holdout_seed
            )
            eval_env.env_method(
                "set_reset_seeds",
                tuple(range(final_eval_seed, final_eval_seed + final_eval_episodes)),
            )
            holdout = evaluate_detailed(
                evaluation_model,
                eval_env,
                final_eval_episodes,
                EVAL_DIAGNOSTIC_KEYS,
                "holdout after training",
            )
            zero_baseline = evaluate_detailed(
                ZeroActionPolicy(config.action_dim),
                eval_env,
                final_eval_episodes,
                EVAL_DIAGNOSTIC_KEYS,
                "paired zero-action baseline",
            )
            paired_reward_delta = (
                holdout["episode_rewards"] - zero_baseline["episode_rewards"]
            )
            paired_ci = paired_bootstrap_interval(paired_reward_delta)
            paths.holdout_log_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                paths.holdout_log_path,
                rewards=holdout["episode_rewards"],
                zero_action_rewards=zero_baseline["episode_rewards"],
                paired_reward_delta=paired_reward_delta,
                diagnostic_names=holdout["diagnostic_names"],
                diagnostics=holdout["diagnostics"],
                zero_action_diagnostics=zero_baseline["diagnostics"],
            )
            after_eval = (
                holdout["mean_reward"],
                holdout["std_reward"],
            )

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
                holdout_metrics=holdout["metric_means"],
                zero_action_metrics=zero_baseline["metric_means"],
                paired_reward_delta={
                    "mean": float(np.mean(paired_reward_delta)),
                    "ci95_lower": paired_ci[0],
                    "ci95_upper": paired_ci[1],
                },
                holdout_log_path=paths.holdout_log_path,
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
    if config.n_envs > 1:
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
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
