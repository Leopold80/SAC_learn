"""Optuna study lifecycle, ASHA reporting, and trial execution."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import json
import math
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from sac_experiments.config import ExperimentConfig, load_config
from sac_experiments.search.artifacts import (
    build_manifest,
    cuda_memory_snapshot,
    hardware_snapshot,
    load_sampler,
    render_config,
    save_sampler,
    trial_result_path,
    write_json,
    write_study_summary,
)
from sac_experiments.search.config import (
    ROLLING_EVALUATION_WINDOW,
    SearchConfig,
)
from sac_experiments.training import TrainingRunOptions, run_experiment
from sac_experiments.variants import Variant


def require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hyperparameter search requires Optuna. "
            "Install requirements-sac-demo.txt."
        ) from exc
    return optuna


class TrialEvaluationReporter(BaseCallback):
    """Report rolling evaluation quality at configured ASHA rungs."""

    def __init__(
        self,
        trial: Any,
        rungs: Sequence[int],
        max_resource: int,
    ) -> None:
        super().__init__()
        self.trial = trial
        self.rungs = frozenset(rungs)
        self.max_resource = max_resource
        self.evaluation_means: deque[float] = deque(
            maxlen=ROLLING_EVALUATION_WINDOW
        )
        self.reported_values: dict[int, float] = {}

    @property
    def objective_value(self) -> float | None:
        if len(self.evaluation_means) < ROLLING_EVALUATION_WINDOW:
            return None
        return float(sum(self.evaluation_means) / len(self.evaluation_means))

    def _on_step(self) -> bool:
        parent = self.parent
        if parent is None or not hasattr(parent, "last_mean_reward"):
            raise RuntimeError(
                "TrialEvaluationReporter must be attached to EvalCallback."
            )
        mean_reward = float(parent.last_mean_reward)
        if not math.isfinite(mean_reward):
            raise FloatingPointError("Evaluation reward is not finite.")
        self.evaluation_means.append(mean_reward)
        objective = self.objective_value
        if objective is None or self.num_timesteps not in self.rungs:
            return True
        self.trial.report(objective, self.num_timesteps)
        self.reported_values[self.num_timesteps] = objective
        if self.num_timesteps < self.max_resource and self.trial.should_prune():
            raise require_optuna().TrialPruned(
                f"Pruned at {self.num_timesteps} transitions with rolling reward "
                f"{objective:.3f}."
            )
        return True


def open_study(
    search_config: SearchConfig,
    resume: bool,
) -> tuple[Any, Any]:
    """Open a new or compatible existing Optuna study."""

    optuna = require_optuna()
    manifest = build_manifest(search_config, optuna)
    study_dir = search_config.study_dir
    if study_dir.exists() and not resume and any(study_dir.iterdir()):
        raise FileExistsError(
            f"Study directory already exists: {study_dir}. "
            "Use --resume or a new study.name."
        )
    if resume:
        if not search_config.manifest_path.is_file():
            raise FileNotFoundError(
                f"Cannot resume without manifest: {search_config.manifest_path}"
            )
        saved = json.loads(
            search_config.manifest_path.read_text(encoding="utf-8")
        )
        for key in (
            "schema_version",
            "base_config_sha256",
            "search_config_sha256",
            "study_name",
        ):
            if saved.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest field {key!r} changed.")
    else:
        study_dir.mkdir(parents=True, exist_ok=True)
        write_json(search_config.manifest_path, manifest)

    sampler = load_sampler(
        search_config.sampler_state_path,
        search_config,
        optuna,
    )
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=search_config.min_resource,
        reduction_factor=search_config.reduction_factor,
        min_early_stopping_rate=0,
    )
    study = optuna.create_study(
        study_name=search_config.study_name,
        storage=f"sqlite:///{search_config.database_path.resolve()}",
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=resume,
    )
    return study, optuna


def run_search(search_config: SearchConfig, resume: bool = False) -> Path:
    """Run enough serial trials to reach ``study.n_trials``."""

    study, optuna = open_study(search_config, resume)
    write_json(
        search_config.study_dir / "hardware_snapshot.json",
        hardware_snapshot(search_config.base_experiment),
    )
    remaining_trials = max(0, search_config.n_trials - len(study.trials))

    def after_trial(current_study: Any, _: Any) -> None:
        save_sampler(search_config.sampler_state_path, current_study.sampler)

    def objective(trial: Any) -> float:
        parameters = {
            parameter.path: parameter.suggest(trial)
            for parameter in search_config.parameters
        }
        trial_root = (
            search_config.study_dir / "trials" / f"trial-{trial.number:04d}"
        )
        resolved_path = render_config(
            search_config,
            parameters,
            search_config.seed,
            trial_root,
        )
        trial_config = load_config(resolved_path)
        reporter_holder: dict[str, TrialEvaluationReporter] = {}

        def callback_factory(_: ExperimentConfig, __: Variant) -> BaseCallback:
            reporter = TrialEvaluationReporter(
                trial,
                search_config.resource_rungs,
                trial_config.timesteps,
            )
            reporter_holder["reporter"] = reporter
            return reporter

        try:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except ModuleNotFoundError:
                pass
            summary_path = run_experiment(
                trial_config,
                TrainingRunOptions(
                    evaluation_callback_factory=callback_factory,
                    persist_models=False,
                    persist_checkpoints=False,
                    persist_tensorboard=False,
                    persist_monitor=False,
                ),
            )
            reporter = reporter_holder["reporter"]
            value = reporter.objective_value
            if value is None or not math.isfinite(value):
                raise RuntimeError(
                    "Trial completed without three finite evaluations."
                )
            memory = cuda_memory_snapshot()
            trial.set_user_attr("resolved_config", str(resolved_path))
            trial.set_user_attr("experiment_summary", str(summary_path))
            trial.set_user_attr("reported_values", reporter.reported_values)
            trial.set_user_attr("cuda_memory", memory)
            write_json(
                trial_result_path(search_config, trial.number),
                {
                    "state": "COMPLETE",
                    "number": trial.number,
                    "value": value,
                    "params": parameters,
                    "reported_values": reporter.reported_values,
                    "experiment_summary": str(summary_path),
                    "cuda_memory": memory,
                },
            )
            return value
        except optuna.TrialPruned as exc:
            reporter = reporter_holder.get("reporter")
            write_json(
                trial_result_path(search_config, trial.number),
                {
                    "state": "PRUNED",
                    "number": trial.number,
                    "params": parameters,
                    "reported_values": (
                        reporter.reported_values if reporter else {}
                    ),
                    "reason": str(exc),
                },
            )
            raise
        except Exception as exc:
            write_json(
                trial_result_path(search_config, trial.number),
                {
                    "state": "FAIL",
                    "number": trial.number,
                    "params": parameters,
                    "error_type": type(exc).__name__,
                    "reason": str(exc),
                },
            )
            raise

    if remaining_trials:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            callbacks=[after_trial],
            catch=(Exception,),
        )
    save_sampler(search_config.sampler_state_path, study.sampler)
    return write_study_summary(search_config, study)
