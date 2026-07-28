"""Config-driven Optuna search and statistical revalidation for LunarLander."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from copy import deepcopy
import csv
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import pickle
import subprocess
import sys
import os
from typing import Any, Literal

from stable_baselines3.common.callbacks import BaseCallback

from sac_experiments.config import (
    DEFAULT_CONFIG,
    ExperimentConfig,
    deep_merge,
    load_config,
    load_yaml_file,
    reject_unknown_keys,
)
from sac_experiments.training import TrainingRunOptions, run_experiment
from sac_experiments.variants import Variant


SEARCH_SCHEMA_VERSION = 1
ROLLING_EVALUATION_WINDOW = 3
HELD_OUT_EVAL_SEED_OFFSET = 1_000_000


@dataclass(frozen=True)
class SearchParameter:
    """One scalar experiment-config value sampled by Optuna."""

    path: str
    kind: Literal["float", "int", "categorical"]
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: tuple[str | int | float | bool, ...] = ()

    def suggest(self, trial: Any) -> str | int | float | bool:
        if self.kind == "float":
            assert isinstance(self.low, (int, float))
            assert isinstance(self.high, (int, float))
            return trial.suggest_float(self.path, float(self.low), float(self.high), log=self.log)
        if self.kind == "int":
            assert isinstance(self.low, int)
            assert isinstance(self.high, int)
            return trial.suggest_int(self.path, self.low, self.high, log=self.log)
        return trial.suggest_categorical(self.path, list(self.choices))


@dataclass(frozen=True)
class RevalidationConfig:
    top_k: int
    seed_base: int
    min_seeds: int
    max_seeds: int
    evaluation_episodes: int


@dataclass(frozen=True)
class SearchConfig:
    config_path: Path
    base_config_path: Path
    study_name: str
    output_root: Path
    n_trials: int
    seed: int
    n_startup_trials: int
    min_resource: int
    reduction_factor: int
    parameters: tuple[SearchParameter, ...]
    revalidation: RevalidationConfig
    base_experiment: ExperimentConfig
    base_merged_config: dict[str, Any]

    @property
    def study_dir(self) -> Path:
        return self.output_root / self.study_name

    @property
    def database_path(self) -> Path:
        return self.study_dir / "study.sqlite3"

    @property
    def manifest_path(self) -> Path:
        return self.study_dir / "study_manifest.json"

    @property
    def sampler_state_path(self) -> Path:
        return self.study_dir / "sampler_state.pkl"

    @property
    def resource_rungs(self) -> tuple[int, ...]:
        rungs: list[int] = []
        resource = self.min_resource
        while resource < self.base_experiment.timesteps:
            rungs.append(resource)
            resource *= self.reduction_factor
        return tuple(rungs)


def _require_optuna() -> Any:
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Hyperparameter search requires Optuna. Install requirements-sac-demo.txt."
        ) from exc
    return optuna


def _require_scipy_t() -> Any:
    try:
        from scipy.stats import t as student_t
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Statistical revalidation requires scipy. Install requirements-sac-demo.txt."
        ) from exc
    return student_t


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    return value


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


def _safe_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if not all(character.isalnum() or character in "._-" for character in value):
        raise ValueError(f"{name} may contain only letters, numbers, dots, underscores, or hyphens.")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {value!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def _resolve_base_config_path(search_path: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("base_config must be a non-empty path string.")
    return (search_path.parent / value).resolve()


def _read_base_merged_config(path: Path) -> dict[str, Any]:
    override = load_yaml_file(path)
    reject_unknown_keys(override, DEFAULT_CONFIG)
    return deep_merge(DEFAULT_CONFIG, override)


def _path_parent_and_key(config: dict[str, Any], dotted_path: str) -> tuple[dict[str, Any], str]:
    parts = dotted_path.split(".")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError(
            f"Search parameter path must be a two-part config path, got {dotted_path!r}."
        )
    section, key = parts
    parent = config.get(section)
    if not isinstance(parent, dict) or key not in parent:
        raise ValueError(f"Unknown search parameter path: {dotted_path}.")
    return parent, key


def _validate_parameter_path(base_config: dict[str, Any], path: str) -> None:
    forbidden_sections = {"experiment", "environment", "training", "evaluation", "output"}
    section = path.split(".", maxsplit=1)[0]
    if section in forbidden_sections:
        raise ValueError(
            f"{path} changes an experiment invariant and is not allowed in a search space."
        )
    parent, key = _path_parent_and_key(base_config, path)
    if isinstance(parent[key], (dict, list)):
        raise ValueError(
            f"{path} is structural. Keep architecture and variants fixed within one study."
        )


def _parse_parameter(path: str, value: Any, base_config: dict[str, Any]) -> SearchParameter:
    _validate_parameter_path(base_config, path)
    data = _mapping(value, f"parameters.{path}")
    allowed = {"type", "low", "high", "log", "choices"}
    unknown = set(data) - allowed
    if unknown:
        raise ValueError(f"Unknown keys in parameters.{path}: {sorted(unknown)}")
    kind = data.get("type")
    if kind not in {"float", "int", "categorical"}:
        raise ValueError(f"parameters.{path}.type must be float, int, or categorical.")
    if kind == "categorical":
        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) < 2:
            raise ValueError(f"parameters.{path}.choices must contain at least two values.")
        if any(
            isinstance(choice, (dict, list)) or choice is None
            for choice in choices
        ):
            raise ValueError(f"parameters.{path}.choices must contain scalar non-null values.")
        return SearchParameter(path=path, kind=kind, choices=tuple(choices))

    low = data.get("low")
    high = data.get("high")
    if kind == "int":
        if isinstance(low, bool) or not isinstance(low, int):
            raise ValueError(f"parameters.{path}.low must be an integer.")
        if isinstance(high, bool) or not isinstance(high, int):
            raise ValueError(f"parameters.{path}.high must be an integer.")
    else:
        low = _finite_number(low, f"parameters.{path}.low")
        high = _finite_number(high, f"parameters.{path}.high")
    if low >= high:
        raise ValueError(f"parameters.{path}.low must be smaller than high.")
    log = data.get("log", False)
    if not isinstance(log, bool):
        raise ValueError(f"parameters.{path}.log must be a boolean.")
    if log and low <= 0:
        raise ValueError(f"parameters.{path}.low must be positive for log sampling.")
    return SearchParameter(path=path, kind=kind, low=low, high=high, log=log)


def _validate_resource_alignment(config: SearchConfig) -> None:
    experiment = config.base_experiment
    if experiment.timesteps % experiment.eval_freq != 0:
        raise ValueError(
            "Search base training.timesteps must be divisible by evaluation.frequency "
            "so the final objective has a deterministic evaluation."
        )
    if config.min_resource < ROLLING_EVALUATION_WINDOW * experiment.eval_freq:
        raise ValueError(
            "pruner.min_resource must allow at least three evaluations before pruning."
        )
    for resource in config.resource_rungs:
        if resource % experiment.eval_freq != 0:
            raise ValueError(
                f"Pruner rung {resource} must be divisible by evaluation.frequency "
                f"({experiment.eval_freq})."
            )
        if resource % experiment.n_envs != 0:
            raise ValueError(
                f"Pruner rung {resource} must be divisible by environment.n_envs "
                f"({experiment.n_envs})."
            )
        if experiment.algorithm == "PPO":
            rollout_size = experiment.n_envs * experiment.ppo["n_steps"]
            if resource % rollout_size != 0:
                raise ValueError(
                    f"PPO pruner rung {resource} must be divisible by rollout size "
                    f"({rollout_size})."
                )


def load_search_config(path: Path) -> SearchConfig:
    raw = load_yaml_file(path)
    allowed = {"base_config", "study", "pruner", "parameters", "revalidation"}
    unknown = set(raw) - allowed
    if unknown:
        raise ValueError(f"Unknown search config keys: {sorted(unknown)}")
    required = allowed - set(raw)
    if required:
        raise ValueError(f"Missing search config keys: {sorted(required)}")

    config_path = path.resolve()
    base_config_path = _resolve_base_config_path(config_path, raw["base_config"])
    base_experiment = load_config(base_config_path)
    if len(base_experiment.variants) != 1:
        raise ValueError("A hyperparameter study must use exactly one experiment variant.")
    base_merged_config = _read_base_merged_config(base_config_path)

    study = _mapping(raw["study"], "study")
    unknown = set(study) - {"name", "output_root", "n_trials", "seed", "n_startup_trials"}
    if unknown:
        raise ValueError(f"Unknown study keys: {sorted(unknown)}")
    study_name = _safe_name(study.get("name"), "study.name")
    output_root_value = study.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("study.output_root must be a non-empty path string.")
    n_trials = _positive_int(study.get("n_trials"), "study.n_trials")
    seed = _non_negative_int(study.get("seed"), "study.seed")
    n_startup_trials = _positive_int(study.get("n_startup_trials"), "study.n_startup_trials")
    if n_startup_trials > n_trials:
        raise ValueError("study.n_startup_trials must not exceed study.n_trials.")

    pruner = _mapping(raw["pruner"], "pruner")
    unknown = set(pruner) - {"min_resource", "reduction_factor"}
    if unknown:
        raise ValueError(f"Unknown pruner keys: {sorted(unknown)}")
    min_resource = _positive_int(pruner.get("min_resource"), "pruner.min_resource")
    reduction_factor = _positive_int(pruner.get("reduction_factor"), "pruner.reduction_factor")
    if reduction_factor < 2:
        raise ValueError("pruner.reduction_factor must be at least 2.")

    raw_parameters = _mapping(raw["parameters"], "parameters")
    if not raw_parameters:
        raise ValueError("parameters must not be empty.")
    parameters = tuple(
        _parse_parameter(str(parameter_path), value, base_merged_config)
        for parameter_path, value in raw_parameters.items()
    )
    if len({parameter.path for parameter in parameters}) != len(parameters):
        raise ValueError("parameters contains duplicate paths.")

    revalidation = _mapping(raw["revalidation"], "revalidation")
    unknown = set(revalidation) - {
        "top_k",
        "seed_base",
        "min_seeds",
        "max_seeds",
        "evaluation_episodes",
    }
    if unknown:
        raise ValueError(f"Unknown revalidation keys: {sorted(unknown)}")
    revalidation_config = RevalidationConfig(
        top_k=_positive_int(revalidation.get("top_k"), "revalidation.top_k"),
        seed_base=_non_negative_int(revalidation.get("seed_base"), "revalidation.seed_base"),
        min_seeds=_positive_int(revalidation.get("min_seeds"), "revalidation.min_seeds"),
        max_seeds=_positive_int(revalidation.get("max_seeds"), "revalidation.max_seeds"),
        evaluation_episodes=_positive_int(
            revalidation.get("evaluation_episodes"),
            "revalidation.evaluation_episodes",
        ),
    )
    if revalidation_config.min_seeds < 2:
        raise ValueError("revalidation.min_seeds must be at least 2 for a confidence bound.")
    if revalidation_config.max_seeds < revalidation_config.min_seeds:
        raise ValueError("revalidation.max_seeds must be at least min_seeds.")

    result = SearchConfig(
        config_path=config_path,
        base_config_path=base_config_path,
        study_name=study_name,
        output_root=Path(output_root_value),
        n_trials=n_trials,
        seed=seed,
        n_startup_trials=n_startup_trials,
        min_resource=min_resource,
        reduction_factor=reduction_factor,
        parameters=parameters,
        revalidation=revalidation_config,
        base_experiment=base_experiment,
        base_merged_config=base_merged_config,
    )
    _validate_resource_alignment(result)
    return result


def _set_config_value(config: dict[str, Any], dotted_path: str, value: Any) -> None:
    parent, key = _path_parent_and_key(config, dotted_path)
    parent[key] = value


def _write_yaml(path: Path, data: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")


def _render_config(
    search_config: SearchConfig,
    parameters: Mapping[str, Any],
    training_seed: int,
    trial_root: Path,
) -> Path:
    rendered = deepcopy(search_config.base_merged_config)
    for parameter in search_config.parameters:
        if parameter.path not in parameters:
            raise ValueError(f"Missing sampled parameter: {parameter.path}.")
        _set_config_value(rendered, parameter.path, parameters[parameter.path])
    _set_config_value(rendered, "training.seed", training_seed)
    _set_config_value(rendered, "output.directory", str(trial_root / "outputs"))
    _set_config_value(rendered, "output.tensorboard_log", str(trial_root / "runs"))
    _set_config_value(rendered, "output.run_tag", None)
    config_path = trial_root / "resolved_config.yaml"
    _write_yaml(config_path, rendered)
    return config_path


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
        self.evaluation_means: deque[float] = deque(maxlen=ROLLING_EVALUATION_WINDOW)
        self.reported_values: dict[int, float] = {}

    @property
    def objective_value(self) -> float | None:
        if len(self.evaluation_means) < ROLLING_EVALUATION_WINDOW:
            return None
        return float(sum(self.evaluation_means) / len(self.evaluation_means))

    def _on_step(self) -> bool:
        parent = self.parent
        if parent is None or not hasattr(parent, "last_mean_reward"):
            raise RuntimeError("TrialEvaluationReporter must be attached to EvalCallback.")
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
            optuna = _require_optuna()
            raise optuna.TrialPruned(
                f"Pruned at {self.num_timesteps} transitions with rolling reward {objective:.3f}."
            )
        return True


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _manifest(search_config: SearchConfig, optuna: Any) -> dict[str, Any]:
    return {
        "schema_version": SEARCH_SCHEMA_VERSION,
        "study_name": search_config.study_name,
        "base_config": str(search_config.base_config_path),
        "base_config_sha256": _sha256_file(search_config.base_config_path),
        "search_config": str(search_config.config_path),
        "search_config_sha256": _sha256_file(search_config.config_path),
        "git_head": _git_head(),
        "optuna_version": optuna.__version__,
        "python_version": sys.version,
        "sampler": {
            "name": "TPESampler",
            "seed": search_config.seed,
            "n_startup_trials": search_config.n_startup_trials,
        },
        "pruner": {
            "name": "SuccessiveHalvingPruner",
            "min_resource": search_config.min_resource,
            "reduction_factor": search_config.reduction_factor,
            "rungs": list(search_config.resource_rungs),
        },
        "objective": {
            "direction": "maximize",
            "metric": "rolling_mean_of_last_3_deterministic_evaluations",
        },
    }


def _write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _hardware_snapshot(experiment: ExperimentConfig) -> dict[str, Any]:
    """Record target-machine facts without changing the configured worker count."""

    import torch

    cpu_count = os.cpu_count()
    snapshot: dict[str, Any] = {
        "cpu_logical_cores": cpu_count,
        "requested_device": experiment.device,
        "cuda_available": torch.cuda.is_available(),
        "configured_n_envs": experiment.n_envs,
    }
    if cpu_count is not None and experiment.n_envs > max(2, cpu_count - 2):
        snapshot["worker_warning"] = (
            "Configured environment workers leave fewer than two logical CPU cores for "
            "the learner and operating system. Consider a smaller n_envs preset."
        )
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        snapshot |= {
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": torch.cuda.get_device_name(0),
            "cuda_total_memory_bytes": properties.total_memory,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
        }
        if torch.cuda.device_count() != 1:
            snapshot["gpu_warning"] = "This study is designed for one serial GPU worker."
    return snapshot


def _cuda_memory_snapshot() -> dict[str, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _atomic_pickle(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(pickle.dumps(value))
    temporary_path.replace(path)


def _load_sampler(path: Path, search_config: SearchConfig, optuna: Any) -> Any:
    if path.exists():
        return pickle.loads(path.read_bytes())
    return optuna.samplers.TPESampler(
        seed=search_config.seed,
        n_startup_trials=search_config.n_startup_trials,
    )


def _open_study(search_config: SearchConfig, resume: bool) -> tuple[Any, Any]:
    optuna = _require_optuna()
    manifest = _manifest(search_config, optuna)
    study_dir = search_config.study_dir
    if study_dir.exists() and not resume:
        if any(study_dir.iterdir()):
            raise FileExistsError(
                f"Study directory already exists: {study_dir}. Use --resume or a new study.name."
            )
    if resume:
        if not search_config.manifest_path.is_file():
            raise FileNotFoundError(f"Cannot resume without manifest: {search_config.manifest_path}")
        saved = json.loads(search_config.manifest_path.read_text(encoding="utf-8"))
        for key in ("schema_version", "base_config_sha256", "search_config_sha256", "study_name"):
            if saved.get(key) != manifest.get(key):
                raise ValueError(f"Cannot resume: manifest field {key!r} changed.")
    else:
        study_dir.mkdir(parents=True, exist_ok=True)
        _write_json(search_config.manifest_path, manifest)

    sampler = _load_sampler(search_config.sampler_state_path, search_config, optuna)
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=search_config.min_resource,
        reduction_factor=search_config.reduction_factor,
        min_early_stopping_rate=0,
    )
    storage = f"sqlite:///{search_config.database_path.resolve()}"
    study = optuna.create_study(
        study_name=search_config.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=resume,
    )
    return study, optuna


def _trial_result_path(search_config: SearchConfig, trial_number: int) -> Path:
    return search_config.study_dir / "trials" / f"trial-{trial_number:04d}" / "trial_result.json"


def _write_study_summary(search_config: SearchConfig, study: Any) -> Path:
    records: list[dict[str, Any]] = []
    for trial in study.trials:
        records.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "value": trial.value,
                "params": trial.params,
                "user_attrs": trial.user_attrs,
            }
        )
    summary = {
        "study_name": search_config.study_name,
        "direction": "maximize",
        "target_trials": search_config.n_trials,
        "completed_trials": sum(record["state"] == "COMPLETE" for record in records),
        "pruned_trials": sum(record["state"] == "PRUNED" for record in records),
        "failed_trials": sum(record["state"] == "FAIL" for record in records),
        "trials": records,
    }
    summary_path = search_config.study_dir / "study_summary.json"
    _write_json(summary_path, summary)
    csv_path = search_config.study_dir / "study_trials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["number", "state", "value", "params"])
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "number": record["number"],
                    "state": record["state"],
                    "value": record["value"],
                    "params": json.dumps(record["params"], sort_keys=True),
                }
            )
    return summary_path


def run_search(search_config: SearchConfig, resume: bool = False) -> Path:
    """Run enough serial trials to reach ``study.n_trials`` and write a summary."""

    study, optuna = _open_study(search_config, resume)
    _write_json(
        search_config.study_dir / "hardware_snapshot.json",
        _hardware_snapshot(search_config.base_experiment),
    )
    existing_trials = len(study.trials)
    remaining_trials = max(0, search_config.n_trials - existing_trials)

    def after_trial(current_study: Any, _: Any) -> None:
        _atomic_pickle(search_config.sampler_state_path, current_study.sampler)

    def objective(trial: Any) -> float:
        parameters = {parameter.path: parameter.suggest(trial) for parameter in search_config.parameters}
        trial_root = search_config.study_dir / "trials" / f"trial-{trial.number:04d}"
        resolved_path = _render_config(search_config, parameters, search_config.seed, trial_root)
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
                raise RuntimeError("Trial completed without three finite evaluations.")
            trial.set_user_attr("resolved_config", str(resolved_path))
            trial.set_user_attr("experiment_summary", str(summary_path))
            trial.set_user_attr("reported_values", reporter.reported_values)
            trial.set_user_attr("cuda_memory", _cuda_memory_snapshot())
            _write_json(
                _trial_result_path(search_config, trial.number),
                {
                    "state": "COMPLETE",
                    "number": trial.number,
                    "value": value,
                    "params": parameters,
                    "reported_values": reporter.reported_values,
                    "experiment_summary": str(summary_path),
                    "cuda_memory": _cuda_memory_snapshot(),
                },
            )
            return value
        except optuna.TrialPruned as exc:
            reporter = reporter_holder.get("reporter")
            _write_json(
                _trial_result_path(search_config, trial.number),
                {
                    "state": "PRUNED",
                    "number": trial.number,
                    "params": parameters,
                    "reported_values": reporter.reported_values if reporter else {},
                    "reason": str(exc),
                },
            )
            raise
        except Exception as exc:
            _write_json(
                _trial_result_path(search_config, trial.number),
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
        study.optimize(objective, n_trials=remaining_trials, callbacks=[after_trial], catch=(Exception,))
    _atomic_pickle(search_config.sampler_state_path, study.sampler)
    return _write_study_summary(search_config, study)


def _confidence_bounds(scores: Sequence[float]) -> tuple[float, float, float, float]:
    if len(scores) < 2:
        raise ValueError("At least two scores are required for a confidence bound.")
    student_t = _require_scipy_t()
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    standard_error = math.sqrt(variance / len(scores))
    critical = float(student_t.ppf(0.95, df=len(scores) - 1))
    return mean, math.sqrt(variance), mean - critical * standard_error, mean + critical * standard_error


def _paired_lower_bound(leader: Sequence[float], challenger: Sequence[float]) -> float:
    if len(leader) != len(challenger):
        raise ValueError("Paired comparisons require the same seed count.")
    differences = [left - right for left, right in zip(leader, challenger, strict=True)]
    return _confidence_bounds(differences)[2]


def _completed_candidates(study: Any, top_k: int) -> list[Any]:
    optuna = _require_optuna()
    completed = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
        and trial.value is not None
        and math.isfinite(float(trial.value))
    ]
    completed.sort(key=lambda trial: float(trial.value), reverse=True)
    if len(completed) < top_k:
        raise RuntimeError(
            f"Revalidation requires {top_k} completed full-budget trials, found {len(completed)}."
        )
    return completed[:top_k]


def _run_revalidation_seed(
    search_config: SearchConfig,
    candidate: Any,
    seed: int,
) -> float:
    candidate_root = (
        search_config.study_dir
        / "revalidation"
        / f"candidate-{candidate.number:04d}"
        / f"seed-{seed}"
    )
    resolved_path = _render_config(search_config, candidate.params, seed, candidate_root)
    experiment = load_config(resolved_path)
    summary_path = run_experiment(
        experiment,
        TrainingRunOptions(
            persist_models=False,
            persist_checkpoints=False,
            persist_tensorboard=False,
            persist_monitor=False,
            final_eval_episodes=search_config.revalidation.evaluation_episodes,
            final_eval_seed=seed + HELD_OUT_EVAL_SEED_OFFSET,
        ),
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    score = float(summary["variants"][0]["after_training"]["mean_reward"])
    if not math.isfinite(score):
        raise FloatingPointError("Held-out revalidation score is not finite.")
    _write_json(
        candidate_root / "revalidation_result.json",
        {
            "candidate_trial": candidate.number,
            "training_seed": seed,
            "held_out_eval_seed": seed + HELD_OUT_EVAL_SEED_OFFSET,
            "evaluation_episodes": search_config.revalidation.evaluation_episodes,
            "score": score,
            "experiment_summary": str(summary_path),
        },
    )
    return score


def _write_champion_config(search_config: SearchConfig, candidate: Any) -> Path:
    champion_root = search_config.study_dir / "champion_training"
    champion_path = _render_config(
        search_config,
        candidate.params,
        search_config.seed,
        champion_root,
    )
    destination = search_config.study_dir / "champion.yaml"
    destination.write_text(champion_path.read_text(encoding="utf-8"), encoding="utf-8")
    return destination


def run_revalidation(search_config: SearchConfig) -> Path:
    """Statistically choose a champion from the highest-ranked completed trials."""

    summary_path = search_config.study_dir / "revalidation_summary.json"
    if summary_path.is_file():
        return summary_path
    revalidation_root = search_config.study_dir / "revalidation"
    if revalidation_root.exists() and any(revalidation_root.iterdir()):
        raise FileExistsError(
            "A partial revalidation already exists. Use a new study.name rather than "
            "overwriting its seed results."
        )
    study, _ = _open_study(search_config, resume=True)
    candidates = _completed_candidates(study, search_config.revalidation.top_k)
    scores: dict[int, list[float]] = {candidate.number: [] for candidate in candidates}
    active = list(candidates)
    separated = False

    for seed_index in range(search_config.revalidation.min_seeds):
        seed = search_config.revalidation.seed_base + seed_index
        for candidate in candidates:
            scores[candidate.number].append(_run_revalidation_seed(search_config, candidate, seed))

    current_seed_count = search_config.revalidation.min_seeds
    while len(active) > 1 and current_seed_count < search_config.revalidation.max_seeds:
        leader = max(active, key=lambda candidate: _confidence_bounds(scores[candidate.number])[2])
        unresolved = [
            candidate
            for candidate in active
            if candidate.number != leader.number
            and _paired_lower_bound(scores[leader.number], scores[candidate.number]) <= 0
        ]
        if not unresolved:
            active = [leader]
            separated = True
            break
        active = [leader, *unresolved]
        seed = search_config.revalidation.seed_base + current_seed_count
        for candidate in active:
            scores[candidate.number].append(_run_revalidation_seed(search_config, candidate, seed))
        current_seed_count += 1

    champion = max(active, key=lambda candidate: _confidence_bounds(scores[candidate.number])[2])
    candidate_stats = {
        str(candidate.number): {
            "params": candidate.params,
            "search_value": candidate.value,
            "scores": scores[candidate.number],
            "mean": _confidence_bounds(scores[candidate.number])[0],
            "std": _confidence_bounds(scores[candidate.number])[1],
            "lower_confidence_bound_95": _confidence_bounds(scores[candidate.number])[2],
            "upper_confidence_bound_95": _confidence_bounds(scores[candidate.number])[3],
            "active_at_finish": candidate in active,
        }
        for candidate in candidates
    }
    champion_path = _write_champion_config(search_config, champion)
    _write_json(
        summary_path,
        {
            "selection_rule": "maximum one-sided 95% Student-t lower confidence bound",
            "top_k": search_config.revalidation.top_k,
            "minimum_seeds": search_config.revalidation.min_seeds,
            "maximum_seeds": search_config.revalidation.max_seeds,
            "seeds_used_for_champion": len(scores[champion.number]),
            "statistically_separated": separated,
            "champion_trial": champion.number,
            "champion_config": str(champion_path),
            "candidates": candidate_stats,
        },
    )
    return summary_path
