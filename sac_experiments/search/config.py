"""Validated configuration contract for Optuna search and revalidation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal

from sac_experiments.config import (
    DEFAULT_CONFIG,
    ExperimentConfig,
    deep_merge,
    load_config,
    load_yaml_file,
    reject_unknown_keys,
)


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
            return trial.suggest_float(
                self.path,
                float(self.low),
                float(self.high),
                log=self.log,
            )
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
        raise ValueError(
            f"{name} may contain only letters, numbers, dots, underscores, or hyphens."
        )
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


def path_parent_and_key(
    config: dict[str, Any],
    dotted_path: str,
) -> tuple[dict[str, Any], str]:
    """Resolve a two-part YAML path such as ``sac.learning_rate``."""

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
    parent, key = path_parent_and_key(base_config, path)
    if isinstance(parent[key], (dict, list)):
        raise ValueError(
            f"{path} is structural. Keep architecture and variants fixed within one study."
        )


def _parse_parameter(
    path: str,
    value: Any,
    base_config: dict[str, Any],
) -> SearchParameter:
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
            raise ValueError(
                f"parameters.{path}.choices must contain at least two values."
            )
        if any(
            isinstance(choice, (dict, list)) or choice is None
            for choice in choices
        ):
            raise ValueError(
                f"parameters.{path}.choices must contain scalar non-null values."
            )
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
    """Load and validate one search YAML plus its referenced experiment YAML."""

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
    unknown = set(study) - {
        "name",
        "output_root",
        "n_trials",
        "seed",
        "n_startup_trials",
    }
    if unknown:
        raise ValueError(f"Unknown study keys: {sorted(unknown)}")
    study_name = _safe_name(study.get("name"), "study.name")
    output_root_value = study.get("output_root")
    if not isinstance(output_root_value, str) or not output_root_value:
        raise ValueError("study.output_root must be a non-empty path string.")
    n_trials = _positive_int(study.get("n_trials"), "study.n_trials")
    seed = _non_negative_int(study.get("seed"), "study.seed")
    n_startup_trials = _positive_int(
        study.get("n_startup_trials"),
        "study.n_startup_trials",
    )
    if n_startup_trials > n_trials:
        raise ValueError("study.n_startup_trials must not exceed study.n_trials.")

    pruner = _mapping(raw["pruner"], "pruner")
    unknown = set(pruner) - {"min_resource", "reduction_factor"}
    if unknown:
        raise ValueError(f"Unknown pruner keys: {sorted(unknown)}")
    min_resource = _positive_int(pruner.get("min_resource"), "pruner.min_resource")
    reduction_factor = _positive_int(
        pruner.get("reduction_factor"),
        "pruner.reduction_factor",
    )
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
        seed_base=_non_negative_int(
            revalidation.get("seed_base"),
            "revalidation.seed_base",
        ),
        min_seeds=_positive_int(
            revalidation.get("min_seeds"),
            "revalidation.min_seeds",
        ),
        max_seeds=_positive_int(
            revalidation.get("max_seeds"),
            "revalidation.max_seeds",
        ),
        evaluation_episodes=_positive_int(
            revalidation.get("evaluation_episodes"),
            "revalidation.evaluation_episodes",
        ),
    )
    if revalidation_config.min_seeds < 2:
        raise ValueError(
            "revalidation.min_seeds must be at least 2 for a confidence bound."
        )
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
