"""Filesystem artifacts and reproducibility metadata for search workflows."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
from typing import Any

from sac_experiments.config import ExperimentConfig
from sac_experiments.search.config import SearchConfig, path_parent_and_key


SEARCH_SCHEMA_VERSION = 1


def render_config(
    search_config: SearchConfig,
    parameters: Mapping[str, Any],
    training_seed: int,
    trial_root: Path,
) -> Path:
    """Render a normal experiment YAML for one trial or revalidation seed."""

    rendered = deepcopy(search_config.base_merged_config)
    for parameter in search_config.parameters:
        if parameter.path not in parameters:
            raise ValueError(f"Missing sampled parameter: {parameter.path}.")
        parent, key = path_parent_and_key(rendered, parameter.path)
        parent[key] = parameters[parameter.path]
    for dotted_path, value in {
        "training.seed": training_seed,
        "output.directory": str(trial_root / "outputs"),
        "output.tensorboard_log": str(trial_root / "runs"),
        "output.run_tag": None,
    }.items():
        parent, key = path_parent_and_key(rendered, dotted_path)
        parent[key] = value

    import yaml

    config_path = trial_root / "resolved_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(rendered, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def hardware_snapshot(experiment: ExperimentConfig) -> dict[str, Any]:
    """Record target-machine facts without changing configured worker counts."""

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
            snapshot["gpu_warning"] = (
                "This study is designed for one serial GPU worker."
            )
    return snapshot


def cuda_memory_snapshot() -> dict[str, int] | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return {
        "max_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_manifest(search_config: SearchConfig, optuna: Any) -> dict[str, Any]:
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


def save_sampler(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(pickle.dumps(value))
    temporary_path.replace(path)


def load_sampler(path: Path, search_config: SearchConfig, optuna: Any) -> Any:
    if path.exists():
        return pickle.loads(path.read_bytes())
    return optuna.samplers.TPESampler(
        seed=search_config.seed,
        n_startup_trials=search_config.n_startup_trials,
    )


def trial_result_path(search_config: SearchConfig, trial_number: int) -> Path:
    return (
        search_config.study_dir
        / "trials"
        / f"trial-{trial_number:04d}"
        / "trial_result.json"
    )


def write_study_summary(search_config: SearchConfig, study: Any) -> Path:
    records = [
        {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
        }
        for trial in study.trials
    ]
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
    write_json(summary_path, summary)

    csv_path = search_config.study_dir / "study_trials.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["number", "state", "value", "params"],
        )
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


def write_champion_config(search_config: SearchConfig, candidate: Any) -> Path:
    champion_path = render_config(
        search_config,
        candidate.params,
        search_config.seed,
        search_config.study_dir / "champion_training",
    )
    destination = search_config.study_dir / "champion.yaml"
    destination.write_text(champion_path.read_text(encoding="utf-8"), encoding="utf-8")
    return destination
