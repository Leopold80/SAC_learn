"""Multi-seed statistical revalidation and champion selection."""

from __future__ import annotations

from collections.abc import Sequence
import json
import math
from pathlib import Path
from typing import Any

from sac_experiments.config import load_config
from sac_experiments.search.artifacts import (
    render_config,
    write_champion_config,
    write_json,
)
from sac_experiments.search.config import (
    HELD_OUT_EVAL_SEED_OFFSET,
    SearchConfig,
)
from sac_experiments.search.runner import open_study, require_optuna
from sac_experiments.training import TrainingRunOptions, run_experiment


def _require_scipy_t() -> Any:
    try:
        from scipy.stats import t as student_t
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Statistical revalidation requires scipy. "
            "Install requirements-sac-demo.txt."
        ) from exc
    return student_t


def confidence_bounds(
    scores: Sequence[float],
) -> tuple[float, float, float, float]:
    if len(scores) < 2:
        raise ValueError("At least two scores are required for a confidence bound.")
    student_t = _require_scipy_t()
    mean = sum(scores) / len(scores)
    variance = sum((score - mean) ** 2 for score in scores) / (len(scores) - 1)
    standard_error = math.sqrt(variance / len(scores))
    critical = float(student_t.ppf(0.95, df=len(scores) - 1))
    return (
        mean,
        math.sqrt(variance),
        mean - critical * standard_error,
        mean + critical * standard_error,
    )


def paired_lower_bound(
    leader: Sequence[float],
    challenger: Sequence[float],
) -> float:
    if len(leader) != len(challenger):
        raise ValueError("Paired comparisons require the same seed count.")
    differences = [
        left - right
        for left, right in zip(leader, challenger, strict=True)
    ]
    return confidence_bounds(differences)[2]


def _completed_candidates(study: Any, top_k: int) -> list[Any]:
    optuna = require_optuna()
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
            f"Revalidation requires {top_k} completed full-budget trials, "
            f"found {len(completed)}."
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
    resolved_path = render_config(
        search_config,
        candidate.params,
        seed,
        candidate_root,
    )
    summary_path = run_experiment(
        load_config(resolved_path),
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
    write_json(
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


def run_revalidation(search_config: SearchConfig) -> Path:
    """Statistically choose a champion from completed full-budget trials."""

    summary_path = search_config.study_dir / "revalidation_summary.json"
    if summary_path.is_file():
        return summary_path
    revalidation_root = search_config.study_dir / "revalidation"
    if revalidation_root.exists() and any(revalidation_root.iterdir()):
        raise FileExistsError(
            "A partial revalidation already exists. Use a new study.name rather "
            "than overwriting its seed results."
        )

    study, _ = open_study(search_config, resume=True)
    candidates = _completed_candidates(study, search_config.revalidation.top_k)
    scores: dict[int, list[float]] = {
        candidate.number: [] for candidate in candidates
    }
    active = list(candidates)
    separated = False

    for seed_index in range(search_config.revalidation.min_seeds):
        seed = search_config.revalidation.seed_base + seed_index
        for candidate in candidates:
            scores[candidate.number].append(
                _run_revalidation_seed(search_config, candidate, seed)
            )

    seed_count = search_config.revalidation.min_seeds
    while len(active) > 1 and seed_count < search_config.revalidation.max_seeds:
        leader = max(
            active,
            key=lambda candidate: confidence_bounds(
                scores[candidate.number]
            )[2],
        )
        unresolved = [
            candidate
            for candidate in active
            if candidate.number != leader.number
            and paired_lower_bound(
                scores[leader.number],
                scores[candidate.number],
            )
            <= 0
        ]
        if not unresolved:
            active = [leader]
            separated = True
            break
        active = [leader, *unresolved]
        seed = search_config.revalidation.seed_base + seed_count
        for candidate in active:
            scores[candidate.number].append(
                _run_revalidation_seed(search_config, candidate, seed)
            )
        seed_count += 1

    champion = max(
        active,
        key=lambda candidate: confidence_bounds(scores[candidate.number])[2],
    )
    candidate_stats = {
        str(candidate.number): {
            "params": candidate.params,
            "search_value": candidate.value,
            "scores": scores[candidate.number],
            "mean": confidence_bounds(scores[candidate.number])[0],
            "std": confidence_bounds(scores[candidate.number])[1],
            "lower_confidence_bound_95": confidence_bounds(
                scores[candidate.number]
            )[2],
            "upper_confidence_bound_95": confidence_bounds(
                scores[candidate.number]
            )[3],
            "active_at_finish": candidate in active,
        }
        for candidate in candidates
    }
    champion_path = write_champion_config(search_config, champion)
    write_json(
        summary_path,
        {
            "selection_rule": (
                "maximum one-sided 95% Student-t lower confidence bound"
            ),
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
