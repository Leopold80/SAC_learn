#!/usr/bin/env python3
"""Summarize fixed-command holdout metrics across Go2 training seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_root",
        type=Path,
        help="Directory containing seed_<N> output directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Summary JSON path (default: EXPERIMENT_ROOT/multiseed_summary.json).",
    )
    parser.add_argument(
        "--tensorboard-root",
        type=Path,
        default=None,
        help="Directory containing seed_<N> TensorBoard runs.",
    )
    return parser.parse_args()


def _mean_metric(
    diagnostics: np.ndarray,
    names: list[str],
    name: str,
    *,
    absolute: bool = False,
) -> float:
    values = diagnostics[:, names.index(name)]
    if absolute:
        values = np.abs(values)
    return float(np.nanmean(values))


def _tail_scalar(event_path: Path, tag: str) -> np.ndarray:
    accumulator = EventAccumulator(
        str(event_path),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    if tag not in accumulator.Tags()["scalars"]:
        return np.asarray([], dtype=float)
    values = np.asarray(
        [event.value for event in accumulator.Scalars(tag)],
        dtype=float,
    )
    return values[max(0, 3 * len(values) // 4):]


def main() -> int:
    args = _parse_args()
    root = args.experiment_root.resolve()
    output = args.output or root / "multiseed_summary.json"
    if args.tensorboard_root is not None:
        tensorboard_root = args.tensorboard_root.resolve()
    else:
        repo_root = Path(__file__).resolve().parents[1]
        try:
            relative = root.relative_to(repo_root / "outputs")
            tensorboard_root = repo_root / "runs" / relative
        except ValueError:
            tensorboard_root = root
    seed_reports: list[dict] = []

    for seed_dir in sorted(root.glob("seed_*")):
        holdout_path = seed_dir / "eval_logs" / "holdout.npz"
        if not holdout_path.is_file():
            print(f"missing holdout: {holdout_path}", file=sys.stderr)
            continue
        with np.load(holdout_path) as data:
            names = data["diagnostic_names"].tolist()
            diagnostics = data["diagnostics"].astype(float)
            rewards = data["rewards"].astype(float)
            zero_rewards = data["zero_action_rewards"].astype(float)
            deltas = rewards - zero_rewards

        termination_codes = diagnostics[:, names.index("termination_code")]
        report = {
            "seed": int(seed_dir.name.removeprefix("seed_")),
            "episodes": int(diagnostics.shape[0]),
            "falls": int(np.count_nonzero(termination_codes)),
            "mean_reward": float(np.mean(rewards)),
            "vx_rmse": _mean_metric(
                diagnostics, names, "episode_vx_rmse"
            ),
            "mean_vx": _mean_metric(
                diagnostics, names, "episode_mean_vx"
            ),
            "mean_abs_vy": _mean_metric(
                diagnostics,
                names,
                "episode_mean_vy",
                absolute=True,
            ),
            "mean_abs_yaw_rate": _mean_metric(
                diagnostics,
                names,
                "episode_mean_yaw_rate",
                absolute=True,
            ),
            "action_saturation": _mean_metric(
                diagnostics,
                names,
                "episode_action_saturation",
            ),
            "paired_reward_delta_mean": float(np.mean(deltas)),
        }
        report["passes_behavior_gate"] = bool(
            report["falls"] <= 1
            and report["vx_rmse"] <= 0.20
            and 0.35 <= report["mean_vx"] <= 0.65
            and report["mean_abs_vy"] <= 0.05
            and report["mean_abs_yaw_rate"] <= 0.10
            and report["action_saturation"] < 0.10
            and report["paired_reward_delta_mean"] > 0.0
        )
        event_paths = sorted(
            (tensorboard_root / seed_dir.name).rglob("events.out.tfevents.*")
        )
        if event_paths:
            event_path = event_paths[-1]
            saturation = _tail_scalar(
                event_path,
                "go2/deterministic_action_saturation",
            )
            clip_fraction = _tail_scalar(event_path, "train/clip_fraction")
            approx_kl = _tail_scalar(event_path, "train/approx_kl")
            report["tail_action_saturation_mean"] = (
                float(np.mean(saturation)) if saturation.size else None
            )
            report["tail_clip_fraction_median"] = (
                float(np.median(clip_fraction)) if clip_fraction.size else None
            )
            report["tail_approx_kl_p95"] = (
                float(np.quantile(approx_kl, 0.95)) if approx_kl.size else None
            )
            report["passes_stability_gate"] = bool(
                saturation.size
                and clip_fraction.size
                and approx_kl.size
                and report["tail_action_saturation_mean"] < 0.10
                and report["tail_clip_fraction_median"] <= 0.25
                and report["tail_approx_kl_p95"] <= 0.05
            )
        else:
            report["tensorboard_missing"] = str(
                tensorboard_root / seed_dir.name
            )
            report["passes_stability_gate"] = False
        report["passes_seed_gate"] = bool(
            report["passes_behavior_gate"]
            and report["passes_stability_gate"]
        )
        seed_reports.append(report)

    if not seed_reports:
        print("No complete seed holdouts found.", file=sys.stderr)
        return 1

    numeric_keys = [
        "mean_reward",
        "vx_rmse",
        "mean_vx",
        "mean_abs_vy",
        "mean_abs_yaw_rate",
        "action_saturation",
        "paired_reward_delta_mean",
    ]
    aggregate = {
        key: {
            "mean": float(np.mean([report[key] for report in seed_reports])),
            "std": float(np.std([report[key] for report in seed_reports])),
        }
        for key in numeric_keys
    }
    passed = sum(report["passes_seed_gate"] for report in seed_reports)
    summary = {
        "experiment_root": str(root),
        "seeds": seed_reports,
        "aggregate": aggregate,
        "passing_seeds": passed,
        "passes_multiseed_gate": passed >= 2,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if summary["passes_multiseed_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
