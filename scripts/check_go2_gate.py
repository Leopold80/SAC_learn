#!/usr/bin/env python3
"""在进入正式多种子训练前检查 Go2 短 gate 是否健康。"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAIL_TAGS = (
    "go2/deterministic_action_saturation",
    "go2/effective_std_mean",
    "train/clip_fraction",
    "train/approx_kl",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reject a Go2 gate with NaN, saturation collapse, or all-episode falls."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("tensorboard_dir", type=Path)
    return parser.parse_args()


def _tail_values(accumulator: EventAccumulator, tag: str) -> np.ndarray:
    if tag not in accumulator.Tags()["scalars"]:
        return np.asarray([], dtype=float)
    values = np.asarray(
        [event.value for event in accumulator.Scalars(tag)],
        dtype=float,
    )
    return values[max(0, 3 * len(values) // 4):]


def main() -> int:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    tensorboard_dir = args.tensorboard_dir.resolve()
    holdout_path = output_dir / "eval_logs" / "holdout.npz"
    event_paths = sorted(tensorboard_dir.rglob("events.out.tfevents.*"))
    failures: list[str] = []

    if not holdout_path.is_file():
        failures.append(f"missing holdout: {holdout_path}")
    if not event_paths:
        failures.append(f"missing TensorBoard event: {tensorboard_dir}")
    if failures:
        for failure in failures:
            print(f"[gate failed] {failure}", file=sys.stderr)
        return 1

    with np.load(holdout_path) as data:
        names = data["diagnostic_names"].tolist()
        diagnostics = data["diagnostics"].astype(float)
        rewards = data["rewards"].astype(float)

    if not np.isfinite(rewards).all() or not np.isfinite(diagnostics).all():
        failures.append("holdout contains NaN or Inf")
    termination_codes = diagnostics[:, names.index("termination_code")]
    action_saturation = diagnostics[:, names.index("episode_action_saturation")]
    falls = int(np.count_nonzero(termination_codes))
    episodes = int(diagnostics.shape[0])
    holdout_saturation = float(np.mean(action_saturation))
    if falls == episodes:
        failures.append(f"all {episodes} holdout episodes terminated by a fall")
    if holdout_saturation >= 0.50:
        failures.append(
            f"holdout action saturation is {holdout_saturation:.3f}, expected < 0.50"
        )

    accumulator = EventAccumulator(
        str(event_paths[-1]),
        size_guidance={"scalars": 0},
    )
    accumulator.Reload()
    tail: dict[str, np.ndarray] = {
        tag: _tail_values(accumulator, tag)
        for tag in TAIL_TAGS
    }
    for tag, values in tail.items():
        if not values.size:
            failures.append(f"missing tail metric: {tag}")
        elif not np.isfinite(values).all():
            failures.append(f"tail metric contains NaN or Inf: {tag}")
    tail_saturation = tail["go2/deterministic_action_saturation"]
    if tail_saturation.size and float(np.mean(tail_saturation)) >= 0.50:
        failures.append(
            "last-quarter deterministic action saturation is "
            f"{float(np.mean(tail_saturation)):.3f}, expected < 0.50"
        )

    print(
        "gate episodes="
        f"{episodes} falls={falls} "
        f"holdout_saturation={holdout_saturation:.3f}"
    )
    if failures:
        for failure in failures:
            print(f"[gate failed] {failure}", file=sys.stderr)
        return 1
    print("Gate health check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
