#!/usr/bin/env python3
"""检查训练产物是否完整、可被 Git 收集且没有异常大文件。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


REQUIRED_OUTPUTS = {
    "experiment summary": "experiment_summary.json",
    "evaluation summary": "eval_summary.json",
    "evaluation arrays": "eval_logs/evaluations.npz",
    "holdout arrays": "eval_logs/holdout.npz",
    "best policy": "best_model/best_model.zip",
    "final policy": "final_model.zip",
}


def _is_ignored(repo_root: Path, path: Path) -> bool:
    result = subprocess.run(
        [
            "git",
            "check-ignore",
            "--no-index",
            "--quiet",
            "--",
            str(path.resolve()),
        ],
        cwd=repo_root,
        check=False,
    )
    if result.returncode == 0:
        return True
    if result.returncode == 1:
        return False
    raise RuntimeError(f"git check-ignore failed for {path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that Go2 analysis artifacts are complete and Git-visible."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("tensorboard_dir", type=Path)
    parser.add_argument(
        "--max-file-mib",
        type=float,
        default=50.0,
        help="Fail when a Git-visible artifact exceeds this size (default: 50 MiB).",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    tensorboard_dir = args.tensorboard_dir.resolve()
    for label, path in (
        ("output directory", output_dir),
        ("TensorBoard directory", tensorboard_dir),
    ):
        if not path.is_relative_to(repo_root):
            print(f"{label} must be inside the repository: {path}", file=sys.stderr)
            return 1

    missing: list[str] = []
    required: list[Path] = []
    for label, relative in REQUIRED_OUTPUTS.items():
        path = output_dir / relative
        if not path.is_file():
            missing.append(f"{label}: {path}")
        else:
            required.append(path)

    train_monitors = sorted(output_dir.glob("monitor/train/*.monitor.csv"))
    eval_monitors = sorted(output_dir.glob("monitor/eval/*.monitor.csv"))
    summary_path = output_dir / "experiment_summary.json"
    expected_train_monitors = None
    if summary_path.is_file():
        try:
            expected_train_monitors = int(
                json.loads(summary_path.read_text(encoding="utf-8"))["n_envs"]
            )
        except (KeyError, TypeError, ValueError):
            missing.append(f"valid n_envs in summary: {summary_path}")
    if expected_train_monitors is not None and len(train_monitors) != expected_train_monitors:
        missing.append(
            "train monitor count: "
            f"expected {expected_train_monitors}, found {len(train_monitors)}"
        )
    if len(eval_monitors) != 1:
        missing.append(f"eval monitor count: expected 1, found {len(eval_monitors)}")
    required.extend(train_monitors)
    required.extend(eval_monitors)

    tensorboard_events = sorted(
        tensorboard_dir.rglob("events.out.tfevents.*")
        if tensorboard_dir.is_dir()
        else []
    )
    if not tensorboard_events:
        missing.append(f"TensorBoard event: {tensorboard_dir}")
    required.extend(tensorboard_events)

    ignored = [path for path in required if _is_ignored(repo_root, path)]
    size_limit = int(args.max_file_mib * 1024 * 1024)
    visible_files = {
        path
        for root in (output_dir, tensorboard_dir)
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and not _is_ignored(repo_root, path)
    }
    oversized = [
        path for path in sorted(visible_files)
        if path.stat().st_size > size_limit
    ]

    print(f"checked={len(required)} output={output_dir}")
    for path in required:
        print(f"  {path.stat().st_size:>10}  {path}")

    if missing:
        print("\nMissing required artifacts:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
    if ignored:
        print("\nRequired artifacts ignored by Git:", file=sys.stderr)
        for path in ignored:
            print(f"  - {path}", file=sys.stderr)
    if oversized:
        print(
            f"\nGit-visible artifacts larger than {args.max_file_mib:g} MiB:",
            file=sys.stderr,
        )
        for path in oversized:
            print(f"  - {path}", file=sys.stderr)

    if missing or ignored or oversized:
        return 1
    print("\nArtifact audit passed: required analysis data is Git-visible.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
