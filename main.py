"""Command-line entrypoint for Go2 locomotion RL workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from sac_experiments.config import DEFAULT_CONFIG_PATH, load_config
from sac_experiments.training import run_experiment


def parse_args(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or tune SAC/PPO on Unitree Go2 locomotion (MuJoCo)."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"YAML experiment config. Default: {default_config_path}",
    )
    source.add_argument(
        "--search-config",
        type=Path,
        default=None,
        help="YAML hyperparameter-search config.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume a search study.")
    parser.add_argument("--revalidate", action="store_true", help="Revalidate completed search trials.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Run one ordinary experiment per seed in seed_<N> subdirectories.",
    )
    parser.add_argument(
        "--benchmark-runtime",
        action="store_true",
        help="Benchmark 8/16/32 envs on CPU/CUDA instead of training.",
    )
    parser.add_argument(
        "--benchmark-transitions",
        type=int,
        default=131_072,
        help="Transitions per runtime benchmark combination.",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("outputs/go2_runtime_benchmark.json"),
        help="Runtime benchmark JSON path.",
    )
    parser.add_argument(
        "--runtime-profile",
        type=Path,
        default=None,
        help="Apply the selected profile from a previous runtime benchmark.",
    )
    args = parser.parse_args(argv)
    if (args.resume or args.revalidate) and args.search_config is None:
        parser.error("--resume and --revalidate require --search-config.")
    if args.resume and args.revalidate:
        parser.error("--resume cannot be combined with --revalidate.")
    if args.seeds is not None:
        if args.search_config is not None:
            parser.error("--seeds cannot be combined with --search-config.")
        if any(seed < 0 for seed in args.seeds):
            parser.error("--seeds must be non-negative.")
        if len(set(args.seeds)) != len(args.seeds):
            parser.error("--seeds must not contain duplicates.")
    if args.benchmark_runtime:
        if args.search_config is not None:
            parser.error("--benchmark-runtime cannot be combined with --search-config.")
        if args.seeds is not None:
            parser.error("--benchmark-runtime cannot be combined with --seeds.")
        if (
            args.benchmark_transitions <= 0
            or args.benchmark_transitions % 2048 != 0
        ):
            parser.error("--benchmark-transitions must be a positive multiple of 2048.")
        if args.runtime_profile is not None:
            parser.error("--benchmark-runtime cannot be combined with --runtime-profile.")
    if args.runtime_profile is not None and args.search_config is not None:
        parser.error("--runtime-profile cannot be combined with --search-config.")
    return args


def main(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    args = parse_args(argv, default_config_path)
    if args.search_config is None:
        config = load_config(args.config or default_config_path)
        if args.benchmark_runtime:
            from sac_experiments.runtime_benchmark import run_runtime_benchmark

            run_runtime_benchmark(
                config,
                transitions=args.benchmark_transitions,
                output_path=args.benchmark_output,
            )
            return
        if args.runtime_profile is not None:
            from sac_experiments.runtime_benchmark import apply_runtime_profile

            config = apply_runtime_profile(config, args.runtime_profile)
        if args.seeds is None:
            run_experiment(config)
            return
        for seed in args.seeds:
            run_experiment(replace(
                config,
                seed=seed,
                output_dir=config.output_dir / f"seed_{seed}",
                tensorboard_log=config.tensorboard_log / f"seed_{seed}",
            ))
        return

    from sac_experiments.search import (
        load_search_config,
        run_revalidation,
        run_search,
    )

    search_config = load_search_config(args.search_config)
    if args.revalidate:
        run_revalidation(search_config)
    else:
        run_search(search_config, resume=args.resume)


if __name__ == "__main__":
    main()
