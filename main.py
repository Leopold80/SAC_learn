"""Readable command-line entrypoint for every LunarLander workflow."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sac_experiments.config import DEFAULT_CONFIG_PATH, load_config
from sac_experiments.training import run_experiment


def parse_args(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> argparse.Namespace:
    """Parse the three supported workflows: train, search, or revalidate."""

    parser = argparse.ArgumentParser(
        description="Train or tune SAC/PPO on LunarLanderContinuous-v3."
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
    parser.add_argument(
        "--revalidate",
        action="store_true",
        help="Revalidate completed search trials.",
    )
    args = parser.parse_args(argv)
    if (args.resume or args.revalidate) and args.search_config is None:
        parser.error("--resume and --revalidate require --search-config.")
    if args.resume and args.revalidate:
        parser.error("--resume cannot be combined with --revalidate.")
    return args


def main(
    argv: Sequence[str] | None = None,
    default_config_path: Path = DEFAULT_CONFIG_PATH,
) -> None:
    """Load configuration, then explicitly dispatch the selected workflow."""

    args = parse_args(argv, default_config_path)
    if args.search_config is None:
        run_experiment(load_config(args.config or default_config_path))
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
