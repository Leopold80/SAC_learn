"""Compatibility entrypoint for the standalone LunarLander SAC baseline.

The workflow lives in ``sac_experiments.lunarlander_baseline`` so that all
LunarLander experiment logic has one package-level reading path.
"""

from sac_experiments.lunarlander_baseline import main


if __name__ == "__main__":
    main()
