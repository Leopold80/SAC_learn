"""Thin entrypoint for LunarLander SAC comparison experiments.

Run the default full comparison:

    conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py

Run a short smoke test:

    conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py \
        --config configs/smoke.yaml

Experiment details live in the YAML config and in
``sac_experiments/lunarlander_compare.py``.
"""

from __future__ import annotations

from sac_experiments.lunarlander_compare import main


if __name__ == "__main__":
    main()
