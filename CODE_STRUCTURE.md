# Code Structure Notes

This repository keeps runnable scripts small and moves reusable experiment logic
into `sac_experiments/`.

## LunarLander Compare Flow

Start here:

```bash
conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py
```

The script only calls:

```python
from sac_experiments.lunarlander_compare import main
```

The default config is:

```text
configs/lunarlander.yaml
```

Use the smoke config for short debugging:

```bash
conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py \
    --config configs/smoke.yaml
```

## Main Modules

- `sac_experiments/lunarlander_compare.py`: YAML loading, training loop, callbacks, summaries.
- `sac_experiments/lunarlander_common.py`: environment creation, action-history wrapper, CUDA checks, evaluation helpers, SAC defaults.
- `sac_experiments/variants.py`: maps variant names to SB3 policy kwargs.
- `sac_experiments/ltc_features.py`: simple LTC and circuit LTC feature extractors.

Current short variant names:

- `mlp`: stacked-observation MLP baseline.
- `ltc`: circuit LTC feature extractor, the main LTC branch.
- `ltc_residual`: circuit LTC plus raw-observation residual / concat fusion.
- `ltc_residual_action`: residual LTC where previous actions enter only the LTC branch.
- `ltc_simple`: legacy simple LTC branch, kept only for old-result checks.

## How To Read The Code

Read in this order:

1. `configs/lunarlander.yaml`
2. `sac_lunarlander_ltc_compare.py`
3. `sac_experiments/lunarlander_compare.py`
4. `sac_experiments/variants.py`
5. `sac_experiments/ltc_features.py`

This mirrors the experiment lifecycle: choose settings, enter the workflow,
construct the variant, train SAC, then evaluate and save summaries.
