# Repository Guidelines

## Project Structure & Module Organization

This repository contains Stable-Baselines3 SAC + LTC experiments for
`LunarLanderContinuous-v3` only.

- `sac_sb3_lunarlander_demo.py`: thin entrypoint for the standalone, non-stacked LunarLander baseline.
- `sac_lunarlander_ltc_compare.py`: thin entrypoint for YAML-driven LunarLander comparisons.
- `render_sac_lunarlander_gif.py`: renders a trained LunarLander policy.
- `configs/`: YAML experiment configs, including full and smoke LunarLander comparison runs.
- `sac_experiments/`: baseline / comparison workflows, environment helpers, variants, and LTC feature extractors.
- `requirements-sac-demo.txt`: Python dependencies for the isolated demo environment.
- `docs/`: architecture, research roadmap, Windows migration guide, and SAC implementation notes.
- `outputs/`, `runs/`, and `training_logs/`: generated artifacts, TensorBoard logs, checkpoints, summaries, and process logs.

## Build, Test, and Development Commands

Use the isolated conda environment; do not install into `cybernetic_env` directly.

```bash
conda create -n sac_sb3_demo --clone cybernetic_env
conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
```

Syntax check scripts:

```bash
conda run -n sac_sb3_demo python -m py_compile \
  sac_sb3_lunarlander_demo.py \
  sac_lunarlander_ltc_compare.py \
  render_sac_lunarlander_gif.py \
  sac_experiments/lunarlander_baseline.py \
  sac_experiments/lunarlander_compare.py \
  sac_experiments/lunarlander_common.py \
  sac_experiments/variants.py \
  sac_experiments/ltc_features.py
```

Run examples:

```bash
conda run -n sac_sb3_demo python sac_sb3_lunarlander_demo.py
conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py
```

For GPU training from this agent environment, use elevated execution because the sandbox may hide CUDA.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, type hints where helpful, and `argparse` only for small script interfaces. Put longer experiment settings in YAML configs. Use descriptive snake_case names for files, functions, variables, output directories, and TensorBoard run names.

## Testing Guidelines

There is no formal test suite yet. Validate changes with `py_compile` and a short smoke run before long training:

```bash
conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py --config configs/smoke.yaml
```

Do not treat smoke-run rewards as research results.

## Commit & Pull Request Guidelines

Use concise imperative commit messages, for example `Refactor LunarLander experiment config`. Pull requests should describe the experiment change, list commands run, mention generated artifacts, and include TensorBoard or summary paths when training behavior changes.

## Security & Configuration Tips

Keep `cybernetic_env` clean. Install new dependencies only in `sac_sb3_demo`. Avoid committing large generated files from `outputs/`, `runs/`, or `training_logs/` unless the artifact is explicitly needed for review.
