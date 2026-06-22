# Repository Guidelines

## Project Structure & Module Organization

This repository contains small Stable-Baselines3 SAC experiments and notes.

- `sac_sb3_pendulum_demo.py`: Pendulum-v1 SAC training demo.
- `render_sac_pendulum_gif.py`: renders a trained Pendulum policy to GIF.
- `sac_sb3_lunarlander_demo.py`: LunarLanderContinuous-v3 SAC baseline.
- `sac_lunarlander_ltc_compare.py`: comparison between stacked MLP SAC and stacked LTC-feature SAC.
- `render_sac_lunarlander_gif.py`: renders a trained LunarLander policy.
- `requirements-sac-demo.txt`: Python dependencies for the isolated demo environment.
- `SAC_IMPLEMENTATIONS.md` and `SAC_SB3_TRICKS.md`: research notes and experiment rationale.
- `outputs/`, `runs/`, and `training_logs/`: generated artifacts, TensorBoard logs, checkpoints, summaries, and process logs.

## Build, Test, and Development Commands

Use the isolated conda environment; do not install into `cybernetic_env` directly.

```bash
conda create -n sac_sb3_demo --clone cybernetic_env
conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
```

Syntax check scripts:

```bash
conda run -n sac_sb3_demo python -m py_compile sac_sb3_pendulum_demo.py sac_sb3_lunarlander_demo.py sac_lunarlander_ltc_compare.py
```

Run examples:

```bash
conda run -n sac_sb3_demo python sac_sb3_pendulum_demo.py
conda run -n sac_sb3_demo python sac_sb3_lunarlander_demo.py
```

For GPU training from this agent environment, use elevated execution because the sandbox may hide CUDA.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, type hints where helpful, and `argparse` for script interfaces. Keep experiment defaults as constants near the top of each script. Use descriptive snake_case names for files, functions, variables, output directories, and TensorBoard run names.

## Testing Guidelines

There is no formal test suite yet. Validate changes with `py_compile` and a short smoke run before long training:

```bash
conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py --timesteps 2000 --eval-episodes 2 --eval-freq 1000 --allow-cpu
```

Do not treat smoke-run rewards as research results.

## Commit & Pull Request Guidelines

No Git history is available in this workspace. Use concise imperative commit messages, for example `Add LTC comparison experiment`. Pull requests should describe the experiment change, list commands run, mention generated artifacts, and include TensorBoard or summary paths when training behavior changes.

## Security & Configuration Tips

Keep `cybernetic_env` clean. Install new dependencies only in `sac_sb3_demo`. Avoid committing large generated files from `outputs/`, `runs/`, or `training_logs/` unless the artifact is explicitly needed for review.
