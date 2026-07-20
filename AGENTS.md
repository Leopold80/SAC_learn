# Repository Guidelines

## Project Structure & Module Organization

This repository contains Stable-Baselines3 SAC/PPO + LTC experiments for
`LunarLanderContinuous-v3` only.

- `main.py`: the only training entrypoint; all experiment choices come from YAML.
- `render_sac_lunarlander_gif.py`: renders a trained LunarLander policy.
- `configs/`: full comparison, single-frame baseline, parallel-environment, and smoke YAML configs.
- `sac_experiments/`: config validation, unified training, environment helpers, variants, and LTC feature extractors.
- `requirements-sac-demo.txt`: Python dependencies for the isolated demo environment.
- `docs/`: architecture, parallel SAC/PPO design, research roadmap, Windows migration guide, and algorithm notes.
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
  main.py \
  render_sac_lunarlander_gif.py \
  sac_experiments/config.py \
  sac_experiments/training.py \
  sac_experiments/lunarlander_common.py \
  sac_experiments/variants.py \
  sac_experiments/ltc_features.py
```

Run examples:

```bash
conda run -n sac_sb3_demo python main.py
conda run -n sac_sb3_demo python main.py --config configs/baseline.yaml
conda run -n sac_sb3_demo python main.py --config configs/parallel_baseline.yaml
conda run -n sac_sb3_demo python main.py --config configs/ppo_parallel.yaml
```

For GPU training from this agent environment, use elevated execution because the sandbox may hide CUDA. Also use elevated execution for `SubprocVecEnv` runs when the macOS sandbox blocks multiprocessing with `PermissionError: Operation not permitted`.

## Coding Style & Naming Conventions

Use Python 3.12-compatible code, 4-space indentation, type hints where helpful, and `argparse` only for small script interfaces. Put longer experiment settings in YAML configs. Use descriptive snake_case names for files, functions, variables, output directories, and TensorBoard run names.

## Testing Guidelines

There is no formal test suite yet. Validate changes with `py_compile` and the relevant short smoke run before long training:

```bash
conda run -n sac_sb3_demo python main.py --config configs/smoke.yaml
conda run -n sac_sb3_demo python main.py --config configs/parallel_smoke.yaml
conda run -n sac_sb3_demo python main.py --config configs/ppo_parallel_smoke.yaml
```

Run `parallel_smoke.yaml` when changing the SAC vector path and `ppo_parallel_smoke.yaml` when changing PPO rollout, minibatch, model dispatch, worker construction, callback frequencies, seeding, or environment cleanup. Do not treat smoke-run rewards as research results.

`evaluation.frequency` is expressed in total transitions. Keep both `training.timesteps` and `evaluation.frequency` divisible by `environment.n_envs`; the training code converts callback and checkpoint frequencies to VecEnv steps. The formal SAC baseline uses eight environments with `train_freq: 1` and `gradient_steps: 8`, preserving an approximately 1:1 gradient-update/transition ratio without copying PPO's more aggressive 16-worker rollout setup.

For PPO, also keep `training.timesteps` divisible by `environment.n_envs * ppo.n_steps`, and keep the rollout size divisible by `ppo.batch_size`. The formal PPO baseline follows the SB3 2.7 RL-Zoo LunarLanderContinuous recipe with 16 environments, 1,024 steps per worker, batch size 64, four epochs, and CPU execution. Treat it as a strong baseline, not a guarantee that 16 processes maximize wall-clock throughput on every machine.

## Commit & Pull Request Guidelines

Use concise imperative commit messages, for example `Refactor LunarLander experiment config`. Pull requests should describe the experiment change, list commands run, mention generated artifacts, and include TensorBoard or summary paths when training behavior changes.

## Security & Configuration Tips

Keep `cybernetic_env` clean. Install new dependencies only in `sac_sb3_demo`. Give every concurrent or repeated run a unique `output.run_tag`; the launcher intentionally refuses to reuse non-empty output or TensorBoard directories. Avoid committing large generated files from `outputs/`, `runs/`, or `training_logs/` unless the artifact is explicitly needed for review.
