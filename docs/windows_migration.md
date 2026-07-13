# Windows Migration Guide for Codex

This guide is for a Codex agent running on a Windows machine that needs to run
the SAC + LTC LunarLander experiments from this repository.

## Goal

Reproduce the current Linux SAC experiment workflow on Windows:

- Stable-Baselines3 SAC on `LunarLanderContinuous-v3`
- four LunarLander comparison variants: `mlp`, `ltc`, `ltc_residual`, `ltc_residual_action`
- CUDA acceleration when an NVIDIA GPU is available
- TensorBoard monitoring with clean run names
- GIF rendering for trained policies

Do not install into any shared research environment. Use a dedicated Windows
conda environment.

## Recommended Environment

Create a fresh environment:

```powershell
conda create -n sac_sb3_demo python=3.12
conda activate sac_sb3_demo
```

Install CUDA PyTorch first. For CUDA 12.4:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.get_float32_matmul_precision()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Then install project dependencies:

```powershell
python -m pip install -r requirements-sac-demo.txt
```

If `gymnasium[box2d]` or Box2D fails on Windows, install SWIG from conda-forge
and retry:

```powershell
conda install -c conda-forge swig
python -m pip install "gymnasium[box2d]"
```

If native Windows Box2D remains painful, prefer WSL2 with NVIDIA CUDA.

## Basic Checks

Run static checks:

```powershell
python -m py_compile `
  sac_lunarlander_ltc_compare.py `
  sac_sb3_lunarlander_demo.py `
  render_sac_lunarlander_gif.py `
  sac_experiments/lunarlander_baseline.py `
  sac_experiments/lunarlander_compare.py `
  sac_experiments/lunarlander_common.py `
  sac_experiments/variants.py `
  sac_experiments/ltc_features.py
```

Check YAML config loading:

```powershell
python -c "from pathlib import Path; from sac_experiments.lunarlander_compare import load_config; c=load_config(Path('configs/lunarlander.yaml')); print(c.variants)"
```

Expected:

```text
['mlp', 'ltc', 'ltc_residual', 'ltc_residual_action']
```

Check observation shapes:

```powershell
python -c "from sac_experiments.lunarlander_common import make_lunarlander_env; e=make_lunarlander_env(42, 4); print(e.observation_space.shape); e.close(); e=make_lunarlander_env(42, 4, use_action_history=True); print(e.observation_space.shape); e.close()"
```

Expected:

```text
(4, 8)
(4, 10)
```

## Running Experiments

Smoke test:

```powershell
python sac_lunarlander_ltc_compare.py --config configs/smoke.yaml
```

Full sequential run:

```powershell
python sac_lunarlander_ltc_compare.py --config configs/lunarlander.yaml
```

The Linux workflow often starts four variants as four background processes.
Do not copy Linux `setsid`, `bash -lc`, `OMP_NUM_THREADS=...`, or SSH tunnel
commands directly into PowerShell. On Windows, either run variants sequentially
with the YAML entrypoint, or create a small Python launcher that starts four
subprocesses with these environment variables:

```text
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VARIANT=<mlp|ltc|ltc_residual|ltc_residual_action>
RUN_TAG=<your_run_tag>
```

For a parallel Windows launcher, call `load_config()`, set `c.variants` to a
single variant, append `RUN_TAG` to `c.output_dir` and `c.tensorboard_log`, then
call `run_experiment(c)`. This mirrors the Linux parallel workflow without
shell-specific behavior.

## TensorBoard on Windows

Windows local TensorBoard does not need SSH port forwarding:

```powershell
tensorboard --logdir runs/lunarlander/<run_tag> --host 127.0.0.1 --port 6009
```

Open:

```text
http://127.0.0.1:6009
```

Keep run names flat and readable. The expected TensorBoard run names are:

```text
mlp_1
ltc_1
ltc_res_1
ltc_act_1
```

If old curves appear, stop TensorBoard and restart it with `--logdir` pointing
to one specific clean run tag, not the whole `runs/lunarlander` root.

## GIF Rendering

Standard models:

```powershell
python render_sac_lunarlander_gif.py `
  --model-path outputs/lunarlander/<run_tag>/mlp/best_model/best_model.zip `
  --output-path outputs/lunarlander/<run_tag>/visualizations/mlp_best.gif
```

Action-history models are inferred automatically from their saved observation space:

```powershell
python render_sac_lunarlander_gif.py `
  --model-path outputs/lunarlander/<run_tag>/ltc_residual_action/best_model/best_model.zip `
  --output-path outputs/lunarlander/<run_tag>/visualizations/ltc_act_best.gif
```

## GPU Notes

The code defaults to `device: cuda` in `configs/lunarlander.yaml`. If CUDA is
not visible, the full run should fail loudly instead of silently using CPU. For
debugging only, use `configs/smoke.yaml`, where CPU fallback is allowed.

On Windows, verify GPU use with:

```powershell
nvidia-smi
```

Expect multiple Python processes if running variants in parallel. MLP usually
uses much less VRAM and runs much faster than LTC variants.

## What To Preserve

Keep these experiment conventions unchanged:

- train/eval environments are separated
- evaluations are deterministic
- best and final models are saved separately
- `eval_summary.json` and `evaluations.npz` are the source of record
- `ltc_simple` remains legacy and is not part of the default comparison
- action history is used only by `ltc_residual_action`
- raw residual branch in `ltc_residual_action` uses only original observation

## Common Failure Modes

- **Box2D install failure**: install `swig` via conda-forge, then reinstall `gymnasium[box2d]`.
- **No CUDA**: install CUDA PyTorch wheel first, then project dependencies.
- **TensorBoard shows old runs**: point `--logdir` to one clean run tag.
- **GIF observation-space mismatch**: do not override `--frame-stack` or `--action-history`; the renderer infers both from the saved model and reports incompatible manual overrides.
- **PowerShell line breaks fail**: use backtick `` ` `` for multi-line commands, not `\`.

## When To Prefer WSL2

Use WSL2 if native Windows dependency setup becomes more expensive than the
experiment itself. WSL2 usually matches the Linux workflow more closely,
especially for Box2D, CUDA tooling, background processes, and long-running
TensorBoard sessions.
