# Code Structure Notes

The repository has one training command and one YAML-driven workflow for all
LunarLander SAC and PPO variants.

## Training Flow

Start with the full comparison:

```bash
conda run -n sac_sb3_demo python main.py
```

Choose another experiment only by changing the config path:

```bash
conda run -n sac_sb3_demo python main.py --config configs/baseline.yaml
conda run -n sac_sb3_demo python main.py --config configs/parallel_baseline.yaml
conda run -n sac_sb3_demo python main.py --config configs/ppo_parallel.yaml
conda run -n sac_sb3_demo python main.py --config configs/smoke.yaml
```

The call path is intentionally short:

```text
main.py
  -> sac_experiments.config.load_config
  -> sac_experiments.training.run_experiment
  -> train_variant
```

`main.py` owns only the command-line boundary. `config.py` owns YAML defaults,
strict validation, and the immutable runtime config. `training.py` owns the
top-to-bottom experiment lifecycle: create environments, build a variant, train,
evaluate, and save models and summaries.

## YAML Contract

Each config uses the same sections:

- `experiment`: environment, algorithm, policy, and variants.
- `environment`: observation frame stacking and number of training environments.
- `training`: timesteps, seed, device, CPU fallback, and progress display.
- `evaluation`: deterministic evaluation episode count and frequency.
- `output`: model directory, TensorBoard directory, and optional run tag.
- `sac`: learning-rate schedule, policy network, and SAC hyperparameters.
- `ppo`: learning-rate schedule, policy network, rollout, GAE, clipping, and PPO hyperparameters.
- `ltc`: LTC extractor dimensions and ODE settings.

Only `LunarLanderContinuous-v3`, SAC/PPO, and `MlpPolicy` are supported. Keeping
these values visible in YAML makes each run self-describing; validation rejects
unsupported values and unknown keys instead of pretending this is a generic RL
framework.

Before training, both resolved output paths must be empty or absent. Reusing a
completed path fails fast instead of overwriting models; use a unique,
single-segment `output.run_tag` for every additional run.

`configs/baseline.yaml` selects only `mlp` and sets `frame_stack: 1`. It uses the
same code as the stacked comparison, so there is no second baseline training
loop to drift out of sync.

`configs/parallel_baseline.yaml` uses eight subprocess training environments.
Worker seeds are `training.seed + worker_index`. Evaluation remains a separate
single environment using `training.seed + n_envs`, and evaluation/checkpoint
frequencies continue to mean total collected transitions rather than
vector-environment calls. Because each
eight-environment step collects eight transitions, this config also sets
`gradient_steps: 8` to preserve the single-environment baseline's approximate
gradient-update/transition ratio.

For the precise VecEnv step semantics, replay-buffer layout, callback-frequency
equations, seed policy, limitations, and comparison protocol, see
[`parallel_sac_training.md`](parallel_sac_training.md).

`configs/ppo_parallel.yaml` uses the same environment and callback lifecycle but
switches the model registry to PPO. Sixteen workers each collect 1,024 steps,
so every rollout contains 16,384 transitions. Configuration validation requires
the total timesteps to contain an exact number of complete rollouts and requires
the rollout size to be divisible by the minibatch size. See
[`parallel_ppo_training.md`](parallel_ppo_training.md) for the parameter basis,
rollout equations, and machine-dependent throughput caveat.

## Main Modules

- `sac_experiments/config.py`: grouped YAML schema and validation.
- `sac_experiments/training.py`: sequential variant training and summaries.
- `sac_experiments/lunarlander_common.py`: environment, wrappers, CUDA checks, and evaluation helpers.
- `sac_experiments/variants.py`: maps variant names to SB3 policy kwargs.
- `sac_experiments/ltc_features.py`: simple, circuit, and residual LTC feature extractors.

The GIF renderer loads a model first and infers whether it needs a single frame,
stacked observations, or previous-action history. Rendering therefore stays
separate from the training entrypoint without needing a second config parser.

## Reading Order

1. `configs/lunarlander.yaml`
2. `configs/ppo_parallel.yaml` when studying PPO
3. `main.py`
4. `sac_experiments/config.py`
5. `sac_experiments/training.py`
6. `sac_experiments/variants.py`
7. `sac_experiments/ltc_features.py`

This mirrors the actual lifecycle: describe the experiment, validate it, train
the selected variants, and inspect the feature implementation only when needed.
