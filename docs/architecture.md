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
conda run -n sac_sb3_demo python main.py --config configs/sac/baseline.yaml
conda run -n sac_sb3_demo python main.py --config configs/sac/parallel/parallel_8env.yaml
conda run -n sac_sb3_demo python main.py --config configs/ppo/parallel.yaml
conda run -n sac_sb3_demo python main.py --config configs/ppo/parallel_large.yaml
conda run -n sac_sb3_demo python main.py --config configs/smoke/sac_ltc.yaml
conda run -n sac_sb3_demo python main.py --search-config configs/search/sac_mlp_4070_balanced.yaml
```

The top-level call path is intentionally explicit:

```text
main.py
  ├─ --config
  │    -> config.load_config
  │    -> training.run_experiment
  │    -> training.train_variant
  │    -> model_factory.build_model
  │
  └─ --search-config
       -> search.load_search_config
       -> search.run_search / search.run_revalidation
       -> training.run_experiment
```

`main.py` owns argument parsing and makes the three workflows—normal training,
search, and revalidation—visible to a new reader. `config.py` validates each
YAML section and creates the immutable runtime config. `training.py` owns the
top-to-bottom lifecycle: create environments, train, evaluate, and clean up.
`model_factory.py` translates a validated config into one SB3 model, while
`reporting.py` owns parameter counting and the stable JSON output contract.

Hyperparameter search is a separate orchestration package, not a second trainer.
`search/config.py` parses the strict search YAML, `search/runner.py` owns
Optuna/ASHA execution, `search/revalidation.py` performs multi-seed selection,
and `search/artifacts.py` writes reproducibility artifacts. Every trial still
renders one normal experiment YAML and calls `training.run_experiment()`.
`hyperparameter_search.py` remains only as a compatibility import facade.

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

Configs are grouped by purpose: `configs/sac/` for formal SAC experiments,
`configs/sac/parallel/` for SAC worker-count or batching studies,
`configs/ppo/` for formal PPO experiments, and `configs/smoke/` for short
pipeline checks. `configs/search/` contains Optuna study YAMLs and their dedicated
search smoke bases. This directory structure is only for discovery; every YAML
still follows the same runtime schema.

`configs/sac/baseline.yaml` selects only `mlp` and sets `frame_stack: 1`. It uses the
same code as the stacked comparison, so there is no second baseline training
loop to drift out of sync.

`configs/sac/parallel/parallel_8env.yaml` uses eight subprocess training environments.
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

`configs/ppo/parallel.yaml` uses the same environment and callback lifecycle but
switches the model registry to PPO. Sixteen workers each collect 1,024 steps,
so every rollout contains 16,384 transitions. Configuration validation requires
the total timesteps to contain an exact number of complete rollouts and requires
the rollout size to be divisible by the minibatch size. See
[`parallel_ppo_training.md`](parallel_ppo_training.md) for the parameter basis,
rollout equations, and machine-dependent throughput caveat.

`configs/ppo/parallel_large.yaml` keeps the same rollout design but uses separate
`[400, 300]` actor/value towers, batch size 256, mandatory CUDA, and distinct
output/TensorBoard roots. It is a capacity-and-hardware experiment, while
`configs/ppo/parallel.yaml` remains the CPU RL-Zoo-style strong baseline.

## Main Modules

- `sac_experiments/config.py`: grouped YAML schema and section validation.
- `sac_experiments/training.py`: sequential variant training lifecycle.
- `sac_experiments/model_factory.py`: SB3 algorithm and policy construction.
- `sac_experiments/reporting.py`: parameter counts and JSON summaries.
- `sac_experiments/search/`: search config, TPE/ASHA execution, artifacts, and
  statistical revalidation.
- `sac_experiments/lunarlander_common.py`: environment, wrappers, CUDA checks, and evaluation helpers.
- `sac_experiments/variants.py`: explicit variant metadata and policy registry.
- `sac_experiments/ltc_features.py`: simple, circuit, and residual LTC feature extractors.
- `sac_experiments/rbf_policies.py`: shared Gaussian RBF layer plus PPO RBF policies.
- `sac_experiments/rbf_sac_policies.py`: SAC RBF actor and twin-Q policy classes.

The GIF renderer loads a model first and infers whether it needs a single frame,
stacked observations, or previous-action history. Rendering therefore stays
separate from the training entrypoint without needing a second config parser.

## Reading Order

1. `main.py`：先看训练、搜索、复验如何分流。
2. `configs/sac/ltc_comparison.yaml`：理解一次实验如何描述。
3. `sac_experiments/config.py`：看每个 YAML section 如何变成运行时配置。
4. `sac_experiments/training.py`：沿着实验和单 variant 生命周期阅读。
5. `sac_experiments/variants.py`：理解 variant 如何选择策略和特征结构。
6. `sac_experiments/ltc_features.py` 或 `rbf_*.py`：最后进入具体网络实现。
7. `sac_experiments/search/`：需要调参时再读独立搜索工作流。

## RBF Policy Extensions

`rbf_64`, `rbf_192`, `rbf_actor_mlp_critic_64`, and
`rbf_actor_mlp_critic_192` select PPO `ActorCriticPolicy` subclasses or SAC
custom policies through the same variant registry; no second training entrypoint
is introduced. SAC additionally exposes `sac_rbf_matched` and
`sac_rbf_actor_mlp_critic_matched`, whose 6050/6255 basis counts match the
existing SAC optimizer capacity. All RBF variants require `frame_stack: 1`,
preserve the environment/evaluation lifecycle, and record network and optimizer
parameter counts in JSON summaries. See [`rbf_ppo.md`](rbf_ppo.md) and
[`rbf_sac.md`](rbf_sac.md) for the respective protocols.

This mirrors the actual lifecycle: describe the experiment, validate it, train
the selected variants, and inspect the feature implementation only when needed.
