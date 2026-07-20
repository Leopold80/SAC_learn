# 多环境 PPO 训练设计

## 1. 目标与边界

PPO 与现有 SAC 共用同一个 `main.py`、YAML 配置入口、LunarLander 环境工厂、
variant 注册表、评估回调和输出目录约定。差异只保留在算法构造和算法专属超参数中：

- SAC 从 replay buffer 反复抽样，是 off-policy；
- PPO 先收满一批同步 rollout，再对这批 on-policy 数据做多轮 minibatch 更新；
- 多环境只并行采集轨迹，仍然只有一个 PPO learner 和一套 policy/value 网络。

正式配置是 [`configs/ppo_parallel.yaml`](../configs/ppo_parallel.yaml)，快速管线检查是
[`configs/ppo_parallel_smoke.yaml`](../configs/ppo_parallel_smoke.yaml)。

## 2. 为什么选择 16 个环境

正式配置以 Stable-Baselines3 2.7 对应的
[RL Baselines3 Zoo PPO 配置](https://github.com/DLR-RM/rl-baselines3-zoo/blob/v2.7.0/hyperparams/ppo.yml)
为强基线。它为 `LunarLanderContinuous-v3` 使用 16 个并行环境，并采用：

```yaml
ppo:
  learning_rate: 0.0003
  learning_rate_schedule: constant
  policy_net_arch: [64, 64]
  n_steps: 1024
  batch_size: 64
  n_epochs: 4
  gamma: 0.999
  gae_lambda: 0.98
  clip_range: 0.2
  ent_coef: 0.01
```

这组参数优先给 PPO 提供大量同时采集、彼此独立的轨迹，并使用长时域折扣和 GAE 来处理
着陆任务。它是一个有依据的起点，不是对所有机器或所有随机种子的全局最优保证。本机只有
8 个逻辑 CPU，因此若目标变成纯 wall-clock 吞吐，应另行比较 `n_envs=4/8/16`；不能预设
进程越多一定越快。

SB3 对 MLP PPO 会优先建议 CPU，因为小型 actor/value 网络通常无法有效利用 GPU，环境
采样和进程通信反而是主要成本。因此正式 PPO 配置显式使用 `device: cpu`。

## 3. Rollout 与更新计数

设：

- $n$ 为 `environment.n_envs`；
- $T$ 为 `ppo.n_steps`，即每个环境每轮采集的步数；
- $B$ 为 `ppo.batch_size`；
- $E$ 为 `ppo.n_epochs`。

一轮 PPO 的新数据量为：

$$
R = nT
$$

每个 epoch 的 minibatch 数和每轮 optimizer step 数分别为：

$$
M = \frac{R}{B}, \qquad U = E\frac{R}{B}
$$

正式配置中：

```text
n = 16
T = 1024
R = 16384 transitions
B = 64
M = 256 minibatches / epoch
E = 4
U = 1024 optimizer steps / rollout
```

每条 rollout 数据会被复用 4 个 epoch，但训练期间不会像 SAC 一样混入更早的 replay
buffer 数据。

## 4. 为什么 timesteps 使用 1048576

SB3 PPO 只有在收满一轮 rollout 后才更新并检查是否达到总步数。如果 YAML 写
`timesteps: 1000000`，16 环境、每环境 1024 步的配置实际会运行到 1,015,808 transitions。

本仓库要求：

$$
training.timesteps \bmod (n_{envs} \times n_{steps}) = 0
$$

正式配置使用：

$$
1{,}048{,}576 = 64 \times 16 \times 1024
$$

因此恰好完成 64 轮 rollout，配置值、`model.num_timesteps` 和 summary 口径完全一致。

同时还要求 rollout size 能被 `batch_size` 整除，避免每个 epoch 出现截短 minibatch。

## 5. 评估、checkpoint 与随机种子

每次 VecEnv step 会从全部 worker 各得到一条 transition，但 SB3 callback 的 `n_calls`
只增加 1。因此 YAML 中按总 transitions 表示的 `evaluation.frequency` 会在训练器内部转换为：

$$
callback\_freq = \frac{evaluation.frequency}{n_{envs}}
$$

正式配置每 32,768 transitions 评估和保存一次，也就是每 2,048 次 VecEnv step。

worker seed 为 `training.seed + worker_index`；评估环境保持独立，使用
`training.seed + n_envs`。评估采用 deterministic policy，best model 与 final model 分开保存。

## 6. 运行与验证

正式训练：

```bash
conda run -n sac_sb3_demo python main.py --config configs/ppo_parallel.yaml
```

多进程 smoke：

```bash
MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
conda run -n sac_sb3_demo python main.py --config configs/ppo_parallel_smoke.yaml
```

smoke 只验证两进程采样、rollout、更新、评估、保存、重新加载与清理是否贯通；它的奖励
没有研究意义。

summary 额外记录：

- `rollout_size_transitions`；
- `minibatches_per_epoch`；
- `optimizer_steps_per_rollout`；
- `sample_reuse_epochs`；
- 实际 `sampled_transitions`、wall-clock 与 transitions/s。

## 7. 当前不启用的选项

- 不加入 `VecNormalize`：RL-Zoo 的这个条目未要求 normalization，避免改变已验证 recipe；
- 不启用 gSDE：保持该 recipe 和 SB3 PPO 默认探索形式；
- 不照搬 SAC 的 `[400, 300]` 网络：PPO 使用其成熟的 `[64, 64]` actor/value MLP；
- 不把 GPU 当作默认优势：MLP PPO 的计算规模通常不足以抵消 CPU/GPU 搬运开销。
