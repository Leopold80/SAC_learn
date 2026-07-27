# RBF-PPO：以径向基函数替代深层策略网络

## 1. 动机与研究问题

这个实验受 RBF 反步自适应控制中的函数逼近思想启发：用一组局部基函数表示非线性映射，
并让中心、宽度与线性读出权重共同适应数据。这里不把 RBF 反步控制律直接搬进强化学习；
PPO 仍然通过 clipped policy-gradient 和 value regression 更新参数，RBF 只是 actor/value
的可学习函数逼近器。因此本实验不提供反步法的闭环稳定性证明，RBF 是否胜任只由同预算
训练后的学习曲线、评估回报与吞吐数据判断。

首版只覆盖单帧、8 维状态的 `LunarLanderContinuous-v3` PPO。这样 RBF 与当前 PPO MLP
强基线看见相同的 Markov observation；没有加入 frame stack、action history 或
`VecNormalize` 以免改变研究问题。

## 2. RBF 层与两类结构

对于输入状态 $x \in \mathbb{R}^d$，第 $j$ 个对角高斯基函数为：

$$
\phi_j(x) = \exp\left[-\frac{1}{2}\sum_{k=1}^d
\left(\frac{x_k-c_{jk}}{\sigma_{jk}}\right)^2\right].
$$

- $c_{jk}$ 是可学习中心；
- $\sigma_{jk}=\operatorname{softplus}(r_{jk})+\sigma_{min}$ 是可学习的正带宽；
- 初始中心均匀采样于 `[-center_init_range, center_init_range]`，初始带宽由
  `initial_bandwidth` 精确设定。

PPO 的高斯动作分布和 value 标量输出仍由 SB3 负责。新增变体是：

| Variant | Actor | Critic | 用途 |
|---|---|---|---|
| `rbf_64` / `rbf_192` | RBF → 线性高斯策略头 | 独立 RBF → 线性 value 头 | 严格检验 RBF actor-critic 是否可学习。 |
| `rbf_actor_mlp_critic_64` / `_192` | RBF → 线性高斯策略头 | 现有 `[64, 64]` MLP → 线性 value 头 | 分离 actor 的 RBF 表示与 value 拟合能力。 |

严格 RBF 分支没有隐藏 MLP；actor 和 critic 分别拥有自己的 RBF 中心和带宽。64 个基函数
是小容量可行性检验，192 个基函数是更大容量检验，不应被视为严格参数量匹配。每次运行的
实际可训练参数量会写入 `eval_summary.json`，应以该数值而不是基函数数量解释差异。

## 3. 公平实验与结果边界

正式配置 [`configs/ppo/rbf_comparison.yaml`](../configs/ppo/rbf_comparison.yaml) 的每个
variant 都使用现有 PPO strong baseline 的设置：16 个同步环境、每环境 1,024 steps、
64 个完整 rollout、共 1,048,576 transitions，且每 32,768 transitions 作 deterministic
评估。五个 variant 在一个 seed 内顺序运行，避免共享 learner、输出目录或 TensorBoard run。

比较重点是：

- `eval/mean_reward`、best evaluation 和 final evaluation；
- `training_wall_time_seconds` 与 `sample_throughput_transitions_per_second`；
- `trainable_parameter_count`，防止把容量差异误读成结构优势；
- 至少三个独立 seed 的均值与离散性。

短配置 [`configs/smoke/ppo_rbf.yaml`](../configs/smoke/ppo_rbf.yaml) 只验证多环境采样、
PPO 更新、评估、保存和重载；它的 reward 不能作为 RBF 有效性的证据。

## 4. 多 seed 运行

脚本 [`run_ppo_rbf_multiseed.sh`](../run_ppo_rbf_multiseed.sh) 会依次运行默认的
`42 123 456` 三个 seed，并在每个 seed 内完成五个 variant。它为每次运行写入独立的
`output.run_tag`、日志和 Matplotlib cache，不覆盖已有结果。

macOS 默认使用 CPU：

```bash
CONDA_ENV=sac_sb3_demo ./run_ppo_rbf_multiseed.sh
```

在带 NVIDIA 驱动的 Ubuntu 主机上，脚本发现 `nvidia-smi` 后默认写入 `device: cuda` 且
禁止 CPU fallback；PyTorch 看不到 CUDA 时该 seed 会以明确错误记入日志。需要强制 CPU 或
指定解释器时可以覆盖：

```bash
DEVICE=cpu PYTHON=/path/to/python ./run_ppo_rbf_multiseed.sh
SEED_LIST="42 123 456 789 2026" ./run_ppo_rbf_multiseed.sh
```

脚本不并行启动 seed：这避免 GPU 显存、CPU worker 和输出目录互相竞争。正式五组 × 三 seed
共 15 个完整训练任务，应在目标主机上执行；本机只运行 smoke。
