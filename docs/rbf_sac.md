# RBF-SAC：以局部基函数替代 SAC 的 actor / Q 网络

## 1. 研究问题与边界

该实验沿用 RBF 自适应函数逼近的想法：用可学习中心、对角带宽和线性读出表示非线性映射。
它不把反步控制律搬进 SAC；SAC 的 replay buffer、熵正则、双 Q、target critic 和 Polyak
更新均保持 Stable-Baselines3 的原始实现。因此它不是闭环稳定性证明，RBF 是否有效只由同
预算多 seed 的学习曲线、评估回报和吞吐量决定。

所有 RBF-SAC 组使用单帧、8 维 `LunarLanderContinuous-v3` observation。这样 RBF 与
MLP 看见同一 Markov 状态，不引入 frame stack、动作历史或归一化作为额外变量。

## 2. SAC 中的 RBF 结构

对输入 $x\in\mathbb{R}^{d}$，第 $j$ 个基函数为：

$$
\phi_j(x)=\exp\left[-\frac{1}{2}\sum_{k=1}^{d}
\left(\frac{x_k-c_{jk}}{\sigma_{jk}}\right)^2\right],\qquad
\sigma_{jk}=\operatorname{softplus}(r_{jk})+\sigma_{\min}.
$$

中心 $c$ 与带宽的无约束参数 $r$ 都由 SAC 的梯度更新学习。actor 使用
`RBF(state) -> Gaussian mean/log-std heads`；严格 RBF critic 对每个独立 Q 网络使用
`RBF([state, action]) -> linear Q head`。SB3 仍创建与维护第二个 online Q 和两份 target
Q；后者不进入 optimizer，只通过 Polyak 更新跟随 online critic。

| Variant | Actor | Online / target critic | 用途 |
|---|---|---|---|
| `mlp` | `[400,300]` MLP | twin MLP Q | SAC 强基线。 |
| `rbf_64` / `rbf_192` | RBF | twin RBF Q | 小/中等 RBF 容量扫描。 |
| `sac_rbf_matched` | 6050-center RBF | twin 6050-center RBF Q | 与 MLP 的实际优化参数量匹配。 |
| `rbf_actor_mlp_critic_64` / `_192` | RBF | twin MLP Q | 只替换 actor 的容量扫描。 |
| `sac_rbf_actor_mlp_critic_matched` | 6255-center RBF | twin MLP Q | 与 MLP actor 参数量匹配。 |

## 3. 为什么匹配 6050 / 6255，而不是 PPO 的参数量

PPO 的 value 网络与 SAC 的 twin-Q / target-Q 结构不同，跨算法对齐总参数量不具备明确的
因果解释。本实验只在同一 SAC 更新图内匹配 **实际进入 optimizer** 的参数量。

在 8 维状态、2 维动作和 `[400,300]` MLP 设置下：

| 模块 | MLP 优化参数 | RBF 参数公式 |
|---|---:|---:|
| actor | 125,104 | $20m+4$ |
| online twin-Q | 250,002 | $42m+2$ |
| actor + online twin-Q | 375,106 | $62m+6$ |

所以严格 RBF 取 $m=6050$，恰好满足 $62m+6=375{,}106$；actor-only 消融取
$m=6255$，恰好满足 $20m+4=125{,}104$。target critic 会被额外记录，但不作为匹配目标，
因为它不是 optimizer 的独立学习容量。

64/192 的结论只能说明小/中等容量 RBF 的表现；若它们失败，不能据此断言 RBF 表示必然
无效。6050/6255 两组用于分离这种容量混淆，同时也应单独报告 wall-clock 成本。

## 4. 公平协议与多 seed 运行

正式配置 [`configs/sac/rbf_comparison.yaml`](../configs/sac/rbf_comparison.yaml) 中，每个
variant 都使用 8 个同步环境、500,000 total transitions、10,000 transition warmup、
每 10,000 transitions deterministic evaluation，以及 `train_freq=1`、`gradient_steps=8`。
因此每个 VecEnv step 收集 8 条 transition 后进行 8 次梯度更新，近似保持 1:1 的
更新/样本比例。7 组 variant 在同一 seed 内串行训练，避免共享输出、TensorBoard 或 learner。

[`run_sac_rbf_multiseed.sh`](../run_sac_rbf_multiseed.sh) 默认运行 `42 123 456` 三个 seed，
即 21 个正式任务。macOS 默认 CPU；Ubuntu 检测到 `nvidia-smi` 时默认 CUDA 且关闭 CPU
fallback。seed 也串行执行，避免 GPU 显存和 SubprocVecEnv worker 竞争：

```bash
CONDA_ENV=sac_sb3_demo ./run_sac_rbf_multiseed.sh
```

可显式覆盖设备、解释器或 seed：

```bash
DEVICE=cpu PYTHON=/path/to/python ./run_sac_rbf_multiseed.sh
SEED_LIST="42 123 456 789 2026" ./run_sac_rbf_multiseed.sh
```

短配置 [`configs/smoke/sac_rbf.yaml`](../configs/smoke/sac_rbf.yaml) 会运行全部七组，
覆盖多环境采样、至少一次 SAC 更新、评估、保存、重载和 summary；其 reward 不是研究结果。

## 5. 结果解释

每个 `eval_summary.json` 记录 RBF 初始化、中心数量、网络结构、总 policy 参数和
`parameter_counts`。SAC 的 `parameter_counts.optimized_total` 是容量匹配的主指标；
`target_critic` 仅用于审计 target 网络副本。正式结果应至少报告：

- 每 seed 的 final / best evaluation reward，以及三 seed 均值和离散性；
- `training_wall_time_seconds` 与采样吞吐；
- actor、online critic、target critic、optimizer 参数量；
- 小/中等容量 RBF 与匹配容量 RBF 的差异，而非只用单个结果宣称结构优劣。
