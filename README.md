# LunarLander SAC / PPO Reinforcement Learning Experiments

本仓库用于研究连续控制强化学习中的算法基线与结构改进。

当前实验环境：

- `LunarLanderContinuous-v3`
- Stable-Baselines3 SAC
- Stable-Baselines3 PPO

研究重点包括：

- SAC / PPO 可复现实验流程；
- 并行环境采样；
- RBF 函数逼近；
- LTC 时序特征提取。

## 快速入口

训练入口只有一个：

```bash
python main.py --config <yaml>
```

`main.py` 根据 YAML 配置创建环境、选择算法、训练 variant 并保存实验摘要。

主要目录：

```text
configs/
    sac/        SAC正式实验
    ppo/        PPO正式实验
    smoke/      快速流程检查

sac_experiments/
    config.py          YAML解析与约束检查
    training.py        统一训练流程
    variants.py        实验variant注册
    ltc_features.py    LTC feature extractor
    rbf_*.py           RBF策略实现

docs/
    architecture.md
    parallel_sac_training.md
    parallel_ppo_training.md
    rbf_sac.md
    rbf_ppo.md
    ltc.md
    research_roadmap.md
```

完整文档导航见：

[`docs/documentation_map.md`](docs/documentation_map.md)

## 当前实验矩阵

### SAC

基础结构：

| Variant | 说明 |
|---|---|
| `mlp` | 标准 MLP SAC baseline |
| `ltc` | Circuit LTC temporal feature extractor |
| `ltc_residual` | 原始 observation + LTC feature fusion |
| `ltc_residual_action` | LTC 分支额外使用 action history |

RBF-SAC：

| Variant | 说明 |
|---|---|
| `rbf_64` / `rbf_192` | RBF actor + critic 容量扫描 |
| `sac_rbf_matched` | 与 SAC optimizer 参数量匹配 |
| `sac_rbf_actor_mlp_critic_matched` | 仅 actor 参数量匹配 |

### PPO

支持：

- 多环境 rollout；
- GAE；
- RBF actor/value 对照。

## 运行示例

SAC baseline：

```bash
python main.py --config configs/sac/baseline.yaml
```

SAC + LTC：

```bash
python main.py --config configs/sac/ltc_comparison.yaml
```

并行 SAC：

```bash
python main.py --config configs/sac/parallel/parallel_8env.yaml
```

并行 PPO：

```bash
python main.py --config configs/ppo/parallel.yaml
```

## 实验记录原则

训练结果保存：

- `eval_summary.json`
- `experiment_summary.json`
- evaluation curve
- TensorBoard logs
- GIF visualization

模型权重和 checkpoint 不作为仓库主要同步内容。

评价不只看最高 reward，同时记录：

- best evaluation reward；
- final evaluation reward；
- 多 seed 稳定性；
- 参数量；
- wall-clock time；
- sample throughput。

## 说明

LTC、RBF 等结构均作为 feature / policy approximation 改进进行研究，不改变 SAC/PPO 的核心优化过程。

它们的有效性需要通过统一预算、多 seed 实验验证。
