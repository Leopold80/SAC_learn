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
    search/     TPE搜索与复验配置
    smoke/      快速流程检查

sac_experiments/
    config.py          分section解析与校验YAML
    training.py        统一训练生命周期编排
    model_factory.py   根据配置构造SB3模型
    reporting.py       参数统计与JSON实验摘要
    variants.py        variant元数据与策略注册表
    search/            TPE/ASHA、产物和统计复验
    ltc_features.py    LTC feature extractor
    rbf_*.py           RBF策略实现

docs/
    architecture.md
    parallel_sac_training.md
    parallel_ppo_training.md
    rbf_sac.md
    rbf_ppo.md
    ltc.md
    hyperparameter_search.md
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

单卡 CUDA 超参数搜索：

```bash
python main.py --search-config configs/search/sac_mlp_4070_balanced.yaml
python main.py --search-config configs/search/sac_mlp_4070_balanced.yaml --revalidate
```

## 超参数搜索工作流索引

超参数系统分为四个职责不同的阶段：

| 阶段 | 负责什么 | 详细说明 |
|---|---|---|
| TPE | 根据历史 trial 建议下一组超参数 | [TPE 的功能与采样过程](docs/hyperparameter_search.md#4-tpe-的功能与一次采样过程) |
| ASHA | 在资源节点停止明显落后的 trial | [ASHA 的功能与剪枝过程](docs/hyperparameter_search.md#5-asha-的功能与剪枝过程) |
| 自动复验 | 对 top-k 配置执行独立多 seed 完整训练和 95% LCB 比较 | [自动统计复验的完整过程](docs/hyperparameter_search.md#9-自动统计复验的完整过程) |
| 正式训练 | 使用冠军配置训练并保存最终模型、checkpoint 和 TensorBoard | [复验之后的正式训练](docs/hyperparameter_search.md#10-复验之后正式训练发生什么) |

文字化的端到端使用说明见：

- [完整命令工作流](docs/hyperparameter_search.md#12-完整命令工作流)；
- [产物目录与推荐阅读顺序](docs/hyperparameter_search.md#13-产物目录与推荐阅读顺序)；
- [常见误解与判断边界](docs/hyperparameter_search.md#14-常见误解与判断边界)。

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

超参数搜索和自动统计复验的完整动机、配置与结果解释见
[`docs/hyperparameter_search.md`](docs/hyperparameter_search.md)。
