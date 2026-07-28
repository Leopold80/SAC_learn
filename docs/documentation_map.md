# Documentation Map

本仓库用于研究 `LunarLanderContinuous-v3` 上的连续控制强化学习实验，包括：

- Stable-Baselines3 SAC；
- 多环境 PPO 强基线；
- 函数逼近结构改进（RBF）；
- 时序特征提取器（LTC）。

LTC 只是其中一条表示学习路线，不作为整个仓库的唯一主题。

## 推荐阅读顺序

### 1. 快速开始

- `README.md`
  - 项目定位
  - 快速运行
  - 实验矩阵
  - 常用命令

### 2. 代码结构

- `docs/architecture.md`
  - `main.py -> config -> training -> variant` 调用链
  - YAML 配置协议
  - 模块职责

### 3. 强化学习算法实验

- `docs/parallel_sac_training.md`
  - VecEnv 并行采样
  - transition 定义
  - replay buffer 更新比例
  - SAC 公平比较协议

- `docs/parallel_ppo_training.md`
  - rollout
  - GAE
  - minibatch 更新
  - PPO 参数来源

### 4. 结构改进实验

- `docs/rbf_sac.md`
  - RBF actor / twin-Q
  - 参数量匹配
  - 多 seed 协议

- `docs/rbf_ppo.md`
  - RBF policy/value approximation
  - PPO 对照实验

- `docs/ltc.md`
  - LTC 数学形式
  - 离散化方法
  - extractor 实现
  - action history 设计

- `docs/research_roadmap.md`
  - 后续研究路线
  - recurrent LTC-SAC
  - sparse LTC 等方向

## 实验原则

所有结构改进均遵循：

1. 不随意修改 SAC/PPO 算法主体；
2. 保持训练预算一致；
3. 同时记录 reward、参数量、吞吐和训练时间；
4. 单次训练曲线不作为结构有效性的充分证据；
5. 重要结论需要多 seed 验证。

## 当前实验主线

```text
SAC / PPO baseline
        |
        +-- RBF function approximation
        |
        +-- LTC temporal feature extractor
        |
        +-- future recurrent architectures
```
LTC 当前主要比较：

```text
MLP
 |
 +-- circuit LTC
 |
 +-- residual LTC
 |
 +-- residual LTC + action history
 |
 +-- future recurrent LTC-SAC
```
