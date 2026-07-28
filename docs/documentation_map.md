# Documentation Map

本仓库用于研究 LunarLanderContinuous-v3 上的 SAC/PPO、LTC 时序特征以及函数逼近结构改进。

## 推荐阅读顺序

### 1. 快速理解项目

- `README.md`
  - 项目定位
  - 运行入口
  - 当前实验分支
  - 快速复现实验

### 2. 代码结构

- `docs/architecture.md`
  - `main.py -> config -> training -> variant` 流程
  - YAML 配置约定
  - 模块职责

### 3. SAC 实验

- `docs/parallel_sac_training.md`
  - VecEnv 并行采样
  - transition 计数
  - replay buffer 与更新比例
  - SAC 公平比较协议

- `docs/sac_implementations.md`
  - SAC 框架选择记录

### 4. PPO 对照实验

- `docs/parallel_ppo_training.md`
  - 多环境 rollout
  - GAE
  - minibatch 更新逻辑
  - PPO 强基线设置

### 5. 表示结构研究

- `docs/research_roadmap.md`
  - LTC 当前定位
  - fixed-window LTC
  - residual/action history
  - recurrent LTC-SAC 后续路线

- `docs/rbf_sac.md`
  - RBF 替代 SAC actor/Q 网络实验
  - 参数量匹配原则
  - 多 seed 评价方法

- `docs/rbf_ppo.md`
  - RBF-PPO 对照实验

## 实验原则

所有结构改进均遵循：

1. 保持 SAC/PPO 算法主体不随意改变；
2. 使用同预算比较；
3. 记录参数量、吞吐、reward 和稳定性；
4. 单次曲线不作为结构有效性的充分证据；
5. 重要结论需要多 seed 验证。

## 当前研究主线

```text
MLP SAC baseline
        |
        +-- circuit LTC feature extractor
        |
        +-- residual LTC
        |
        +-- action-history LTC
        |
        +-- recurrent LTC-SAC (future)

parallel experiments
        |
        +-- RBF policy/value approximation
```
