# Isaac Lab Migration Roadmap — Unitree Go2

## 背景

当前项目使用 Mujoco + Stable-Baselines3 训练 Go2 运动控制。Mujoco 适合快速原型开发（Mac CPU 可跑），但在训练吞吐量和多地形泛化方面有局限。Isaac Lab 是 NVIDIA 官方推出的机器人 RL 训练框架，提供 GPU 并行 + 地形课程学习 + 域随机化的完整解决方案。

## Isaac Lab 的优势

| 维度 | Mujoco (当前) | Isaac Lab (迁移目标) |
|---|---|---|
| 并行环境数 | 8-64 (CPU) | 4096+ (GPU) |
| 训练吞吐 | ~1k steps/s | ~100k steps/s (Brax PPO) |
| 多地形 | 需自建 | 内置 terrain curriculum |
| 传感器 | 自定义 | 内置深度/IMU/接触 |
| Sim2Real | 需自建 | 官方 pipeline (unitree_rl_lab) |
| 硬件要求 | CPU/GPU 均可 | NVIDIA GPU (RTX 3060+) |

## 关键参考项目

### 1. [unitreerobotics/unitree_rl_lab](https://github.com/unitreerobotics/unitree_rl_lab)
宇树官方 Isaac Lab 仓库，支持 Go2/H1/G1，提供完整的训练→Sim2Sim→Sim2Real pipeline。

### 2. [google-deepmind/mujoco_playground](https://github.com/google-deepmind/mujoco_playground)
Google DeepMind 的 Mujoco 训练框架，使用 Brax PPO (JAX)。Go1 locomotion 任务的参考参数：

| 参数 | 值 |
|---|---|
| Learning rate | 5e-4 |
| Discount (γ) | 0.97 |
| Entropy coef | 5e-3 |
| Policy network | [512, 256, 128] |
| Training steps | 200M |
| 并行环境 | 1024 |
| Max grad norm | 1.0 |
| Reward scaling | 0.1 |

### 3. [legged_gym](https://github.com/leggedrobotics/legged_gym)
Isaac Gym 时代的四足运动训练标准，许多参数直接沿用到 Isaac Lab。

## 迁移检查清单

### 阶段 1: 环境移植
- [ ] 在 Ubuntu+NVIDIA 机器上安装 Isaac Lab
- [ ] 用 `unitree_rl_lab` 的 Go2 URDF 替换当前的 mujoco_menagerie MJCF
- [ ] 对齐观测/动作空间 (49-dim obs, 12-dim action)
- [ ] 对齐奖励函数设计

### 阶段 2: 算法适配
- [ ] Isaac Lab 默认使用 RSL-RL (PPO 实现)，需要适配或替换为 SB3
- [ ] 或使用 Isaac Lab 原生的 PPO 实现，将当前 SB3 baseline 参数迁移
- [ ] SAC 在 Isaac Lab 的 GPU 并行环境中需要 Off-Policy 适配 (replay buffer 设计)

### 阶段 3: 多地形训练
- [ ] 启用 Isaac Lab 的地形生成器 (flat, rough, stairs, slopes)
- [ ] 配置 terrain curriculum: 从平坦逐步过渡到崎岖
- [ ] 添加域随机化: 质量、摩擦、电机增益、地形参数

### 阶段 4: Sim2Real
- [ ] 用 `unitree_rl_lab` 的 deploy 工具将 policy 部署到 Mujoco (Sim2Sim 验证)
- [ ] 通过 ROS2 部署到真实 Go2 机器人
- [ ] 记录并分析 Sim2Real gap

## Isaac Lab PPO Baseline (参考值)

| 参数 | 值 | 来源 |
|---|---|---|
| Learning rate | 1e-3 | unitree_rl_lab |
| Discount (γ) | 0.99 | standard |
| GAE λ | 0.95 | standard |
| Clip range | 0.2 | standard |
| Entropy coef | 1e-3 → 1e-5 (decay) | legged_gym |
| Policy network | [512, 256, 128] | mujoco_playground |
| 训练步数 | 20k-100k iterations | 取决于任务 |
| 环境数 | 4096 | GPU 批量 |
| 域随机化 | 质量 ±20%, 摩擦 ±50%, 地形 ±5cm | standard |

## 预期训练时间

| 配置 | 当前 (Mujoco + SB3) | Isaac Lab + RSL-RL |
|---|---|---|
| 5M steps, 8 envs | ~1-2 hours (GPU) | N/A |
| 200M steps, 4096 envs | N/A | ~4-8 hours (RTX 4090) |

## 参考资料

- [Isaac Lab 官方文档](https://isaac-sim.github.io/IsaacLab/)
- [unitree_rl_lab GitHub](https://github.com/unitreerobotics/unitree_rl_lab)
- [mujoco_playground GitHub](https://github.com/google-deepmind/mujoco_playground)
- [legged_gym GitHub](https://github.com/leggedrobotics/legged_gym)
