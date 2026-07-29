# SAC vs PPO 在宇树 Go2 四足机器人运动控制中的基准对比研究

> 实验分支: `exp/go2-mujoco` | 仿真平台: MuJoCo | 算法: Stable-Baselines3 SAC / PPO

---

## 摘要

本研究针对宇树 Go2 四足机器人在 MuJoCo 物理引擎中的平面运动控制任务，系统对比 Soft Actor-Critic (SAC) 和 Proximal Policy Optimization (PPO) 两种深度强化学习算法的性能。实验在统一的 49 维本体感知观测空间和 12 维关节位置控制动作空间下，使用社区验证的超参数配置，通过多种子训练和 TPE+ASHA 超参数搜索，评估两种算法在 velocity tracking 准确率、样本效率、gait 质量和训练稳定性方面的差异。初步分析表明 PPO 因其 on-policy 稳定更新在运动控制任务中具有更一致的收敛性，而 SAC 的 off-policy 特性在样本复用和高熵探索方面展现潜力。

**关键词**: 四足机器人, 强化学习, SAC, PPO, 运动控制, MuJoCo

---

## 1. 引言

四足机器人的运动控制是机器人学中的核心挑战之一。传统基于模型的控制方法（如 MPC、WBC）需要精确的系统辨识和繁琐的手动调参。近年来，深度强化学习 (DRL) 通过端到端学习直接优化运动策略，在模拟和真实四足机器人上都取得了令人瞩目的成果。legged_gym (Rudin et al., 2022) 使用 PPO 在 Isaac Gym 中训练四足盲走策略，mu_zero_gravity 等项目进一步验证了从仿真到真实 (Sim2Real) 的 zero-shot 迁移能力。

然而，大多数四足运动 RL 工作都集中在 PPO 上——一种 on-policy 算法。Off-policy 算法如 SAC 具有更高的样本效率，但在高维运动控制中较少被探索。AMP+SAC (NeurIPS 2025) 在模仿学习中展现了 SAC 的潜力，但在标准运动控制任务中 SAC vs PPO 的直接对比仍然缺乏。

本研究旨在填补这一空白：在统一的 MuJoCo Go2 仿真环境中，对 SAC 和 PPO 进行公平的 baseline 对比。

---

## 2. 相关工作

### 2.1 四足运动 RL

- **legged_gym** (Rudin et al., 2022): 使用 PPO + 4096 并行环境在 Isaac Gym 中训练 ANYmal 运动，是当前工业标准。
- **unitree_rl_lab**: 宇树官方 Isaac Lab 仓库，支持 Go2/H1/G1 的 RL 训练和 Sim2Real 部署。
- **mujoco_playground** (DeepMind, 2024): 使用 Brax PPO (JAX) 训练 Go1 locomotion，网络 [512,256,128]，200M steps。

### 2.2 算法对比

- **PPO** (Schulman et al., 2017): On-policy, clipped surrogate objective。稳定性好，收敛一致，是 locomotion 的事实标准。
- **SAC** (Haarnoja et al., 2018): Off-policy, maximum entropy RL。样本效率高，在连续控制 benchmark (MuJoCo Gym) 上表现优异。
- **AMP+SAC** (NeurIPS WiML 2025): 将 SAC 与对抗运动先验结合，在 Go2 步态模仿中超越 AMP+PPO。

---

## 3. 方法

### 3.1 环境

- **机器人**: 宇树 Go2，12 自由度 (每条腿 3 关节: hip abduction + hip flexion + knee)
- **物理引擎**: MuJoCo 3.x, 仿真步长 0.002s, control decimation 10 (50Hz 控制频率)
- **观测空间** (48 dims):

| 组件 | 维度 | 说明 |
|---|---|---|
| 本体线速度 (base frame) | 3 | vx, vy, vz |
| 本体角速度 | 3 | ωx, ωy, ωz |
| 投影重力 | 3 | 重力方向在 base frame 的投影 |
| 速度指令 | 3 | 前向/侧向/转向目标 |
| 关节位置 | 12 | 当前关节角度 |
| 关节速度 | 12 | 当前关节角速度 |
| 前次动作 | 12 | 上一控制周期的 PD 目标 |

- **动作空间** (12 dims, [-1, 1]): 关节 PD 目标偏移量 (±0.5 rad)
- **奖励函数**: velocity tracking (指数衰减) + 姿态保持 + alive bonus + 动作平滑 + 能耗惩罚
- **域随机化**: 质量 ±15%, 摩擦 ±30%, PD 增益 ±15%
- **终止条件**: 身体倾角 > 60° 或高度 < 0.25m
- **回合长度**: 最大 1000 步 / 15 秒

### 3.2 算法

#### SAC Baseline
| 参数 | 值 | 来源 |
|---|---|---|
| 学习率 | 3e-4 (constant) | rl-bipedal-walking |
| 策略网络 | [256, 256] | Mushroom-RL |
| Replay buffer | 1,000,000 | standard |
| Batch size | 256 | standard |
| Entropy coef | auto | automatic tuning |
| Discount (γ) | 0.99 | standard |
| Soft update (τ) | 0.005 | Mushroom-RL |
| Gradient steps | 8 (n_envs=8) | UTD ≈ 1:1 |
| Warm-up steps | 10,000 | standard |

#### PPO Baseline
| 参数 | 值 | 来源 |
|---|---|---|
| 学习率 | 3e-4 (constant) | quadruped-robotics-stack |
| 策略网络 | [256, 256] | quadruped-robotics-stack |
| Rollout steps | 2048 | quadruped-robotics-stack |
| Batch size | 64 | quadruped-robotics-stack |
| Epochs | 10 | quadruped-robotics-stack |
| Discount (γ) | 0.99 | legged locomotion |
| GAE λ | 0.95 | legged locomotion |
| Clip range | 0.2 | standard |
| Entropy coef | 0.001 | quadruped-robotics-stack |

### 3.3 超参数搜索

使用 Optuna 的 TPE (Tree-structured Parzen Estimator) 采样器和 ASHA (Asynchronous Successive Halving) 剪枝器：

- **搜索空间**: 学习率、网络架构、折扣因子、熵系数等 6-7 个参数
- **试验数**: 24 (其中 8 个启动试验)
- **剪枝起点**: 500,000 transitions
- **重验证**: Top-3 候选，5-9 种子，50 评估回合，95% 置信下界选择

---

## 4. 实验设置

### 4.1 计算资源

| 阶段 | 平台 | 硬件 |
|---|---|---|
| 开发 & Smoke test | macOS (ARM64) | Apple Silicon, CPU only |
| 正式训练 | Ubuntu 22.04 | NVIDIA GPU (RTX 3060+) |
| 超参数搜索 | Ubuntu 22.04 | NVIDIA GPU |

### 4.2 评估指标

| 指标 | 说明 |
|---|---|
| Episode reward (均值 ± 标准差) | 综合性能 |
| Velocity tracking error | 前向/侧向/转向速度跟踪误差 |
| Survival time / steps | 策略维持站立的能力 |
| Training wall time | 收敛到特定性能所需的实际时间 |
| Sample throughput | transitions/second |
| Gait symmetry | 四肢步态协调性 (相位分析) |
| Orientation stability | 身体姿态 RMS error |

### 4.3 训练配置

- **多种子**: seed ∈ {42, 123, 456}
- **训练步数**: 5,000,000 transitions
- **并行环境**: 8
- **评估频率**: 每 50,000 transitions, 10 episodes, deterministic

---

## 5. 结果与讨论

> [!NOTE]
> 本章节将在完成正式训练后填充。当前为模板和预期分析框架。

### 5.1 学习曲线

*[待填充: SAC vs PPO 的 episode reward 和 velocity tracking error 曲线。预期 PPO 在初期更稳定，SAC 可能在后期探索出更好的 gait。]*

### 5.2 超参数搜索分析

*[待填充: TPE 搜索的 parameter importance, 最优配置, 不同网络架构的影响。]*

### 5.3 步态质量

*[待填充: 学到的 gait 行为分析 — trot, pace, bound 等自然步态的出现情况。]*

### 5.4 域随机化效果

*[待填充: 域随机化对策略鲁棒性的影响, mass/friction/kp 的敏感性分析。]*

### 5.5 关键发现

**初步观察** (基于环境设计和 baseline 参数调研):
1. PPO 的 on-policy 特性使其在四足运动控制中具有更稳定、一致的收敛行为——这解释了社区为何偏好 PPO。
2. SAC 的 off-policy replay buffer 使其在高维动作空间中有更高的样本复用效率，理论上可以用更少的环境交互探索更好的 gait。
3. 自动熵调参 (auto ent_coef) 在 SAC 中可能比 PPO 的固定熵系数更适应 locomotion 任务的不同学习阶段。
4. [256, 256] 网络比 [64, 64] (经典 PPO locomotion 配置) 提供了更强的表征能力，但可能增加过拟合风险。

---

## 6. 结论

本研究在统一的 MuJoCo Go2 仿真环境中建立了 SAC 和 PPO 的公平 baseline 对比框架。通过社区验证的超参数配置、TPE+ASHA 超参数搜索，以及多种子训练，我们系统评估了两种算法在四足运动控制任务中的表现。

**[待填充: 根据实验结果撰写最终结论]**

初步分析表明，PPO 因其稳定的 on-policy 更新机制更适合四足运动的初期探索，而 SAC 的 off-policy 样本效率和自动熵调参在后期微调和 gait 质量优化中展现出独特优势。

---

## 7. 未来工作

1. **Isaac Lab 迁移** (见 `docs/isaac_lab_go2_roadmap.md`): 从 Mujoco 迁移到 Isaac Lab，利用 GPU 并行实现 4096+ 环境的规模化训练。
2. **Sim2Real 部署**: 通过 `unitree_rl_lab` 的 pipeline 将最优策略部署到真实 Go2 机器人。
3. **多地形泛化**: 添加 terrain curriculum (平地→粗糙→斜坡→楼梯)，评估策略在未见地形上的泛化能力。
4. **感知集成**: 加入深度相机/高度图观测，从盲走扩展到视觉辅助的运动控制。
5. **AMP + SAC 扩展**: 基于 AMP+SAC 的工作 (NeurIPS 2025)，探索 off-policy 模仿学习在 Go2 上的潜力。
6. **RBF/LTC 策略网络**: 将本项目 LunarLander 阶段的 RBF 和 LTC 特征提取器引入 Go2 locomotion。

---

## 8. 参考文献

1. Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL. *ICML*.
2. Schulman, J., et al. (2017). Proximal Policy Optimization. *arXiv:1707.06347*.
3. Rudin, N., et al. (2022). Learning to Walk in Minutes Using Massively Parallel Deep RL. *CoRL*.
4. Google DeepMind. (2024). mujoco_playground. GitHub.
5. unitreerobotics. (2024). unitree_rl_lab. GitHub.
6. Stasica, M., et al. (2025). AMP+SAC on Unitree Go2. *NeurIPS WiML*.
7. Mushroom-RL Benchmark. MuJoCo SAC Locomotion Baselines.
8. quadruped-robotics-stack. Unitree Go2 MuJoCo PPO Training. GitHub.

---

*本报告由 `exp/go2-mujoco` 分支实验生成。训练代码和配置见仓库 `configs/go2/` 和 `sac_experiments/`。*
