# Go2 Locomotion — SAC / PPO Reinforcement Learning

Unitree Go2 四足机器人 MuJoCo 运动控制 RL 训练框架。

## 环境

- `Go2Locomotion-v0` — 12-DoF Go2 平面运动控制
- 物理引擎: MuJoCo 3.x
- 算法: Stable-Baselines3 SAC / PPO + Optuna 超参搜索

## 快速开始

```bash
conda activate cybernetic_env
pip install -r requirements-sac-demo.txt
bash scripts/setup_go2_assets.sh          # 下载 Go2 MuJoCo 模型
PYTHONPATH=. python3 main.py --config configs/go2/sac_baseline.yaml
```

## 配置

| 配置 | 说明 |
|---|---|
| `configs/go2/sac_baseline.yaml` | SAC 基准参数 |
| `configs/go2/ppo_baseline.yaml` | PPO 基准参数 |
| `configs/go2/ppo_32env.yaml` | 32 环境、完整命令空间与域随机化的 PPO 训练配置 |
| `configs/go2/ppo_stabilized_forward.yaml` | 有界 Tanh PPO、固定前向指令的 3-seed 稳定化验证 |
| `configs/search/go2_sac_tpe.yaml` | SAC TPE 超参搜索 |
| `configs/search/go2_ppo_tpe.yaml` | PPO TPE 超参搜索 |
| `configs/smoke/go2_sac.yaml` | SAC 冒烟测试 (Mac CPU) |
| `configs/smoke/go2_ppo.yaml` | PPO 冒烟测试 (Mac CPU) |

## 可视化

```bash
# 交互式 viewer (Mac 需 mjpython)
mjpython visualize_go2.py

# 渲染模型步态视频
python3 render_go2_video.py

# 可视化训练好的 policy
mjpython render_trained_policy.py outputs/.../best_model.zip --mode viewer
python3 render_trained_policy.py outputs/.../best_model.zip --mode video
```

## 项目结构

```
sac_experiments/          # 核心框架
  go2_env.py              #   Go2 gymnasium 环境
  env_utils.py            #   通用环境工具
  config.py               #   YAML 配置系统
  training.py             #   训练编排
  model_factory.py        #   模型构建
  policies.py             #   有界 Tanh-Gaussian PPO 策略
  reporting.py            #   实验报告
  search/                 #   超参搜索 (Optuna TPE + ASHA)
configs/go2/              # Go2 实验配置
configs/search/           # 超参搜索配置
configs/smoke/            # 冒烟测试配置
assets/unitree_go2/       # Go2 MuJoCo 模型 (XML + meshes)
docs/                     # 文档
```

## 平台

- **开发**: macOS (ARM64), CPU — smoke test & 可视化
- **训练**: Ubuntu + NVIDIA GPU — 正式训练 & 超参搜索

## PPO 稳定化验证

```bash
# Ubuntu：推荐用脚本顺序执行 gate、benchmark、三个种子和产物审计
bash scripts/run_go2_ppo_stabilization.sh

# Mac：只验证完整训练/评估/保存链路
PYTHONPATH=. python3 main.py --config configs/smoke/go2_ppo.yaml

# Ubuntu：先选取 8/16/32 envs 与 CPU/CUDA policy 的匹配运行配置
PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_stabilized_forward.yaml \
  --benchmark-runtime

# Ubuntu：三个种子顺序执行，每个约 2M transitions
# profile 仅允许在生成它的同一主机、同一份 YAML 上使用
PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_stabilized_forward.yaml \
  --runtime-profile outputs/go2_runtime_benchmark.json \
  --seeds 0 1 2

# 训练后确认分析数据没有缺失或被 .gitignore 隐藏
python3 scripts/check_training_artifacts.py \
  outputs/go2_ppo_stabilized_forward/seed_0 \
  runs/go2_ppo_stabilized_forward/seed_0

# 三个种子完成后生成统一验收报告
python3 scripts/summarize_go2_multiseed.py \
  outputs/go2_ppo_stabilized_forward
```

脚本支持中断后重新运行：已完成且产物完整的阶段会被跳过，不完整目录
不会被自动覆盖。默认复用已有 runtime profile；只有确认需要重新测试
同一台机器时才使用 `bash scripts/run_go2_ppo_stabilization.sh --rebenchmark`。

## 文档

- **[Go2 MuJoCo PPO 环境与训练改进说明](docs/go2_mujoco_ppo_improvements.md)** — 环境正确性修复、奖励与观测设计、PPO rollout/KL 机制、完整域随机化、100M 训练设置，以及 `n_steps=64` 等争议问题的详细解释与验证方法
- **[Go2 PPO 稳定化实施与验收](docs/go2_ppo_stabilization.md)** — 补充日志结论、有界策略、固定评估、轻量硬件 benchmark、Git 产物策略与 3-seed 验收门槛
- [Isaac Lab 迁移路线图](docs/isaac_lab_go2_roadmap.md)
- [SAC vs PPO 学术报告](docs/report_go2_sac_ppo_baseline.md)
