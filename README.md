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
PYTHONPATH=. python3 main.py --config configs/go2/sac_baseline.yaml
```

## 配置

| 配置 | 说明 |
|---|---|
| `configs/go2/sac_baseline.yaml` | SAC 基准参数 |
| `configs/go2/ppo_baseline.yaml` | PPO 基准参数 |
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

## 文档

- `docs/isaac_lab_go2_roadmap.md` — Isaac Lab 迁移路线图
- `docs/report_go2_sac_ppo_baseline.md` — SAC vs PPO 学术报告
