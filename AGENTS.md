# AGENTS.md — Go2 Locomotion RL

面向 AI coding agent 的项目指南。

## 项目概述

Unitree Go2 四足机器人 MuJoCo 运动控制 RL 训练框架。核心代码在 `sac_experiments/` 下。

## 关键文件

```
main.py                           # CLI 入口 (--config / --search-config)
sac_experiments/config.py         # YAML 配置加载与验证
sac_experiments/training.py       # 训练编排
sac_experiments/model_factory.py  # SB3 模型构建
sac_experiments/go2_env.py        # Go2Locomotion-v0 环境定义
sac_experiments/env_utils.py      # 通用环境工具 (evaluate, make_vec_env)
sac_experiments/reporting.py      # JSON 实验报告
sac_experiments/search/           # 超参搜索子包
```

## 环境

- **conda env**: `cybernetic_env`
- **运行需设 PYTHONPATH**: `PYTHONPATH=. python3 main.py ...`
- **Mac viewer 需 mjpython**: `mjpython visualize_go2.py`

## 配置约定

- YAML 实验配置需包含 `experiment`, `environment`, `training`, `evaluation`, `output`, `sac`/`ppo` 段落
- SAC: `gradient_steps` 应与 `n_envs` 匹配，保持 UTD ≈ 1:1
- PPO: `timesteps` 必须能被 `n_envs * n_steps` 整除
- 只支持 `Go2Locomotion-v0` 环境

## 提交规范

使用简洁的命令式 commit message，例如 `Add Go2 SAC baseline config`。PR 应描述实验变更、列出运行命令、提及生成的 artifacts。

## 平台注意

- Mac (ARM64): 开发 & smoke test，用 `device: cpu`, `allow_cpu: true`
- Ubuntu + NVIDIA: 正式训练，用 `device: cuda`
