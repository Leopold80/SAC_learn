# Go2 PPO 稳定化实施与验收

## 已确认的训练问题

远端提交 `0e131b9` 补齐了 fixed 与 v2 的 TensorBoard、评估 NPZ、
Monitor CSV 和 v2 best 权重。完整数据确认：

- fixed 的 `train/std` 从约 `1.00` 增长到 `440.64`，v2 增长到
  `257.57`；两者均使用无界高斯和 `ent_coef=0.01`。
- fixed/v2 best 的确定性动作元素饱和率分别约为 `99.982%` 和
  `99.974%`。环境实际接收到的控制接近 bang-bang `±1`。
- fixed 的真实最佳 20-episode 均值为 `1043.25`，v2 为 `748.67`；
  原报告的 `1466.82/1351.12` 是每次评估第一个 episode 的最大值。
- v2 有真实学习趋势，但平均前向速度仅约 `0.127 m/s`，而平均指令
  为 `0.582 m/s`。它改善了部分横向/偏航指标，没有解决前向跟踪。
- fixed 主要学会了存活；v2 的完整 episode 比例明显更低。raw reward
  因奖励定义不同，不能直接横比。

因此首先修复动作概率语义和评估协议，而不是继续增加到 100M。

## 本分支的设计

### 有界动作策略

`TanhActorCriticPolicy` 使用 SB3 的
`SquashedDiagGaussianDistribution`：

- rollout buffer 保存 tanh 后动作；
- PPO 对相同的有界动作计算带 Jacobian 修正的 log-prob；
- `log_std_init=-1`，有效范围 `[-5, 0]`；
- 第一阶段 `ent_coef=0`，避免再次用熵项持续放大 std；
- TensorBoard 额外记录 effective std、Gaussian mean 和确定性动作
  饱和率。

旧权重不能 resume 到新策略，因为动作分布与旧 log-prob 语义不同。

### 奖励

线速度和偏航采用相对站立基线的平滑指数差：

```text
1.5 * [exp(-||v_xy-command_xy||² / 0.25)
       - exp(-||command_xy||² / 0.25)]

0.75 * [exp(-(yaw-command_yaw)² / 0.25)
        - exp(-command_yaw² / 0.25)]
```

站立不动时 tracking reward 为零，不再除以接近零的指令误差。
第一阶段 `alive_weight=0`，保留原有姿态、垂向速度、动作变化和扭矩
惩罚。奖励结构参考
[Isaac Lab Go2 配置](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/go2/rough_env_cfg.py)，
但没有照搬不同控制尺度下的惩罚系数。

### 固定评估

- checkpoint command 固定为 `(0.5, 0, 0)`；
- 关闭域随机化；
- 每次评估重放 seeds `10000–10009`；
- 训练后 holdout 使用 seeds `20000–20019`；
- NPZ 同时保存回报、长度、速度 RMSE、平均速度、动作饱和、扭矩、
  reward components 和终止原因；
- `best_eval_reward()` 使用每个 checkpoint 的 episode 均值。

## 4070 Laptop + i7-13650HX 运行

正式训练前运行：

```bash
PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_stabilized_forward.yaml \
  --benchmark-runtime
```

benchmark 在相同 2048-transition rollout 下比较：

| envs | n_steps |
|---:|---:|
| 8 | 256 |
| 16 | 128 |
| 32 | 64 |

每组分别测试 CPU/CUDA policy，记录吞吐、进程树 RSS、swap、显存、
GPU 利用率和温度。候选必须满足：

- RSS 不超过 12 GiB；
- 新增 swap 不超过 512 MiB；
- 显存不超过 6 GiB。

吞吐差距小于 10% 时优先选择内存和进程数更少的组合。三个种子必须
顺序运行，不能并发。`n_envs/n_steps` 会改变并行度和单条轨迹长度，
因此它们属于实验设计参数，而不只是部署参数：benchmark 报告会绑定
YAML SHA-256、主机与 Python/CUDA 信息；`--runtime-profile` 只接受同一
主机、同一 YAML 生成且通过资源门槛的候选，并在应用后重新检查 rollout、
batch、评估频率和总步数约束。

## 正式短训练与验收

推荐直接顺序执行完整流程：

```bash
bash scripts/run_go2_ppo_stabilization.sh
```

脚本会依次运行两个 gate、资源 benchmark、seed 0/1/2、逐阶段产物审计
和三种子汇总。中断后可直接重跑：完整阶段会被审计后跳过，不完整目录
会停止并等待人工检查，绝不会自动删除或覆盖。已有 runtime profile 默认
复用，避免续跑时改变 `n_envs/n_steps/device`；只有明确需要重新 benchmark
时使用 `--rebenchmark`。

先运行两个 524k 单种子 gate：

```bash
PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_stabilized_gate_relative.yaml
PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_stabilized_gate_smooth.yaml
```

前者只作为 v2 奖励对照；正式候选采用平滑奖励。任一 gate 出现 NaN、
持续超过 50% 的动作饱和或存活完全坍塌时，应先修实现，不继续 2M。

```bash
PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_stabilized_forward.yaml \
  --runtime-profile outputs/go2_runtime_benchmark.json \
  --seeds 0 1 2
```

每个种子训练 `2,000,896` transitions。至少 2/3 个种子的 holdout
需要满足：

- 20 回合跌倒不超过 1 次；
- `vx RMSE <= 0.20 m/s`；
- mean vx 位于 `[0.35, 0.65] m/s`；
- `mean |vy| <= 0.05 m/s`；
- `mean |yaw_rate| <= 0.10 rad/s`；
- 最后 25% updates 的动作饱和率低于 10%；
- median clip fraction 不高于 0.25；
- approx-KL 95 分位不高于 0.05。

生成三种子物理指标汇总：

```bash
python3 scripts/summarize_go2_multiseed.py \
  outputs/go2_ppo_stabilized_forward
```

未通过时不得直接扩大到 100M。先根据日志区分动作分布、奖励尺度、
观测尺度和稳定性问题。

## Git 训练产物策略

默认上传：

- `experiment_summary.json`、`eval_summary.json`
- `evaluations.npz`
- train/eval Monitor CSV
- TensorBoard event
- 分析图和报告
- best/final 权重

默认忽略：

- 周期 checkpoint
- replay buffer
- cache/tmp
- 视频和非分析 GIF

训练结束后必须运行：

```bash
python3 scripts/check_training_artifacts.py OUTPUT_DIR TENSORBOARD_DIR
```

该检查会在必需文件缺失、被 Git 忽略或单文件超过默认 50 MiB 时失败。
