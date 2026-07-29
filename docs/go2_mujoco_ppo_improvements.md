# Go2 MuJoCo PPO 环境与训练改进说明

本文档记录 `exp/go2-mujoco` 分支中 Unitree Go2 MuJoCo PPO 实验的环境修复、训练配置调整、设计理由，以及讨论过程中出现的争议和疑问。

本文档并不宣称当前参数已经最优。全文始终区分三类结论：

1. **环境正确性修复**：原实现存在明确的语义或代码问题，必须修复；
2. **有公开四足基线支持的工程选择**：方向有依据，但数值不能机械照搬；
3. **仍需本项目实验验证的超参数**：只能作为起点，不能写成定论。

---

## 目录

- [1. 当前实验目标与配置](#1-当前实验目标与配置)
- [2. 已确认的环境正确性问题](#2-已确认的环境正确性问题)
- [3. 动作与观测改进](#3-动作与观测改进)
- [4. 奖励函数改进](#4-奖励函数改进)
- [5. PPO 训练结构改进](#5-ppo-训练结构改进)
- [6. `target_kl`：PPO-Clip 之外的更新保险](#6-target_klppo-clip-之外的更新保险)
- [7. 为什么保留完整任务和域随机化](#7-为什么保留完整任务和域随机化)
- [8. 争议与疑问详解](#8-争议与疑问详解)
- [9. 环境诊断与训练监控](#9-环境诊断与训练监控)
- [10. 验证边界与消融顺序](#10-验证边界与消融顺序)
- [11. 参考实现与迁移原则](#11-参考实现与迁移原则)
- [12. 推荐运行流程](#12-推荐运行流程)

---

## 1. 当前实验目标与配置

当前任务保留完整的平面速度指令跟踪，而不是仅训练固定前向速度：

- 前向速度指令：`[0.0, 1.0] m/s`；
- 横向速度指令：`[-0.3, 0.3] m/s`；
- 偏航角速度指令：`[-0.5, 0.5] rad/s`；
- 质量随机化：`±15%`；
- 摩擦随机化：`±30%`；
- PD 增益随机化：`±15%`；
- 并行环境数：32；
- 策略控制频率：50 Hz；
- episode 上限：1000 个策略步，即约 20 s；
- 训练量：约 100M transitions。

当前 PPO 的 rollout size 为：

```text
32 environments × 64 steps = 2048 transitions
```

配置验证器要求总训练步数能够被 rollout size 整除，因此使用：

```yaml
training:
  timesteps: 100_001_792
```

计算如下：

```text
100,001,792 / 2,048 = 48,829 rollouts
```

所以这里的 “100M” 是指所有并行环境累计产生约一亿条 transition，而不是每个环境各自运行一亿步。

当前核心 PPO 配置为：

```yaml
ppo:
  learning_rate: 0.0003
  learning_rate_schedule: constant
  policy_net_arch: [256, 256]

  n_steps: 64
  batch_size: 512
  n_epochs: 5

  gamma: 0.99
  gae_lambda: 0.95
  clip_range: 0.2
  normalize_advantage: true

  ent_coef: 0.01
  vf_coef: 1.0
  max_grad_norm: 1.0
  target_kl: 0.01
```

这些数值不是一个不可质疑的“标准答案”。其中 `n_steps`、熵系数、价值损失系数、网络规模和奖励权重都需要用本项目自己的训练结果验证。

---

## 2. 已确认的环境正确性问题

以下内容不是风格偏好，而是会改变 MDP、奖励语义或物理参数分布的明确问题。

### 2.1 `frame_skip=10` 原本没有真正执行

环境设置了：

```python
frame_skip = 10
```

MuJoCo 模型的 physics timestep 为：

```text
0.002 s
```

因此一个 policy step 应持续：

```text
10 × 0.002 s = 0.02 s
```

即：

```text
policy frequency = 1 / 0.02 = 50 Hz
```

原实现虽然声明 `frame_skip=10`，但在 `step()` 中只执行一次物理积分，相当于：

```text
1 × 0.002 s = 0.002 s
```

实际策略频率因此接近 500 Hz。

这不是单纯“频率高一点”的问题。它会同时改变：

- 每个动作在物理系统中保持的时间；
- PD 控制器的实际闭环行为；
- 动作平滑惩罚对应的真实时间尺度；
- episode 对应的物理时长；
- 折扣因子对应的真实时间范围；
- 策略探索时动作变化对机体产生的实际效果。

修复后使用：

```python
self.do_simulation(ctrl, self.frame_skip)
```

每次策略动作真正推进 10 个物理子步。

### 2.2 episode 时间语义统一

修复 control decimation 后：

```text
1000 policy steps × 0.02 s = 20 s
```

因此 Gymnasium 注册中的 `max_episode_steps=1000` 对应 20 秒 episode。

环境内部原有的 15 秒截断与外部 `TimeLimit` 重复，而且二者对 termination/truncation 的语义可能不同。现在由 `TimeLimit` 统一处理正常超时：

- `terminated=True`：摔倒、姿态失稳、数值非法等任务终止；
- `truncated=True`：达到时间上限，但状态本身并非失败。

这个区分会影响 critic bootstrap。正常时间截断通常不应被当作失败状态强行把未来价值置零。

### 2.3 域随机化原本存在累乘漂移

原实现每次 reset 类似执行：

```python
model.body_mass *= random_scale
model.geom_friction *= random_scale
```

假设第一次质量缩放为 `1.1`，第二次为 `0.9`，第二次得到的不是标称质量的 `0.9` 倍，而是：

```text
1.1 × 0.9 = 0.99
```

重复大量 reset 后，参数会形成乘法随机游走：

\[
m_k
=
m_0\prod_{i=1}^{k}\alpha_i.
\]

即使每个 \(\alpha_i\) 都在规定范围内，累乘结果也可能逐渐远离标称区间。

现在初始化时保存：

```python
_nominal_body_mass
_nominal_body_inertia
_nominal_geom_friction
```

每次 reset 都重新计算：

```python
current_mass = nominal_mass * sampled_mass_scale
current_inertia = nominal_inertia * sampled_mass_scale
current_friction = nominal_friction * sampled_friction_scale
```

所以随机化仍然开启，但不会跨 episode 累积。

### 2.4 观测与奖励的速度坐标系原本不一致

观测中的 base linear velocity 已经转换到了机体坐标系，但奖励原本直接读取世界坐标系速度。

考虑机器人偏航 90°：

- 机器人自身前向方向对应世界坐标系 y 轴；
- 此时世界系 \(v_x\) 不再代表机器人前向速度；
- 若 command 的 \(v_x^{\mathrm{cmd}}\) 被解释为机体前向速度，奖励却拿它和世界系 \(v_x\) 比较，就发生了语义冲突。

现在统一为：

\[
v^{\mathrm{body}}
=
R_{WB}^{\mathsf T}v^{\mathrm{world}}.
\]

然后：

- observation 使用机体系线速度；
- command 中的 x/y 分量表示机体前向/横向速度；
- tracking reward 同样使用机体系速度。

MuJoCo free joint 的角速度在当前接口语义下已经是局部机体系表达，因此不重复旋转。

### 2.5 “动作平滑”原本实际是动作幅值惩罚

原奖励使用：

\[
\lVert a_t\rVert^2.
\]

它鼓励动作接近零，但不能直接衡量动作是否平滑。

真正的 action-rate penalty 是：

\[
\lVert a_t-a_{t-1}\rVert^2.
\]

二者作用不同：

- \(\lVert a_t\rVert^2\)：抑制动作偏离默认姿态；
- \(\lVert a_t-a_{t-1}\rVert^2\)：抑制相邻控制周期的突变。

动作幅值大但变化缓慢，可以很平滑；动作幅值小但每步正负跳变，也可能很不平滑。

此外，奖励计算必须发生在：

```python
previous_action = current_action
```

之前，否则差分恒为零。

---

## 3. 动作与观测改进

### 3.1 动作语义

策略输出：

```text
a_t ∈ [-1, 1]^12
```

动作被解释为默认站姿上的关节位置偏置：

\[
q_{\mathrm{target}}
=
q_{\mathrm{default}}
+
s_a a_t,
\]

其中当前：

\[
s_a=0.25\ \mathrm{rad}.
\]

底层 PD 控制为：

\[
\tau
=
K_p(q_{\mathrm{target}}-q)
-
K_d\dot q.
\]

因此策略并不是直接输出力矩，而是在较低频率上输出关节位置目标偏置。

### 3.2 为什么 action scale 从 0.5 改为 0.25

主要理由：

1. 初始随机高斯策略容易输出较大动作；
2. `0.5 rad` 对某些关节意味着很大的瞬时目标偏移；
3. 目标偏移再乘以较高 \(K_p\)，会产生较大力矩；
4. 早期策略可能主要学到“避免随机动作导致摔倒”，而不是速度跟踪；
5. `0.25 rad` 与公开 Go2 locomotion 配置的常见数量级接近。

但 `0.25` 不是由定理推出的，也不是所有 Go2 模型的唯一正确值。它依赖：

- 默认站姿；
- 关节限位；
- PD 增益；
- 电机力矩范围；
- 模型惯量；
- policy action distribution 的初始标准差。

判断 action scale 是否合适，应观察：

- 动作是否频繁打到 \(\pm1\)；
- 关节目标是否频繁靠近限位；
- actuator control 是否频繁饱和；
- 是否因为动作范围不足而迈不开腿；
- 是否因为范围过大而早期剧烈摔倒。

### 3.3 相对关节位置观测

关节位置输入改为：

\[
q-q_{\mathrm{default}}.
\]

原因是动作本身也是相对默认姿态的偏置。这样 observation 与 action 在同一个局部坐标语义下：

- `0` 表示默认站姿；
- 正负值表示相对站姿的偏移；
- 网络不需要先自行减去一个固定偏置。

这通常提高数值条件，但并不增加新的信息；绝对角度和相对角度在已知默认姿态时是等价表示。

### 3.4 observation scaling

当前使用：

```python
2.0 * local_linear_velocity
0.25 * local_angular_velocity
projected_gravity
commands
joint_position_error
0.05 * joint_velocity
previous_action
```

目的不是改变物理意义，而是使不同通道处于更接近的数量级。

若不缩放，可能出现：

- joint velocity 的数值远大于 gravity；
- 网络前几层更容易被大尺度通道主导；
- 相同学习率对不同输入通道产生不均衡的敏感度。

固定缩放和运行时标准化是两种不同方案：

- 固定缩放：可解释、不会随训练分布漂移；
- `VecNormalize` 等运行统计：适应数据，但训练和评估必须严格保存/加载统计量。

当前先使用固定缩放，避免同时引入额外状态。

### 3.5 为什么改为 `float32`

Stable-Baselines3 和常见神经网络默认使用单精度。环境若返回 `float64`，通常仍会在进入网络前转换成 `float32`。

直接返回 `float32` 可以：

- 减少进程间传输量；
- 避免重复类型转换；
- 与 observation space 声明一致；
- 避免某些 wrapper 对 dtype 发出警告。

MuJoCo 内部物理积分仍可以使用自己的双精度数据；这里改变的是交给策略网络的 observation/action 接口类型。

---

## 4. 奖励函数改进

当前奖励为：

\[
\begin{aligned}
r_t={}&
1.0r_{\mathrm{lin}}
+0.5r_{\mathrm{yaw}}
-2.0v_z^2
-0.05\lVert\omega_{xy}\rVert^2\\
&-\lVert g_{xy}\rVert^2
-0.01\lVert a_t-a_{t-1}\rVert^2
-2\times10^{-5}\lVert\tau\rVert^2
-5I_{\mathrm{terminated}}.
\end{aligned}
\]

### 4.1 线速度跟踪

\[
r_{\mathrm{lin}}
=
\exp\left(
-\frac{
\lVert
v_{xy}^{\mathrm{cmd}}
-
v_{xy}^{\mathrm{body}}
\rVert^2
}{0.25}
\right).
\]

它是稠密奖励：每一个控制步都有反馈，而不是只有走完整段轨迹才给奖励。

指数形式的性质：

- 误差为零时奖励接近 1；
- 误差增大时平滑下降；
- 不需要人为裁剪负无穷惩罚；
- 带宽参数决定奖励对误差的敏感程度。

分母 `0.25` 是工程带宽，不是方差估计或严格概率模型。带宽过小会使大部分早期状态奖励接近零；带宽过大则会使速度误差很大时仍能获得较高奖励。

### 4.2 偏航角速度跟踪

\[
r_{\mathrm{yaw}}
=
\exp\left(
-\frac{
(\omega_z^{\mathrm{cmd}}-\omega_z)^2
}{0.25}
\right).
\]

将平移速度和偏航速度拆开，而不是放在一个混合范数中，优点是：

- 可以分别设置权重；
- 可以分别记录日志；
- 避免不同单位和范围未经解释地混合；
- 能判断策略究竟是不会走，还是不会转。

### 4.3 姿态相关惩罚

竖直速度：

\[
-v_z^2
\]

抑制不必要的弹跳。

横滚/俯仰角速度：

\[
-\lVert\omega_{xy}\rVert^2
\]

抑制机身快速翻滚和俯仰。

projected gravity 水平分量：

\[
-\lVert g_{xy}\rVert^2
\]

反映机体偏离竖直方向的程度。机器人保持水平时，重力在机体系中主要位于 z 方向，水平分量接近零。

### 4.4 action-rate penalty

\[
-\lVert a_t-a_{t-1}\rVert^2
\]

它鼓励时间上连续的动作，但存在权衡：

- 太小：动作可能抖动；
- 太大：策略可能不愿快速调整腿部，形成僵硬步态；
- 若 command 突变，过强 action-rate 会拖慢响应。

所以它应结合动作频谱、关节轨迹和 tracking error 判断，而不是只根据总奖励判断。

### 4.5 torque penalty

\[
-\lVert\tau\rVert^2.
\]

作用是抑制不必要的大力矩，但当前系数 `2e-5` 仍属于工程起点。

风险包括：

- 系数过大：宁愿不走也不出力；
- 系数过小：学出冲击大、能耗高、不可迁移的步态；
- actuator force 的数值尺度取决于模型和控制接口，不能直接照抄别的框架。

### 4.6 为什么删除固定 alive bonus

固定 alive bonus 每步只要不摔倒就得到同样奖励。

它的好处是：

- 早期策略容易获得稳定信号；
- 可以强烈鼓励先学会不摔。

它的风险是：

- 如果速度跟踪奖励不够强，原地站立就能获得高回报；
- episode 越长，站立累计的 bonus 越大；
- return 上升不一定表示 tracking 改善。

本项目删除 alive bonus，是为了减少“站着不动”的局部最优，但这也不是无条件正确。如果训练表现为策略完全无法先学会稳定站立，可以重新评估：

- 是否需要小的 survival term；
- 是否需要改变 termination penalty；
- 是否需要更温和的初始化或随机化；
- 是否需要分阶段训练。

### 4.7 奖励分解必须记录

总 return 无法回答策略为什么得分。

因此每步记录：

```python
info["reward_info"]
```

包括：

- `tracking_lin_vel`；
- `tracking_ang_vel`；
- `lin_vel_z`；
- `ang_vel_xy`；
- `orientation`；
- `action_rate`；
- `torque`；
- `termination`；
- `total`。

一个高 return 策略可能是：

- 真正跟踪速度；
- 只在低速度命令时站稳；
- 用滑动或跳跃投机；
- 依靠某个权重漏洞获得分数。

必须结合视频和分项指标判断。

---

## 5. PPO 训练结构改进

### 5.1 原配置的问题不只是 `n_steps` 大

原 32 环境配置近似为：

```text
n_steps = 1000
rollout size = 32,000
batch_size = 64
n_epochs = 10
```

每个 epoch 的 minibatch 数：

```text
32,000 / 64 = 500
```

每次 rollout 的优化 minibatch 次数：

```text
500 × 10 = 5,000
```

这意味着：

- 收集一次数据后，长时间反复优化；
- 在重新采集最新 on-policy 数据前，策略已经更新很多次；
- batch 很小，单次梯度方差较大；
- 训练吞吐可能被大量小 minibatch 更新拖慢。

问题不能简单概括为“rollout 太长”，而是 rollout、batch size、epoch 数和并行环境数的组合不协调。

### 5.2 当前结构

```text
n_steps = 64
rollout size = 32 × 64 = 2,048
batch_size = 512
minibatches per epoch = 2,048 / 512 = 4
n_epochs = 5
optimizer minibatches per rollout = 4 × 5 = 20
```

优点：

- 更频繁地重新收集当前策略的数据；
- 每个 minibatch 更大；
- 每批数据重复利用次数更可控；
- 更接近常见 locomotion PPO 的“短 per-env rollout + 大并行 batch”结构。

但本项目只有 32 个环境，不具备官方数千环境 baseline 的总 batch 规模，所以 `64` 仍然需要实验验证。

### 5.3 `n_steps` 控制的不是 episode 长度

`n_steps=64` 表示每个环境为一次 PPO rollout buffer 提供 64 步。

在 50 Hz 下：

```text
64 × 0.02 s = 1.28 s
```

这只是一次参数更新前收集的轨迹片段长度。

如果环境没有终止，状态会跨 rollout 连续：

```text
episode:
step 0 ------- 64 ------- 128 ------- 192 ------- ... ------- 1000
               PPO update  PPO update  PPO update
```

不是：

```text
step 0 ------- 64 -> reset
step 0 ------- 64 -> reset
```

因此，“狗需要约一秒才走起来，所以 64 步结束时还没走几步，训练只看到站立”这一批评混淆了 rollout boundary 和 episode boundary。

### 5.4 rollout 尾部如何处理：bootstrap

在一个 rollout 的最后一个状态 \(s_T\)，若 episode 并未真正终止，critic 会估计：

\[
V(s_T).
\]

TD residual 为：

\[
\delta_t
=
r_t+\gamma V(s_{t+1})-V(s_t).
\]

GAE 为：

\[
\hat A_t
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l\delta_{t+l},
\]

并在 rollout 尾部通过 \(V(s_T)\) 进行 bootstrap。

所以 rollout 结束不等于把后续回报设为零。

只有真实失败 termination 才应阻断未来价值。时间上限 truncation 通常仍允许 bootstrap，具体依赖 Gymnasium/SB3 wrapper 对终止标志的处理。

### 5.5 64 步的 GAE 有效范围

当前：

\[
\gamma=0.99,\qquad\lambda=0.95,
\]

所以：

\[
\gamma\lambda=0.9405.
\]

相隔 \(l\) 步的 TD residual 权重为：

\[
(0.9405)^l.
\]

在 64 步处：

\[
(0.9405)^{64}\approx0.0197.
\]

这个数字表示第 64 步远的单个 residual 系数约为起始系数的 2%。对无限几何级数而言，64 步之后的尾部占总权重的比例也约为 1.97%。

因此 64 步覆盖了 GAE 大部分有效衰减范围。

但要注意：

- 这不意味着 critic 一定准确；
- bootstrap error 仍会传回 rollout 内；
- 环境中若存在很长延迟奖励，短 rollout 会更依赖 value function；
- 当前奖励是逐步稠密 tracking reward，因此比稀疏终局奖励更适合短 rollout。

### 5.6 `n_steps=64` 真正可能的问题

它真正可能的问题不是“狗没时间走”，而是：

1. 每轮总样本只有 2048；
2. 完整命令空间和域随机化使样本分布更复杂；
3. 32 个环境对不同运动阶段的覆盖有限；
4. critic 每轮拟合的数据较少；
5. gradient estimate 可能比大 batch baseline 更噪；
6. 100M 步会产生约 48,829 次 rollout update，更新非常频繁。

因此 `64` 是偏激进的高频更新起点，而不是无争议最优值。

若出现：

- `approx_kl` 高频触发；
- `clip_fraction` 持续偏高；
- value loss 大幅振荡；
- explained variance 长期为负；
- evaluation return 高方差；
- 策略行为在相邻 checkpoint 间反复退化；

可考虑：

```yaml
n_steps: 128
batch_size: 1024
```

此时：

```text
32 × 128 = 4096 transitions
4096 / 1024 = 4 minibatches per epoch
```

保持每 epoch 4 个 minibatch，但增加每次更新的数据覆盖。

### 5.7 为什么不直接使用 24 步

公开 Unitree/Isaac Lab locomotion PPO 常见每环境 24 步左右，但它们通常有数千并行环境：

```text
4096 × 24 = 98,304 transitions
```

本项目：

```text
32 × 24 = 768 transitions
```

两者不是同一个总 batch 量级。

因此不能只复制 `num_steps_per_env=24`，而忽略并行环境数。迁移时应同时考虑：

- per-env horizon；
- total rollout size；
- minibatch 数；
- batch size；
- epoch 数；
- learning rate；
- 归一化方式。

### 5.8 为什么使用 5 个 epoch

PPO 会用同一批 rollout 数据训练多轮。

epoch 太少：

- 样本利用率低；
- 每轮策略变化可能太小；
- critic 可能拟合不足。

epoch 太多：

- 策略反复拟合同一批旧数据；
- 新策略逐渐偏离采样策略；
- clip fraction 和 KL 可能变大；
- on-policy 近似变弱。

5 个 epoch 是成熟 locomotion 实现中常见的折中，但仍需结合 `approx_kl` 和 explained variance 判断。

### 5.9 batch size 与 minibatch 数

当前：

```text
rollout size = 2048
batch size = 512
minibatches per epoch = 4
```

这里选择“4 个 minibatch”是参考常见 locomotion PPO 训练结构，而不是理论规定。

更大的 batch：

- 梯度更稳定；
- 更新次数更少；
- 可能降低优化噪声。

更小的 batch：

- 更新次数更多；
- 噪声可能帮助探索；
- 但更容易产生不稳定 KL 和 value loss。

### 5.10 熵系数

当前：

```yaml
ent_coef: 0.01
```

策略熵奖励鼓励动作分布保持一定随机性，防止标准差过早收缩。

对 12 维连续动作，早期过早确定化可能导致：

- 只会站立；
- 卡在单一不良步态；
- 无法探索腿间协调。

但熵过大也可能导致：

- 动作噪声长期过强；
- 策略无法形成稳定周期；
- evaluation deterministic 表现与训练 stochastic 表现差异过大；
- torque/action-rate penalty 持续很高。

因此应同时观察：

- `train/std`；
- entropy loss；
- deterministic evaluation；
- 动作轨迹平滑性。

### 5.11 `vf_coef=1.0`

价值损失权重决定 critic loss 在总优化目标中的相对强度。

提高 `vf_coef` 的动机是让 critic 更充分拟合 bootstrap value，尤其短 rollout 更依赖 \(V(s_T)\)。

但若 actor 和 critic 共享特征层，价值损失过强可能压制策略特征。判断依据包括：

- explained variance；
- value loss；
- policy gradient loss；
- KL 和 clip fraction；
- actor/critic 是否使用独立网络。

---

## 6. `target_kl`：PPO-Clip 之外的更新保险

### 6.1 PPO 概率比

旧策略为 \(\pi_{\theta_{\mathrm{old}}}\)，当前策略为 \(\pi_\theta\)。

概率比：

\[
r_t(\theta)
=
\frac{
\pi_\theta(a_t\mid s_t)
}{
\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)
}.
\]

若 \(r_t=1\)，说明新旧策略对已采样动作给出的概率密度相同。

### 6.2 PPO-Clip 目标

\[
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_t
\left[
\min
\left(
r_t\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

当前：

```yaml
clip_range: 0.2
```

即 clipping 区间为：

```text
[0.8, 1.2]
```

### 6.3 clip 为什么不是硬约束

常见误解：

> clip_range=0.2 保证所有概率比最终都在 [0.8, 1.2]。

实际上它只改变 surrogate objective 的形状。

例如某些样本已经进入 clip 区域后：

- 这些样本不再通过对应 clipped 项继续提供同样的收益；
- 但其他样本仍然产生梯度；
- 所有样本共享同一套网络参数；
- 更新其他状态时，已 clip 样本的输出仍可能被间接改变；
- 多个 minibatch 和 epoch 会继续累积变化。

所以 clip 是“减少继续走远的优化动力”，不是“投影到一个硬可行集合”。

### 6.4 KL divergence 衡量什么

对状态 \(s\)：

\[
D_{\mathrm{KL}}
\left(
\pi_{\mathrm{old}}(\cdot\mid s)
\|
\pi_{\mathrm{new}}(\cdot\mid s)
\right)
=
\mathbb E_{a\sim\pi_{\mathrm{old}}}
\left[
\log
\frac{
\pi_{\mathrm{old}}(a\mid s)
}{
\pi_{\mathrm{new}}(a\mid s)
}
\right].
\]

它衡量动作概率分布整体改变了多少。

它不等于：

- 参数变化百分比；
- 动作均值变化百分比；
- 每个动作概率变化百分比。

因此：

```yaml
target_kl: 0.01
```

不能解释为“策略只允许变化 1%”。

### 6.5 一个一维高斯直觉

假设：

\[
\pi_{\mathrm{old}}
=
\mathcal N(\mu_{\mathrm{old}},\sigma^2),
\qquad
\pi_{\mathrm{new}}
=
\mathcal N(\mu_{\mathrm{new}},\sigma^2),
\]

且方差相同，则：

\[
D_{\mathrm{KL}}
=
\frac{
(\mu_{\mathrm{new}}-\mu_{\mathrm{old}})^2
}{
2\sigma^2
}.
\]

若 KL 为 `0.01`：

\[
|\mu_{\mathrm{new}}-\mu_{\mathrm{old}}|
\approx
0.141\sigma.
\]

但真实 Go2 策略为 12 维高斯，均值和标准差都可能变化，而且不同状态下输出不同，所以这个例子只能建立量级直觉。

### 6.6 SB3 中 `target_kl` 如何生效

它不是加入 loss 的固定 KL penalty，也不是 TRPO 那样求解严格约束问题。

SB3 使用 sampled actions 估计 approximate KL。当前常见实现使用类似：

\[
\widehat D_{\mathrm{KL}}
=
\mathbb E
\left[
\exp(\Delta\log\pi)-1-\Delta\log\pi
\right],
\]

其中：

\[
\Delta\log\pi
=
\log\pi_{\mathrm{new}}(a_t\mid s_t)
-
\log\pi_{\mathrm{old}}(a_t\mid s_t).
\]

若 approximate KL 超过实现设定的容许范围，SB3 会停止当前 rollout 剩余的 epoch。常见实现会以约 `1.5 × target_kl` 作为停止判断，具体应以项目安装的 SB3 版本源码为准。

例如：

```yaml
n_epochs: 5
target_kl: 0.01
```

可能执行：

```text
epoch 1: update
epoch 2: update
epoch 3: approximate KL exceeds threshold -> stop
epoch 4: skipped
epoch 5: skipped
```

它不会：

- 停止整个训练；
- reset 环境；
- 回滚已经发生的更新；
- 自动减小 learning rate；
- 保证真实 KL 严格小于 0.01。

### 6.7 clip、KL 和梯度裁剪的区别

| 机制 | 作用层面 | 主要作用 |
|---|---|---|
| `clip_range` | 样本概率比目标 | 减少过大策略变化继续带来的 surrogate 收益 |
| `target_kl` | 新旧策略分布 | 变化过大时提前停止剩余 epoch |
| `max_grad_norm` | 参数梯度 | 限制单次反向传播的梯度范数 |

三者互补，但都不是绝对稳定性证明。

### 6.8 为什么当前加入 `target_kl=0.01`

完整任务同时包含：

- 12 维连续动作；
- 前向、横向和偏航命令；
- 质量、摩擦、PD 增益随机化；
- 不同环境处于不同步态阶段；
- 每批数据训练多个 epoch。

某个 minibatch 可能包含较大的优势值，使一次 rollout 内策略变化异常剧烈。

`target_kl` 的定位是保险丝：

> 正常情况由 PPO-Clip 学习；若当前这批数据已经把策略推得较远，则少做后续 epoch。

### 6.9 什么时候 KL 保险可能反而妨碍训练

若 `target_kl` 太小：

- 每次只完成很少 epoch；
- 策略更新长期过弱；
- 训练吞吐被频繁 early stopping 限制；
- critic 也可能得不到计划中的训练次数。

若频繁触发，应先检查：

- learning rate；
- advantage 是否存在离群值；
- reward scale；
- batch size；
- observation scale；
- policy std 是否剧烈变化。

不能看到 early stopping 就立即提高 target KL，也不能看到 KL 低就立即增大学习率，必须结合性能指标。

---

## 7. 为什么保留完整任务和域随机化

曾考虑先关闭随机化、固定前向命令，让机器人先学会走，再逐步扩展任务。

这种课程学习有明确优点：

- 更容易确认环境和奖励是否可学习；
- 初期状态分布更简单；
- 调试成本更低；
- 失败时更容易定位原因。

但本实验最终保留完整任务和随机化，是因为目标不是只制作一个最容易成功的演示，而是直接验证：

> 在完整 planar command tracking 与模型参数变化下，32 环境 PPO 是否能够学习鲁棒 locomotion。

当前保留：

- mass randomization；
- friction randomization；
- PD-gain randomization；
- 完整 x/y/yaw command。

这是一种实验目标选择，不代表课程学习不正确。

### 7.1 保留随机化的风险

随机化会增加条件分布宽度：

\[
p(s,a,s')
=
\int
p(s,a,s'\mid\xi)p(\xi)\,d\xi,
\]

其中 \(\xi\) 表示质量、摩擦、控制增益等参数。

风险包括：

- 同一动作在不同环境中产生不同结果；
- critic 更难拟合；
- 2048 样本可能不足以覆盖所有条件；
- 早期策略尚未形成基础步态就要同时适应模型变化；
- evaluation 方差更高。

所以保留随机化后，更需要：

- 固定 seed 的可复现实验；
- 记录每个随机参数；
- 分项 evaluation；
- 必要时做消融。

### 7.2 为什么修复随机化不等于关闭随机化

修复的是：

> 每次 reset 必须从标称模型独立采样。

不是：

> 所有 episode 都使用相同模型。

正确随机化：

\[
m_k=m_0\alpha_k.
\]

错误累乘：

\[
m_k=m_0\prod_{i=1}^{k}\alpha_i.
\]

前者是有界独立采样，后者是跨 episode 漂移。

---

## 8. 争议与疑问详解

本节集中回答讨论中容易产生误解的地方。

### 8.1 “这些改进是不是都来自权威四足 baseline？”

不是。

应分三类看待。

#### 第一类：直接由代码语义确认的问题

不依赖权威 baseline：

- `frame_skip` 声明为 10，却只推进一个物理步；
- 质量和摩擦在 reset 中累乘；
- observation 与 reward 的速度坐标系不一致；
- action magnitude 被称为 action smoothness；
- 更新 `previous_action` 顺序可能使差分项失效。

这些问题可以直接通过代码、物理时间和变量定义判断。

#### 第二类：多个成熟 baseline 的共同结构

例如：

- 约 50 Hz 的 locomotion policy rate；
- 相对默认关节姿态；
- body-frame velocity tracking；
- action-rate penalty；
- 短 per-env rollout；
- 5 个左右的 epoch；
- 少量大 minibatch；
- KL 监控或 early stopping。

这些方向有公开实现支持。

#### 第三类：本项目自己的工程适配

例如：

- `n_steps=64`；
- `batch_size=512`；
- `[256, 256]` 网络；
- `vf_coef=1.0`；
- 当前奖励权重；
- `target_kl=0.01`；
- 100M 总训练量。

这些不是从某个官方配置逐项复制的，需要本项目实验验证。

### 8.2 “`n_steps=64` 只有 1.28 秒，狗还没走起来就结束了”

这句话中的时间计算正确：

```text
64 × 0.02 = 1.28 s
```

但把 rollout 结束理解为 episode reset 是错误的。

PPO 更新可以发生在连续 episode 中间。机器人在第 64 步的状态会继续成为第 65 步的起点，除非它真正摔倒或达到 episode 上限。

所以训练仍然能看到：

- 从站立进入步态；
- 步态稳定阶段；
- 转向过程；
- 长时间 tracking；
- 摔倒前的状态。

这些状态可能分布在相邻多个 rollout buffer 中。

真正的争议应表述为：

> 2048 个样本一次更新，在完整随机化任务下是否足够稳定？

而不是：

> 机器人每 1.28 秒就被重置。

### 8.3 “轨迹被切开后，长期回报不会丢失吗？”

不会简单丢失，因为 rollout 尾部用 critic bootstrap。

但也不能说完全没有代价。

短 rollout 更依赖：

\[
V(s_T)
\]

的准确性。如果 critic 很差，尾部 bootstrap error 会影响优势估计。

因此需要观察：

- explained variance；
- value loss；
- return 与 value prediction 的一致性；
- `n_steps=64` 和 `128` 的消融。

当前任务每步都有速度 tracking 和稳定性奖励，属于稠密奖励，因此对完整 episode return 的依赖低于稀疏终局任务。

### 8.4 “为什么官方 24 步可以，我们 64 步还可能太短？”

因为 per-env steps 不是唯一变量。

官方可能是：

```text
4096 envs × 24 steps = 98,304 samples
```

本项目是：

```text
32 envs × 64 steps = 2,048 samples
```

虽然本项目单环境轨迹更长，总 batch 却小得多。

因此比较 rollout 时必须同时比较：

- 并行环境数；
- 总 rollout size；
- minibatch size；
- epoch 数；
- 数据归一化；
- optimizer 和 learning rate。

### 8.5 “为什么不直接用 `n_steps=128`？”

`128` 是合理候选，而且在完整随机化任务下可能更稳。

未直接确定为 128 的原因：

- 64 能更频繁获取新 on-policy 数据；
- 当前 GAE 的主要权重范围已被覆盖；
- 需要先用实际 KL、critic 和 return 方差判断；
- 增加 rollout size 会降低更新频率；
- 100M 步下两者的 wall-clock 和优化行为不同。

更严谨的结论是：

- 64 不是“严重错误”；
- 128 可能是更保守的选择；
- 应通过固定 seed 的对照实验决定。

建议对照：

```text
A: n_steps=64,  batch_size=512
B: n_steps=128, batch_size=1024
```

二者都保持每 epoch 4 个 minibatch，减少混杂变量。

### 8.6 “有 clip 了，为什么还需要 KL？”

clip 不是硬约束。

即使某些样本被 clip：

- 其他样本仍更新共享网络；
- 多个 minibatch 继续改变参数；
- 多个 epoch 累积变化；
- 高维高斯的均值和标准差都可能变化。

KL 检测的是整体分布是否已经移动过远。

所以：

- clip 是局部目标函数整形；
- KL 是全局分布变化监控。

### 8.7 “`target_kl=0.01` 是限制策略只变 1% 吗？”

不是。

KL 没有这种直接百分比含义。它是分布差异量，受动作维数、均值变化和标准差变化共同影响。

对于 12 维动作，总 KL 还会聚合各维变化，因此不能拿一维直觉直接解释为每个关节允许变化多少。

### 8.8 “KL early stopping 会不会把训练停掉？”

不会停掉整个训练。

它只可能停止当前 rollout 数据剩余的 epoch：

```text
collect rollout
epoch 1
epoch 2
KL too large
skip remaining epochs
collect next rollout
continue training
```

环境也不会因为 KL early stopping 自动 reset。

### 8.9 “`target_kl` 会不会把 PPO 变成 TRPO？”

不会。

TRPO 的核心是近似求解带 KL trust-region 约束的优化问题。

当前 SB3 PPO 仍然优化 clipped surrogate，`target_kl` 只是事后监控并提前停止，不提供严格 trust-region 保证。

### 8.10 “动作尺度 0.25 会不会太小，导致迈不开腿？”

有可能，所以它属于需要验证的参数，而不是确定事实。

应查看：

- 动作是否长期饱和；
- joint target range；
- 步长和足端 clearance；
- 高速命令下 tracking 是否系统性不足；
- 增大 action scale 后是否改善速度但增加摔倒。

如果动作长期接近 \(\pm1\) 且速度仍上不去，0.25 可能限制能力；如果动作很小但机器人仍剧烈运动，问题更可能在 PD 增益或模型。

### 8.11 “保留随机化会不会让它根本学不会走？”

有这个风险。

保留随机化是为了直接训练鲁棒策略，但会降低优化难度之外的可诊断性。

如果 100M 步仍失败，不能直接判断 PPO 不行。应做消融：

1. 固定物理参数，保留完整命令；
2. 只开 friction randomization；
3. 再开 PD randomization；
4. 最后开 mass randomization；
5. 比较每一步对性能和方差的影响。

这不是承认完整任务设置错误，而是定位失败来源。

### 8.12 “为什么删 alive bonus，不怕狗一开始直接摔吗？”

删掉它是为了避免站立局部最优，但确实会减少早期生存信号。

是否需要 alive bonus 取决于奖励相对尺度：

- termination penalty 是否足够；
- orientation reward 是否能提供站立信号；
- velocity tracking 是否在静止时仍有梯度；
- reset 初始姿态是否稳定。

若策略早期完全无法延长 episode，可以考虑小的 alive bonus，但必须检查它是否重新主导总 return。

### 8.13 “指数 tracking reward 会不会在误差大时梯度太小？”

会，这是指数奖励的典型风险。

\[
r=\exp(-e^2/\sigma)
\]

当 \(e^2\gg\sigma\) 时，奖励接近零，不同坏状态之间难以区分。

当前分母 `0.25` 控制有效区间。应记录 tracking error 分布：

- 如果大多数样本 reward 接近零，带宽可能过窄；
- 如果误差很大仍 reward 很高，带宽可能过宽；
- 也可以考虑 curriculum、线性/Huber 形式或多尺度 tracking reward。

### 8.14 “100M 步是不是肯定够？”

不是。

100M 只是训练预算，不是收敛保证。

影响因素包括：

- 环境吞吐和更新比；
- 奖励可学习性；
- 随机化难度；
- critic 稳定性；
- policy std；
- seed；
- 网络容量；
- 是否存在奖励漏洞。

正确做法是设置中间 checkpoint 和 evaluation，而不是训练结束后才第一次看视频。

### 8.15 “为什么不能把修复前后的 return 直接比较？”

因为 MDP 和时间尺度已经变化：

- 一个 action 的物理持续时间从约 2 ms 变成 20 ms；
- episode 的真实时间含义改变；
- reward 坐标系改变；
- action smoothness 定义改变；
- action scale 改变；
- observation 表达改变；
- PPO rollout/update 结构改变。

即使 return 数值相同，也不代表行为质量相同。

修复后应视为新的 baseline 实验。

### 8.16 “训练环境带随机化，评估环境也随机化吗？”

需要区分两种 evaluation：

#### 固定基准评估

使用固定命令、固定物理参数和固定 seed，用于比较 checkpoint 的学习进展。

#### 鲁棒性评估

在随机质量、摩擦和 PD 参数上运行多组 episode，用于衡量分布内鲁棒性。

若只做随机评估，曲线噪声可能掩盖真实进展；若只做固定评估，又无法证明鲁棒性。

### 8.17 “episode length 很长是不是就代表会走？”

不是。

长 episode 只说明没有触发 termination。

策略可能：

- 原地站立；
- 缓慢滑动；
- 趴低但未越过阈值；
- 以错误方向运动；
- 完全忽略 command。

必须同时看 tracking RMSE、位移、姿态、动作和视频。

---

## 9. 环境诊断与训练监控

### 9.1 环境诊断脚本

新增：

```text
scripts/diagnose_go2_env.py
```

运行：

```bash
PYTHONPATH=. python3 scripts/diagnose_go2_env.py
```

它检查：

1. 100 个 policy step 是否推进约 2 s；
2. observation 是否为 48 维 `float32`；
3. 域随机化连续 reset 后是否仍在标称范围；
4. action-rate 是否使用 \(\lVert a_t-a_{t-1}\rVert^2\)；
5. 一步 reward decomposition。

诊断通过只能证明这些局部机制符合预期，不能证明策略一定能学出自然步态。

### 9.2 PPO 优化指标

至少观察：

- `train/approx_kl`；
- `train/clip_fraction`；
- `train/entropy_loss`；
- `train/std`；
- `train/explained_variance`；
- `train/value_loss`；
- `train/policy_gradient_loss`；
- learning rate；
- early stopping 发生频率。

### 9.3 locomotion 指标

至少记录：

- episode return；
- episode length；
- fall rate；
- local \(v_x\) tracking RMSE；
- local \(v_y\) tracking RMSE；
- yaw-rate tracking RMSE；
- forward/lateral displacement；
- orientation RMS；
- action-rate cost；
- torque cost；
- actuator saturation rate；
- action saturation rate；
- 每个 reward component 的均值和分位数。

### 9.4 指标组合诊断

#### KL 高，clip fraction 高

策略更新激进。检查：

- learning rate；
- batch size；
- advantage outliers；
- reward scale；
- policy std。

#### KL 很低，性能不变

策略可能更新太弱。检查：

- learning rate；
- advantage 是否接近零；
- entropy/value loss 是否压制 actor；
- observation 是否有有效 command 信息。

#### explained variance 为负

critic 预测比简单常数基线还差。检查：

- reward scale；
- value network；
- observation；
- bootstrap/termination 语义；
- rollout size。

#### episode length 上升但 tracking 不改善

可能学会不摔但没学会服从 command，需要检查 alive/orientation/termination 与 tracking 权重关系。

#### return 上升但视频更差

可能存在 reward hacking，必须查看 reward decomposition。

---

## 10. 验证边界与消融顺序

### 10.1 已确认的修复

- 真正执行 `frame_skip`；
- 统一 episode 时间语义；
- 防止域随机化累乘；
- 统一 body-frame velocity；
- action-rate 使用动作差分；
- reward 在更新 previous action 之前计算；
- observation/action dtype 一致。

### 10.2 有公开实现支持但仍需验证的选择

- 50 Hz policy rate；
- action scale `0.25`；
- 相对默认关节姿态；
- 固定 observation scaling；
- 短 rollout；
- 5 epochs；
- 4 minibatches per epoch；
- entropy coefficient `0.01`；
- approximate-KL early stopping。

### 10.3 工程起点

- `n_steps=64` 或 `128`；
- `batch_size=512`；
- `vf_coef=1.0`；
- `[256,256]` 网络；
- tracking bandwidth `0.25`；
- torque coefficient；
- termination penalty；
- 当前随机化范围；
- 100M 训练预算。

### 10.4 推荐消融顺序

若正式训练失败，建议尽量一次只改变一个主因素：

1. 验证环境诊断；
2. 固定 seed 重跑短训练；
3. 比较 `n_steps=64` 与 `128`；
4. 查看 critic 指标；
5. 固定物理参数做对照；
6. 分别恢复 friction、PD、mass randomization；
7. 检查 reward bandwidth 和权重；
8. 最后才更换网络或算法。

避免同时修改：

- rollout；
- reward；
- randomization；
- network；
- learning rate。

否则即使性能改善，也无法知道原因。

---

## 11. 参考实现与迁移原则

设计判断参考了以下公开实现的共同结构：

- Unitree `unitree_rl_gym` Go2 locomotion 配置；
- ETH Zurich `legged_gym` PPO 结构；
- NVIDIA Isaac Lab Go2 PPO 配置；
- Google DeepMind MuJoCo Playground locomotion；
- Stable-Baselines3 PPO 实现；
- OpenAI Spinning Up PPO 说明。

这些参考的作用是确认合理方向，而不是证明本项目的具体数值正确。

不同框架可能存在：

- 数千环境与 32 环境的差异；
- GPU 向量化模拟与多进程 MuJoCo 的差异；
- privileged critic；
- observation normalization；
- adaptive learning rate；
- 不同 termination 和 timeout 处理；
- 不同动作标准差参数化；
- 不同 reward scale；
- 不同 actuator 模型。

迁移参数时应比较完整训练结构，而不能只复制单个字段。

---

## 12. 推荐运行流程

```bash
git switch exp/go2-mujoco
git pull origin exp/go2-mujoco

PYTHONPATH=. python3 scripts/diagnose_go2_env.py

PYTHONPATH=. python3 main.py \
  --config configs/go2/ppo_32env.yaml
```

当前训练应被视为：

> 在修复环境时间尺度、域随机化、坐标系和奖励定义后，对完整速度命令空间、完整域随机化及约 100M transitions 的 32 环境 PPO 基线进行重新验证。

它不是修复前实验的简单续跑。环境的物理时间、奖励语义、观测表达和 PPO 更新结构都已经发生实质变化。
