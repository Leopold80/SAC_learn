# Go2 MuJoCo PPO 环境与训练改进说明

本文档记录 `exp/go2-mujoco` 分支中对 Unitree Go2 MuJoCo 强化学习实验所做的主要改进、设计理由、参数含义和验证边界。

本文档的目标不是宣称当前配置已经是最优配置，而是明确区分：

1. 必须修复的环境正确性问题；
2. 有成熟四足 locomotion 基线支持的设计选择；
3. 仍需通过训练曲线和策略行为验证的工程超参数。

---

## 1. 当前实验目标

当前任务仍然是完整的平面速度指令跟踪，而不是仅向前行走的简化课程：

- 前向速度指令：`[0.0, 1.0] m/s`；
- 横向速度指令：`[-0.3, 0.3] m/s`；
- 偏航角速度指令：`[-0.5, 0.5] rad/s`；
- 质量随机化：`±15%`；
- 摩擦随机化：`±30%`；
- PD 增益随机化：`±15%`。

训练仍使用 32 个并行 MuJoCo 环境，并计划运行约 100M transitions。

由于 PPO 的 rollout size 为

```text
32 environments × 64 steps = 2048 transitions
```

而配置验证器要求总训练步数能够被 rollout size 整除，因此采用：

```yaml
training:
  timesteps: 100_001_792
```

它对应 48,829 次 rollout，略高于 100M transitions。

---

## 2. 必须修复的环境正确性问题

以下修改不是单纯的超参数偏好，而是原环境实现中的正确性问题。

### 2.1 正确执行 control decimation

环境声明了：

```python
frame_skip = 10
```

MuJoCo 模型的 physics timestep 为 `0.002 s`，因此每个策略动作应持续：

```text
10 × 0.002 s = 0.02 s
```

也就是 50 Hz 策略控制频率。

原实现虽然设置了 `frame_skip=10`，但每次 `step()` 只执行一次 `mj_step`。这会使真实控制周期变成约 `0.002 s`，即 500 Hz，与奖励尺度、episode 时长和四足 locomotion 常用设置均不一致。

现在改为：

```python
self.do_simulation(ctrl, self.frame_skip)
```

因此每个 policy step 会真正执行 10 个物理子步。

该修改同时改变了对 episode 长度的解释：

```text
1000 policy steps × 0.02 s = 20 s
```

Gymnasium 的 `TimeLimit` 负责 1000 步超时，环境内部不再额外使用与之冲突的 15 秒截断。

### 2.2 修复域随机化的累乘漂移

原实现每次 reset 直接执行类似：

```python
model.body_mass *= random_scale
model.geom_friction *= random_scale
```

这里的基准是上一次随机化后的参数，而不是标称参数。经过大量 reset 后，质量和摩擦会形成乘法随机游走，逐渐漂离预期范围。

现在初始化时保存不可变的标称参数：

```python
_nominal_body_mass
_nominal_body_inertia
_nominal_geom_friction
```

每次 reset 都重新从标称参数采样：

```python
current_mass = nominal_mass * sampled_scale
current_friction = nominal_friction * sampled_scale
```

因此随机化仍然开启，但始终受配置范围约束。

### 2.3 统一速度跟踪坐标系

观测中的线速度原本已经被转换到机体坐标系，但奖励却直接使用世界坐标系中的 `qvel[0]` 和 `qvel[1]`。

这会导致同一条速度指令在观测和奖励中代表不同物理量。例如机器人转向 90° 后，世界坐标系 x 速度不再等于机器人自身的前向速度。

现在：

- base linear velocity 转换到机体坐标系；
- command 中的前向和横向速度按机体坐标系解释；
- velocity tracking reward 也使用机体坐标系速度。

MuJoCo free joint 的角速度已经位于局部机体系，因此不再对其重复旋转。

### 2.4 修复动作平滑项

原奖励中的所谓“动作平滑”实际是：

```python
sum(action ** 2)
```

它惩罚动作幅值，而不是动作变化。真正的 action-rate penalty 应是：

```python
sum((action_t - action_{t-1}) ** 2)
```

即：

\[
J_{\mathrm{action\ rate}}
=
\lVert a_t-a_{t-1}\rVert_2^2.
\]

此外，必须在更新 `previous_action` 之前计算奖励，否则该差分会恒为零。

---

## 3. 动作与观测改进

### 3.1 动作尺度改为 0.25 rad

策略动作仍限制在：

```text
[-1, 1]^12
```

目标关节位置为：

\[
q_{\mathrm{target}}
=
q_{\mathrm{default}}+0.25a_t.
\]

将 action scale 从 `0.5 rad` 降到 `0.25 rad`，主要理由是：

- 降低随机初始策略产生剧烈关节目标变化的概率；
- 与 Unitree Go2 的公开 legged locomotion 配置保持相近量级；
- 在保留足够步态运动范围的同时提高早期训练稳定性。

这不是硬性理论要求，后续仍应根据关节轨迹、饱和率和最终步态判断是否需要调整。

### 3.2 使用相对默认姿态的关节观测

关节位置观测由绝对角度改为：

\[
q-q_{\mathrm{default}}.
\]

这样策略直接看到关节偏离名义站姿的程度，通常比绝对关节角更适合作为基于位置偏置的四足控制观测。

### 3.3 观测缩放

当前观测采用固定缩放：

```python
2.0 * local_linear_velocity
0.25 * local_angular_velocity
projected_gravity
commands
joint_position_error
0.05 * joint_velocity
previous_action
```

作用是缓解不同物理量数量级差异，使网络输入更容易保持在相近范围。

### 3.4 使用 `float32`

observation space、action space 和实际返回的 observation 改为 `float32`，与 Stable-Baselines3 和神经网络训练的常见数据类型一致，避免不必要的双精度传输和类型转换。

---

## 4. 奖励函数改进

当前奖励结构为：

\[
\begin{aligned}
r_t={}&
1.0r_{\mathrm{lin}}
+0.5r_{\mathrm{yaw}}
-2.0v_z^2
-0.05\lVert\omega_{xy}\rVert^2\\
&-1.0\lVert g_{xy}\rVert^2
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
-\frac{\lVert v_{xy}^{\mathrm{cmd}}-v_{xy}^{\mathrm{body}}\rVert^2}{0.25}
\right).
\]

线速度使用机体坐标系，因此前向和横向命令在机器人转向后仍具有一致含义。

### 4.2 偏航角速度跟踪

\[
r_{\mathrm{yaw}}
=
\exp\left(
-\frac{(\omega_z^{\mathrm{cmd}}-\omega_z)^2}{0.25}
\right).
\]

线速度和偏航角速度被拆为两个独立项，而不是混合成单一误差范数。这样可以独立调整二者权重，并在日志中分别诊断。

### 4.3 稳定性与平滑性惩罚

当前包含：

- 竖直线速度惩罚；
- 横滚和俯仰角速度惩罚；
- projected gravity 水平分量惩罚；
- action-rate penalty；
- 关节力矩平方惩罚；
- 摔倒终止惩罚。

原来的固定 alive bonus 被移除，原因是持续存在的存活奖励可能使“原地站着”成为较强局部最优，尤其当速度跟踪奖励较弱时。

### 4.4 奖励分解日志

每个 reward component 都写入：

```python
info["reward_info"]
```

包含：

- `tracking_lin_vel`；
- `tracking_ang_vel`；
- `lin_vel_z`；
- `ang_vel_xy`；
- `orientation`；
- `action_rate`；
- `torque`；
- `termination`；
- `total`。

因此不能只观察 episode return，还应检查策略究竟通过什么行为获得奖励。

---

## 5. PPO 训练配置

当前核心配置为：

```yaml
ppo:
  learning_rate: 0.0003
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

### 5.1 为什么从长 rollout 改为短 rollout

原配置中每个环境收集约 1000 步后才更新。32 个环境时，一次 rollout 包含：

```text
32 × 1000 = 32,000 transitions
```

并且原配置使用 `batch_size=64`、`n_epochs=10`，意味着同一批 rollout 会产生大量 minibatch 更新，策略在重新收集数据前会反复使用较旧的数据。

当前配置为：

```text
32 × 64 = 2048 transitions per rollout
2048 / 512 = 4 minibatches per epoch
5 epochs × 4 minibatches = 20 gradient passes per rollout
```

这样可以更频繁地重新采集 on-policy 数据，同时减少每批数据的重复利用次数。

### 5.2 `n_steps=64` 是否意味着 episode 只有 1.28 秒

不是。

`n_steps=64` 表示 PPO 每次从每个环境中收集 64 步后进行一次参数更新：

```text
64 × 0.02 s = 1.28 s per rollout segment
```

但 rollout 结束不会自动 reset 环境。若机器人没有摔倒且未达到 episode 时间上限，下一次 rollout 会从当前状态继续：

```text
episode:
0 ---- 64 ---- 128 ---- 192 ---- ... ---- 1000
       update   update    update
```

因此不能把 rollout boundary 解释成 episode boundary。

在 rollout 最后一个状态，GAE 会使用 critic 的价值估计进行 bootstrap，而不是把未来价值直接置零。只有真正的 termination 才会截断 bootstrap。

当前：

\[
\gamma\lambda=0.99\times0.95=0.9405,
\]

64 步后的 GAE 权重约为：

\[
(0.9405)^{64}\approx0.0197.
\]

因此 64 步已经覆盖了 GAE 中大部分有效衰减范围。

不过，`n_steps=64` 并非没有代价。真正需要关注的是每轮只有 2048 个样本，在完整 command space 和域随机化下，梯度方差可能偏大。如果训练中出现明显噪声、critic 不稳定或 KL 经常触发，可以考虑：

```yaml
n_steps: 128
batch_size: 1024
```

这样仍保持每个 epoch 4 个 minibatch，但每次 rollout 增加到 4096 个样本。

所以当前结论是：

- `n_steps=64` 不是因为“机器人没来得及走起来”而错误；
- 它是一种偏高频更新的选择；
- 是否应改成 128，应依据实际训练指标，而不是依据 rollout 时长等同于 episode 时长的误解。

### 5.3 为什么使用 5 个 epoch

PPO 每次使用 rollout 数据训练多个 epoch。epoch 太多会使策略反复拟合同一批旧数据，增加策略偏离行为策略的风险。

使用 5 个 epoch 是在样本利用率和 on-policy 程度之间的折中，也与常见 legged locomotion PPO 配置接近。

### 5.4 为什么提高 entropy coefficient

当前使用：

```yaml
ent_coef: 0.01
```

较高的熵正则有助于防止 12 维连续动作策略在早期过快收缩，尤其是在：

- 完整速度指令空间；
- 多种域随机化；
- 初始策略尚未形成稳定步态；

同时存在时。

但熵并非越高越好。若训练后期动作标准差持续过大、步态噪声明显或 return 无法收敛，应重新评估该值。

---

## 6. `target_kl`：PPO-Clip 之外的更新保险

### 6.1 PPO-Clip 的作用

PPO 使用概率比：

\[
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\theta_{\mathrm{old}}}(a_t\mid s_t)}.
\]

clip objective 为：

\[
L^{\mathrm{CLIP}}(\theta)
=
\mathbb E_t\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

当 `clip_range=0.2` 时，超出 `[0.8, 1.2]` 的概率比不会继续从 clipped surrogate 中获得同样的优化收益。

但 clip 不是严格的硬约束。共享网络参数、多个 minibatch 和多个 epoch 仍可能使新策略整体偏离旧策略较远。

### 6.2 KL divergence 测量什么

KL divergence 衡量的是新旧动作概率分布的整体差异：

\[
D_{\mathrm{KL}}\left(
\pi_{\mathrm{old}}(\cdot\mid s)
\|\pi_{\mathrm{new}}(\cdot\mid s)
\right).
\]

它不是参数变化百分比，也不能把 `0.01` 直接解释成“策略只变化 1%”。

### 6.3 Stable-Baselines3 中的作用方式

`target_kl` 在当前配置中不是新的 loss 项，也不是类似 TRPO 的严格约束优化。

Stable-Baselines3 会在每个训练 epoch 后估计 approximate KL。如果新旧策略分布变化过大，就提前终止当前 rollout 剩余的 PPO epoch。

例如：

```yaml
n_epochs: 5
target_kl: 0.01
```

可能出现：

```text
epoch 1: update
epoch 2: update
epoch 3: approximate KL too large -> stop
epoch 4/5: skipped for this rollout
```

它不会停止整个训练，也不会 reset 环境，只是防止继续反复使用当前这批数据把策略推得更远。

因此三种保护分别作用于不同层面：

| 参数 | 作用对象 |
|---|---|
| `clip_range` | 单个样本概率比在 surrogate objective 中的贡献 |
| `target_kl` | 新旧策略整体分布差异及 epoch early stopping |
| `max_grad_norm` | 单次反向传播的梯度范数 |

加入 `target_kl=0.01` 的目的，是在完整命令空间和域随机化带来较高数据异质性时，为偶发的激进策略更新提供一道保险。

---

## 7. 域随机化为何保留

本实验没有采用“先关闭全部随机化，学会向前走后再开启”的简化课程，而是保留：

- mass randomization；
- friction randomization；
- PD-gain randomization。

理由是当前目标是直接评估完整任务下的鲁棒 locomotion 学习，而不是只建立最容易收敛的演示基线。

但保留随机化并不意味着忽略可诊断性。必须确保：

1. 随机化不会累积漂移；
2. 每个随机量都能记录或复现；
3. evaluation 的随机性和 seed 被明确管理；
4. 若训练失败，能够分别关闭某一种随机化进行消融，而不是立即归因于 PPO。

---

## 8. 环境诊断脚本

新增：

```text
scripts/diagnose_go2_env.py
```

它用于在长时间训练前进行快速检查：

1. 100 个 policy steps 是否准确推进约 2 秒；
2. observation 是否为 48 维 `float32`；
3. 域随机化在连续 reset 后是否仍处于标称范围；
4. action-rate 是否确实使用 `a_t-a_{t-1}`；
5. 打印一次 reward component 快照。

运行：

```bash
PYTHONPATH=. python3 scripts/diagnose_go2_env.py
```

诊断脚本通过只能说明若干关键机制符合预期，不能证明奖励函数一定能产生自然步态。

---

## 9. 训练时必须观察的指标

不能只看 episode return。至少应同时观察：

### PPO 优化指标

- `train/approx_kl`；
- `train/clip_fraction`；
- `train/entropy_loss`；
- `train/std`；
- `train/explained_variance`；
- `train/value_loss`；
- `train/policy_gradient_loss`。

### locomotion 指标

- episode length；
- fall rate；
- local `v_x`、`v_y` 和 yaw-rate tracking RMSE；
- 前向位移和横向漂移；
- orientation RMS；
- action-rate cost；
- torque cost；
- 各 reward component 的均值。

### 对 `target_kl` 的判断

- `approx_kl` 长期接近零且性能停滞：更新可能过弱；
- 经常显著超过目标并触发 early stopping：更新可能过猛；
- `clip_fraction` 和 KL 同时很高：大量样本已进入 clip 区域，策略变化较快；
- KL 合理但 critic 的 explained variance 长期为负：问题更可能在价值函数、观测或奖励尺度。

---

## 10. 当前改进的验证边界

### 已具有明确依据的修改

- 正确执行 `frame_skip`；
- 防止域随机化累乘漂移；
- 统一机体速度坐标系；
- action-rate 使用动作差分；
- 使用相对默认姿态观测；
- 分解并记录 reward components；
- rollout boundary 不等于 episode boundary。

### 有成熟 baseline 支持、但仍需本项目验证的选择

- 50 Hz policy rate；
- action scale `0.25`；
- 短 rollout；
- 5 epochs；
- 4 minibatches per epoch；
- entropy coefficient `0.01`；
- approximate-KL early stopping。

### 仍属于工程起点的参数

- `n_steps=64` 还是 `128`；
- 当前 reward 权重；
- torque penalty 数量级；
- termination penalty；
- 网络规模 `[256, 256]`；
- 固定学习率还是自适应/线性衰减；
- 当前域随机化强度是否过强。

这些参数不能仅凭“看起来像官方 baseline”判定正确，必须通过本项目自己的训练曲线、视频和消融实验验证。

---

## 11. 参考实现

本次设计判断主要参考以下公开实现的共同结构，而不是机械复制某一套参数：

- Unitree `unitree_rl_gym` Go2 locomotion configuration；
- ETH Zurich `legged_gym` PPO locomotion structure；
- NVIDIA Isaac Lab Go2 PPO configuration；
- Google DeepMind MuJoCo Playground locomotion tasks；
- Stable-Baselines3 PPO implementation and `target_kl` early stopping；
- OpenAI Spinning Up PPO explanation。

不同框架在并行环境数、rollout buffer、网络结构、归一化和优化器实现上存在差异，因此不能直接把数千环境下的参数原样移植到 32 环境 SB3 实验。

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

> 在修复环境时序、随机化和奖励定义之后，对完整速度命令空间、完整域随机化和约 100M transitions 的 32 环境 PPO 基线进行重新验证。

它不应与修复前的训练曲线直接视为同一实验续跑，因为环境时间尺度、奖励含义、观测表达和 PPO 更新结构都已经发生实质变化。
