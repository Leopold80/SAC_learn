# LTC-SAC 改进思路

## 1. 当前定位

当前代码中的 LTC 可以概括为：

```text
fixed-window LTC temporal feature extractor
```

也就是利用 `frame_stack` 提供的最近若干帧，在窗口内部做 LTC 状态递推，再把最终状态作为 SAC actor / critic 的特征输入。

这个设计的特点是：

```text
最近 K 帧 observation → LTC encoder → feature → SAC
```

它不是静态 MLP，也不是完整的跨步 recurrent SAC，而是一个轻量、直接、容易训练验证的折中方案。

---

## 2. simple LTC 作为 legacy

`ltc_simple` 建议保留，但只作为 legacy / 历史对照分支，不再作为默认主线。

理由：

- 它对应早期的简化 LTC encoder，便于复查旧结果；
- 代码轻，适合作为调试参考；
- 早期笔记曾记录过中后期退化，但当前仓库没有保留对应原始产物，因此不能把它当作已复现实验结论；
- 后续正式对比应优先看 `mlp` 和公式版 `ltc`。

当前命名约定：

```text
mlp        → stacked observation MLP baseline
ltc        → 公式版 circuit LTC，当前主线
ltc_residual → circuit LTC + raw residual / concat
ltc_residual_action → 动作历史只进入 LTC 分支
ltc_simple → legacy 简化 LTC
```

后续判断不靠单次曲线，而应通过多 seed 的 best reward、final reward、last10 稳定性和训练 FPS 综合比较。

---

## 3. 使用 residual / concat 结构

当前 LTC feature 如果完全替代原始观测，可能丢失一些直接有用的信息，例如高度、角度、速度、接触状态等。

更推荐改成：

```text
raw observation projection ┐
                           ├→ concat → fusion MLP → actor / critic
LTC temporal feature      ┘
```

这样做的好处：

- 保留原始观测中的直接 Markov 信息；
- LTC 只负责补充时序特征，而不是承担全部信息压缩；
- actor / critic 可以自动选择依赖 raw feature 还是 temporal feature；
- 训练稳定性通常更好。

建议结构：

```python
raw_features = raw_proj(obs_flat)
ltc_features = ltc_encoder(obs_seq)
features = torch.cat([raw_features, ltc_features], dim=-1)
features = fusion(features)
```

其中：

```text
obs_flat = 展平后的 stacked observation
obs_seq  = reshape 后的时序 observation
```

---

## 4. 加入过去动作历史

对于真实控制任务，尤其是 USV/UAV，过去动作往往很重要。

当前 frame stack 主要提供：

```text
[o_{t-K+1}, ..., o_t]
```

可以进一步扩展成：

```text
[(o_{t-K+1}, a_{t-K+1}), ..., (o_t, a_{t-1})]
```

也就是让 observation 中包含：

```text
当前状态
最近 K 步状态历史
最近 K 步动作历史
```

这样 LTC 不只看到状态变化，也能看到控制输入历史，从而更容易感知：

- 执行器滞后；
- 控制输入惯性；
- 舵机 / 推进器延迟；
- 状态变化与动作之间的动态关系。

这一路线仍然可以保持标准 SB3 SAC，不需要重写算法主体。

当前实现中，`ltc_residual_action` 只让动作历史进入 LTC encoder；raw residual 分支仍只使用原始 observation。这样可以更清楚地判断动作历史对时序分支的增益，而不是让所有分支同时获得额外信息。

---

## 5. 跨步状态递推方案

除了 fixed-window encoder，也可以进一步做跨步状态递推，即：

```text
obs_t, h_{t-1}
        ↓
LTC recurrent core
        ↓
h_t
        ↓
actor / critic head
```

形式上：

```text
h_t = LTC(h_{t-1}, o_t, a_{t-1})
a_t = π(h_t)
```

这时 LTC 的 hidden state 不再只存在于 frame stack 内部，而是在 rollout 中持续传递。

如果采用这个方向，需要相应支持：

```text
1. rollout 阶段维护 hidden state；
2. episode done / truncated 时重置 hidden state；
3. replay buffer 存连续 sequence；
4. 训练时 sample sequence batch；
5. 使用 burn-in 恢复序列初始 hidden state；
6. actor / critic 按序列计算 loss；
7. evaluation 阶段同样维护 hidden state。
```

这个方案更接近 full recurrent LTC-SAC，也更能体现 LTC 的连续时间动态特征。对于 LunarLander 这种简单环境，可以直接作为一个可训练验证的对比版本：效果好坏看实验结果，不需要预设结论。

---

## 6. 实验基础设施 TODO

### 并行环境采样 / 训练

当前实现和实验口径详见 [`parallel_sac_training.md`](parallel_sac_training.md)。

- [x] 先为当前单环境 SAC 基线建立可复现的 `VecEnv` 版本（`n_envs=1` 使用 `DummyVecEnv`，多环境使用 `SubprocVecEnv`），保持算法、评估环境和保存口径不变。
- [ ] 比较 `n_envs = 1, 2, 4, ...` 的 sample throughput、wall-clock、GPU 利用率、显存、final / best eval 和学习稳定性；不能只看 FPS。
- [x] `eval_freq` 与 checkpoint 频率统一按总 transition 计，并在 summary 中记录 `n_envs`、VecEnv 类型、内部 callback 频率、wall-clock 和吞吐率。
- [x] 每个并行 worker 使用 `seed + worker_index`，summary 记录全部 worker seed；评估使用不与 worker 重合的 `seed + n_envs` 单环境。

### 多环境 PPO

当前实现和 rollout 口径详见 [`parallel_ppo_training.md`](parallel_ppo_training.md)。

- [x] 在同一个 YAML 入口和训练生命周期中加入 PPO，不复制环境、评估或保存流程。
- [x] 加入 16 环境强基线，并严格校验完整 rollout 数和完整 minibatch。
- [x] summary 记录 rollout size、每 epoch minibatch 数、每轮 optimizer step 和数据复用 epoch 数。
- [ ] 用多个 seed 分别验证 CPU RL-Zoo 强基线与 CUDA 大网络 PPO 的学习稳定性。
- [ ] 分别比较 `n_envs=4/8/16` 的 wall-clock、throughput 与最终评估；16 环境强调轨迹多样性，不预设其吞吐一定最高。

### 随机 / Sobol / TPE 超参数搜索与调度（SAC / PPO）

- [ ] 使用 Optuna 建立统一的 trial 记录与 YAML 配置生成；采用 `RandomSampler`、Sobol `QMCSampler` 或 TPE sampler，不引入更复杂的模型驱动搜索。Sobol 适合小预算连续空间的均匀覆盖，随机采样适合混合离散/条件搜索空间，TPE 适合在已有 trial 结果后集中探索更有希望的区域。
- [ ] 使用 ASHA / Hyperband 进行多保真调度，而不是改变 SAC/PPO 的同步训练逻辑。SAC rung 使用 `100k → 250k → 500k` total transitions；PPO rung 必须是完整 rollout 数，使用 `262,144 → 524,288 → 1,048,576` transitions。
- [ ] SAC 搜索时固定 `n_envs=8`、`train_freq=1`、`gradient_steps=8`、replay-buffer 口径和正式预算；优先搜索 learning rate、`tau`、`gamma`、batch size、entropy coefficient，以及 RBF 的中心范围/初始带宽/最小带宽。6050/6255 容量匹配组只使用筛选出的候选配置复验，不进入首轮大搜索。
- [ ] PPO 搜索时保持 16 环境与完整 rollout/minibatch 约束；优先搜索 learning rate、`clip_range`、`gae_lambda`、`gamma`、`ent_coef`、`n_epochs`、batch size 和 MLP/RBF 容量。CPU `[64,64]` 基线与 CUDA `[400,300]` 实验分别建 study，不能混合吞吐或参数量结论。
- [ ] ASHA 的淘汰指标采用最近 3 次 deterministic evaluation 的平均回报，而不是单点 best reward；SAC 的首个 rung 必须大于 10k warmup，失败、NaN 和异常退出应显式记为失败 trial。
- [ ] 每个搜索 trial 先使用一个 seed；对每个算法/结构家族的前 3–5 个候选，以 3 个独立 seed、完整预算复验。MLP、LTC、RBF 都必须获得相同 trial 数与总 transition 调参预算，避免选择偏差。
- [ ] versioned summary 记录 sampler、pruner、rung、trial 参数、seed、final/best/last-3 evaluation、wall-clock、吞吐和失败原因；最终选型以多 seed 均值与离散性为准。

## 6. 推荐实验分支

可以把后续实验分成三条线：

```text
A. mlp baseline
B. fixed-window circuit LTC + residual / concat
C. recurrent LTC-SAC
```

其中 B 是对当前代码的低成本增强，C 是更完整的跨步状态递推版本。

建议优先比较：

```text
1. mlp
2. ltc
3. ltc_residual
4. ltc_residual_action
5. recurrent_ltc
```

主要观察：

- learning curve 上升速度；
- best model reward；
- final model 是否退化；
- 多 seed 方差；
- 训练 FPS；
- 参数量；
- 推理耗时。

---

## 7. 最终主线

当前最值得先实现的是：

```text
standard SB3 SAC
+ circuit LTC temporal encoder
+ raw residual / concat
+ optional action history
```

同时保留 recurrent LTC-SAC 作为进一步对比方案：

```text
sequence replay
+ hidden state carry
+ recurrent LTC actor / critic feature core
```

在 LunarLander 这种简单环境里，方案好坏可以直接通过训练结果判断。若 recurrent 版本明显提升学习速度或稳定性，再考虑把它迁移到更复杂的 USV/UAV 控制任务中。
