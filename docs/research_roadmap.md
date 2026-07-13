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

- [ ] 先为当前单环境 SAC 基线建立可复现的 `VecEnv` 版本（例如 `SubprocVecEnv`），保持算法、评估环境和保存口径不变。
- [ ] 比较 `n_envs = 1, 2, 4, ...` 的 sample throughput、wall-clock、GPU 利用率、显存、final / best eval 和学习稳定性；不能只看 FPS。
- [ ] 明确 vectorized callback 的 `eval_freq` 与 checkpoint 频率按环境步数还是总 transition 计，并在 summary 中记录 `n_envs`。
- [ ] 为每个并行 worker 设计可追溯但不同的随机种子，并验证 train / eval 环境仍然严格分离。

### 贝叶斯超参数搜索

- [ ] 选定轻量的贝叶斯优化工具（例如 Optuna），以 YAML 定义搜索空间、trial 预算、pruner 和输出目录。
- [ ] 优先搜索 learning rate、batch size、`tau`、learning starts、network 容量，以及 LTC 的 hidden size / ODE unfolds；先限制维度，避免盲目大搜索。
- [ ] 目标函数应同时考虑 deterministic eval、收敛速度、训练成本和失败率；单 seed 的偶然 best reward 不能直接作为结论。
- [ ] 将候选最优配置固定后，用独立 seeds 复验，并把 trial 参数、随机种子、wall-clock 与原始曲线写入版本化 summary。

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
