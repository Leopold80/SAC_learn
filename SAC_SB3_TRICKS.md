# SAC Demo 中新增的 Stable-Baselines3 / RL-Zoo Trick

本文档说明 `sac_sb3_pendulum_demo.py` 相比最朴素的 Stable-Baselines3 SAC 示例额外加入了哪些工程化设置，以及这些设置分别解决什么问题。

这些设置主要参考：

- Stable-Baselines3 官方 SAC 用法
- RL-Baselines3-Zoo 对 `Pendulum-v1` 的 tuned 配置

See `SAC_SB3_TRICKS.md` for the SB3/RL-Zoo tricks used by this demo.

## Demo 默认目标

当前 demo 面向 `Pendulum-v1`，目标不是只做 smoke test，而是训练出能明显优于随机策略的 SAC agent。

默认配置约定：

- 环境：`Pendulum-v1`
- 算法：Stable-Baselines3 `SAC`
- policy：`MlpPolicy`
- 默认训练步数：`20_000`
- 默认学习率：`1e-3`
- 默认评估 episode 数：`10`
- 默认评估间隔：`1_000` steps
- 默认输出目录：`outputs/sac_pendulum/`
- 默认 TensorBoard 日志目录：`runs/sac_pendulum/`

## 已采用的 Trick

### 1. RL-Zoo tuned hyperparameter

朴素 SB3 示例通常直接使用 SAC 默认参数，并只训练一个较短步数。这个 demo 采用 RL-Baselines3-Zoo 对 `Pendulum-v1` 的关键配置：

- `n_timesteps=20000`
- `learning_rate=1e-3`

作用：

- `20_000` steps 比官方最小示例更适合观察训练效果。
- `1e-3` 是 RL-Zoo 针对 `Pendulum-v1` 给出的 tuned learning rate，比盲目使用默认值更贴近该环境。

### 2. `Monitor` wrapper

训练环境和评估环境都使用 `Monitor` 包装。

作用：

- 记录每个 episode 的 reward 和 length。
- 让 SB3 的 logger、callback 和评估流程能拿到标准化的 episode 信息。
- 方便后续对训练曲线、checkpoint 和最终结果进行复查。

### 3. Train/eval 环境分离

demo 使用独立的 training environment 和 evaluation environment。

作用：

- 避免评估过程复用训练环境的内部状态。
- 让训练采样和效果评估职责分离。
- 更接近 RL-Zoo 和常见 RL benchmark 的实验写法。

### 4. Deterministic evaluation

评估时使用 deterministic action，而不是继续从随机策略中采样。

作用：

- SAC 训练时需要随机性和熵正则来探索。
- 评估时更关心当前策略均值动作的稳定表现。
- 降低评估结果方差，方便比较训练前后效果。

### 5. `EvalCallback`

训练过程中定期调用 `EvalCallback` 做评估。

作用：

- 每隔固定步数评估当前策略。
- 自动跟踪当前 best mean reward。
- 当策略刷新最好成绩时，保存 best model。
- 避免只保存最后一次模型，因为最后一次模型不一定是训练过程中表现最好的模型。

### 6. `CheckpointCallback`

训练过程中定期保存 checkpoint。

作用：

- 长训练中断后可以保留阶段性结果。
- 方便比较不同训练阶段的策略表现。
- 后续扩展到更长训练环境，例如 `LunarLanderContinuous` 时更有价值。

### 7. TensorBoard logging

demo 开启 TensorBoard 日志输出。

作用：

- 可视化 reward、loss、entropy coefficient、learning rate 等训练指标。
- 方便排查训练是否发散、过慢或无明显提升。
- 让 demo 结果可以从控制台输出升级为可视化实验记录。

### 8. Final model 和 best model 分开保存

demo 同时保存：

- final model：训练结束时的模型
- best model：评估过程中平均 reward 最好的模型

作用：

- final model 表示训练流程的最终状态。
- best model 表示评估指标上的最佳策略。
- 两者分开保存可以避免误把最后一步模型当成最好模型。

### 9. 训练前后 reward 对比

demo 在训练前先评估一次初始策略，训练后再评估一次 final model。

作用：

- 直接展示 SAC 学到了什么。
- 比只打印训练日志更适合作为 demo 验收标准。
- 可以快速判断当前超参数和训练步数是否有效。

### 10. JSON evaluation summary

demo 将关键评估结果保存为 JSON，例如：

- 环境名
- seed
- timesteps
- 训练前 mean/std reward
- 训练后 mean/std reward
- final model 路径
- best model 路径

作用：

- 便于复现实验结果。
- 方便后续脚本读取和汇总多次实验。
- 比只依赖控制台输出更可靠。

### 11. 隔离 conda 环境 `sac_sb3_demo`

不直接在控制科学研究公用环境 `cybernetic_env` 中安装 SB3 相关依赖，而是克隆出专用环境：

```bash
conda create -n sac_sb3_demo --clone cybernetic_env
conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
```

作用：

- 避免污染 `cybernetic_env`。
- 保留原有控制科学研究环境的依赖版本。
- 如果 SB3/Gymnasium 依赖解析带来问题，可以直接删除 `sac_sb3_demo`，不会影响公用环境。
- `tensorboard` 和兼容版 `setuptools<81` 也只安装到 `sac_sb3_demo`，用于支持 SB3 的 TensorBoard logging 和 TensorBoard CLI。

## 刻意没有加入的 Trick

### 1. 不启用 gSDE

gSDE 是 SB3 中用于连续控制任务的一种 generalized State-Dependent Exploration 方法。

本 demo 暂不启用，原因是：

- RL-Baselines3-Zoo 的 `Pendulum-v1` SAC 配置没有启用 `use_sde`。
- `Pendulum-v1` 足够简单，默认 SAC 随机策略和熵正则已经能完成有效探索。
- 先保持与 RL-Zoo Pendulum tuned 配置一致，减少不必要变量。

### 2. 不启用 `VecNormalize`

`VecNormalize` 可用于 observation/reward normalization。

本 demo 暂不启用，原因是：

- `Pendulum-v1` 的 observation 和 reward 尺度相对简单。
- RL-Zoo 的 `Pendulum-v1` SAC 配置没有要求 normalization。
- 加入 normalization 后需要额外保存和加载 normalization statistics，会让最小 demo 复杂不少。

### 3. 不加入 action noise

SAC 本身是 stochastic policy，并通过 entropy regularization 平衡探索和利用。

本 demo 暂不加入额外 action noise，原因是：

- SB3 SAC 默认不需要像 TD3/DDPG 那样显式配置 action noise。
- 额外 action noise 可能改变 SAC 原本的探索机制。
- 对 `Pendulum-v1` 来说没有必要。

### 4. 不渲染画面

demo 默认不调用 `render_mode="human"`。

原因是：

- 避免 GUI/display 依赖。
- 方便在服务器、远程 shell、CI 或无桌面环境中运行。
- 当前目标是训练效果和可复现实验记录，而不是可视化演示。

## 运行约定

为了避免污染 `cybernetic_env`，推荐流程是：

```bash
conda create -n sac_sb3_demo --clone cybernetic_env
conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
conda run -n sac_sb3_demo python sac_sb3_pendulum_demo.py
```

如果只想快速检查脚本是否能跑通，可以减少训练步数：

```bash
conda run -n sac_sb3_demo python sac_sb3_pendulum_demo.py --timesteps 1000 --eval-episodes 2 --eval-freq 500
```

正式观察效果时使用默认配置：

```bash
conda run -n sac_sb3_demo python sac_sb3_pendulum_demo.py
```

## 后续扩展：LunarLanderContinuous

`LunarLanderContinuous` 已列入后续计划，但本 demo 暂不实现。

后续扩展时需要重新考虑：

- Box2D 相关依赖安装。
- 更长训练步数，例如 RL-Zoo 中约 `500_000` steps 级别。
- 是否需要学习率 schedule。
- 是否需要更大的网络结构，例如 `[400, 300]`。
- checkpoint 和评估频率是否需要调整。
