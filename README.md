# LunarLander SAC + LTC 控制实验记录

这个仓库用于验证 Stable-Baselines3 SAC 在 `LunarLanderContinuous-v3` 上的训练流程，并进一步探索把 LTC（Liquid Time-constant）时序特征引入 SAC policy / critic 的效果。为保持实验边界清晰，仓库只保留 LunarLander 环境。

## 代码入口与文档

| 入口 / 文档 | 用途 |
|---|---|
| [`main.py`](main.py) | 唯一训练入口；读取 YAML 后调用统一训练流程。 |
| [`render_sac_lunarlander_gif.py`](render_sac_lunarlander_gif.py) | 根据保存模型的 observation space 自动匹配环境并输出 GIF。 |
| `configs/` | 正式、单帧 baseline、并行采样与 smoke YAML 配置。 |
| [`docs/architecture.md`](docs/architecture.md) | 模块职责、配置契约与推荐阅读路径。 |
| [`docs/parallel_sac_training.md`](docs/parallel_sac_training.md) | 并行采样架构、LaTeX 计数公式、replay buffer、seed、callback 与公平对比协议。 |
| [`docs/research_roadmap.md`](docs/research_roadmap.md) | LTC 设计说明与研究路线。 |
| [`docs/sac_implementations.md`](docs/sac_implementations.md) | SAC 框架选型笔记。 |
| [`docs/windows_migration.md`](docs/windows_migration.md) | Windows 复现实验说明。 |

## 研究目标

核心问题不是“能不能跑通 SAC”，而是：

- 标准 stacked-observation MLP SAC 在 LunarLander 上能达到什么水平；
- 公式版 circuit LTC 是否能提供更有用的短时序特征；
- raw observation residual / concat 是否能缓解 LTC 过度压缩 Markov 信息的问题；
- 过去动作历史是否能帮助 LTC 更好地建模控制输入和状态变化之间的关系；
- 后续是否值得进入 full recurrent LTC-SAC。

当前不使用 `ncps`，也不引入 NCP sparse wiring。它们后续可以作为权威参考或结构消融，但主线先保持自实现、可解释、可控。

## 当前对照分支

默认 LunarLander 实验比较四组：

| Variant | 思想 |
|---|---|
| `mlp` | 4 帧 stacked observation + SB3 默认 FlattenExtractor + MLP。作为强基线。 |
| `ltc` | 4 帧 observation 输入公式版 circuit LTC，再交给 SAC actor / critic。 |
| `ltc_residual` | raw stacked observation projection 与 circuit LTC feature concat，再 fusion。 |
| `ltc_residual_action` | 在 `ltc_residual` 基础上，让 LTC 分支额外看到 previous action history。raw residual 分支仍只看原始 observation。 |

`ltc_simple` 是 legacy 分支，用于复查早期简化 LTC 结果，不再作为默认主线。

## LTC 形式

公式版 circuit LTC 对应的核心形式是：

```text
dx_i/dt = -x_i / tau_i + sum_j f_ij(x_j, u; theta) * (A_ij - x_i)
```

当前实现中：

- `tau_i = softplus(raw_tau_i) + tau_min`
- `A_ij` 是可学习 reversal potential
- `f_ij` 是由当前 liquid state 和输入 observation 参数化的 sigmoid gate
- 使用 semi-implicit Euler update，提高数值稳定性
- 默认 `hidden_dim=128`、`features_dim=256`、`ode_unfolds=4`

`ltc_residual_action` 的动作历史设计是有意约束的：动作历史只进入 LTC encoder，不进入 raw residual branch。这样可以更清楚地观察“动作历史是否改善时序建模”，而不是让所有分支同时获得额外信息。

## SAC / SB3 Trick

LunarLander 主实验沿用 SB3 SAC 与 RL-Zoo 风格设置：

- `learning_rate = linear_schedule(7.3e-4)`
- `net_arch = [400, 300]`
- `buffer_size = 1_000_000`
- `batch_size = 256`
- `learning_starts = 10_000`
- `tau = 0.01`
- `gamma = 0.99`
- `ent_coef = "auto"`
- train / eval 环境分离
- deterministic evaluation
- `EvalCallback` 保存 best model
- `CheckpointCallback` 保存阶段 checkpoint
- TensorBoard logging
- final model 与 best model 分开保存
- JSON summary 记录可复查结果

刻意不启用：

- `gSDE`：先保持 SAC 默认随机策略与熵正则。
- `VecNormalize`：LunarLander 当前不先加入 normalization，减少变量。
- 额外 action noise：SAC 本身是 stochastic policy。
- GUI render：训练和评估默认适配 SSH / 服务器环境。

## 运行方式

使用隔离环境，避免污染 `cybernetic_env`：

```bash
conda create -n sac_sb3_demo --clone cybernetic_env
conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
```

默认正式对比：

```bash
conda run -n sac_sb3_demo python main.py
```

快速检查：

```bash
MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
conda run -n sac_sb3_demo python main.py --config configs/smoke.yaml
```

单帧 8 维 observation baseline：

```bash
conda run -n sac_sb3_demo python main.py --config configs/baseline.yaml
```

四进程并行采样 baseline：

```bash
conda run -n sac_sb3_demo python main.py --config configs/parallel_baseline.yaml
```

并行配置使用 `SubprocVecEnv`，每个 worker 使用不同 seed；评估仍是独立单环境。
`evaluation.frequency` 继续表示总 transition 数。四环境配置同步把
`gradient_steps` 设为 4，以维持单环境 baseline 约 1:1 的更新/样本比例。

`main.py` 只接受 `--config`；环境、算法、variant、训练参数、评估频率和输出路径全部写在 YAML 中。配置按 `experiment`、`environment`、`training`、`evaluation`、`output`、`sac` 和 `ltc` 分组。未知字段会直接报错，避免拼写错误被静默忽略。

训练器会按 `experiment.variants` 的顺序训练各组，而不是自行并行。若要并行启动多个单 variant 进程，必须为每个进程设置不同的 `output.run_tag`，避免模型和 TensorBoard 文件互相覆盖。TensorBoard run 名保持扁平：

```text
mlp_1
ltc_1
ltc_res_1
ltc_act_1
```

`output.run_tag: null` 会把首次运行直接写入 `output.directory` 与 `output.tensorboard_log`。训练器拒绝写入已有内容的运行目录，防止覆盖模型和评估结果；再次运行、正式实验和 multi-seed 实验必须设置新的单段安全 tag（字母、数字、点、下划线或连字符）。

SSH 端口转发：

```bash
ssh -L 6009:127.0.0.1:6009 <user>@<server>
```

浏览器打开：

```text
http://127.0.0.1:6009
```

## 验证方式

### 静态与配置检查

```bash
conda run -n sac_sb3_demo python -m py_compile \
  main.py \
  render_sac_lunarlander_gif.py \
  sac_experiments/config.py \
  sac_experiments/training.py \
  sac_experiments/lunarlander_common.py \
  sac_experiments/variants.py \
  sac_experiments/ltc_features.py
```

```bash
MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
conda run -n sac_sb3_demo python -c \
  "from pathlib import Path; from sac_experiments.config import load_config; c=load_config(Path('configs/lunarlander.yaml')); print(c.variants)"
```

期望：

```text
('mlp', 'ltc', 'ltc_residual', 'ltc_residual_action')
```

### Observation shape 检查

```bash
MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
conda run -n sac_sb3_demo python -c \
  "from sac_experiments.lunarlander_common import make_lunarlander_env; e=make_lunarlander_env(42, 4); print(e.observation_space.shape); e.close(); e=make_lunarlander_env(42, 4, use_action_history=True); print(e.observation_space.shape); e.close()"
```

期望：

```text
(4, 8)
(4, 10)
```

### TensorBoard 观察指标

重点看：

- `eval/mean_reward`：最重要的对比指标，评估环境 deterministic policy。
- `rollout/ep_rew_mean`：训练采样过程的平均回报，噪声更大。
- `rollout/ep_len_mean`：episode length，LunarLander 中长 episode 不一定坏，需结合 reward 看。
- `train/critic_loss`：过高或持续爆炸可能说明 Q 学习不稳定。
- `train/actor_loss`：不同模型间绝对值不可直接比较，主要看是否异常发散。
- `train/ent_coef`：自动熵系数，反映探索强度变化。
- `time/fps`：训练效率。LTC 分支显著慢于 MLP 是预期现象。

建议不要在 10k 或 20k steps 过早下结论，因为 `learning_starts=10_000`，真正梯度更新刚开始。更可靠的观察点是 100k、300k、500k，以及多 seed。

### 结果文件

每个 variant 应生成：

```text
outputs/lunarlander/[<run_tag>/]<variant>/best_model/best_model.zip
outputs/lunarlander/[<run_tag>/]<variant>/final_model.zip
outputs/lunarlander/[<run_tag>/]<variant>/eval_summary.json
outputs/lunarlander/[<run_tag>/]<variant>/eval_logs/evaluations.npz
```

`eval_summary.json` 会记录：

- variant
- seed
- timesteps
- feature extractor
- 是否使用 action history
- raw/action 维度
- best / final model path
- 训练前后 reward

### GIF 可视化

普通模型：

```bash
conda run -n sac_sb3_demo python render_sac_lunarlander_gif.py \
  --model-path outputs/lunarlander/<run_tag>/mlp/best_model/best_model.zip \
  --output-path outputs/lunarlander/<run_tag>/visualizations/mlp_best.gif
```

动作历史模型同样直接指定模型路径：

```bash
conda run -n sac_sb3_demo python render_sac_lunarlander_gif.py \
  --model-path outputs/lunarlander/<run_tag>/ltc_residual_action/best_model/best_model.zip \
  --output-path outputs/lunarlander/<run_tag>/visualizations/ltc_act_best.gif
```

渲染器会从模型保存的 observation space 自动识别单帧 / frame stack 与 action history；`--frame-stack` 和 `--action-history` 仅用于显式一致性校验。GIF 用于直观看落地姿态、主发动机和侧向控制是否稳定，不能替代多 episode evaluation。

## 当前判断标准

单次训练的判断优先级：

1. best eval mean reward
2. final eval mean reward
3. 300k 之后的稳定性
4. last10 eval mean / std
5. 训练 FPS 和推理复杂度
6. GIF 中的落地质量

研究结论必须多 seed 支撑。推荐 seeds：

```text
0, 1, 2, 3, 4
```

如果 `ltc_residual_action` 在多 seed 上同时提升 learning speed 和 best/final reward，才说明动作历史和 LTC 分支结合有稳定收益。

## TODO / Roadmap

这个 TODO 不是普通工程待办，而是当前 SAC + LTC 学习路线的研究备忘。优先级顺序是：先把实验统计做扎实，再扩展 LTC 结构，最后再进入真正 recurrent 的 SAC。

更完整的 LTC 结构说明见 [docs/research_roadmap.md](docs/research_roadmap.md)。

### 新增实验基础设施 TODO

- [ ] 完成并行环境对比实验：可复现 `VecEnv`、随机种子、callback / checkpoint 频率和 summary 口径已经实现；下一步比较 `n_envs=1/2/4` 的 sample throughput、wall-clock、显存和最终 eval。
- [ ] 加入贝叶斯超参数搜索：以独立 `run_tag`、固定搜索预算和多 seed 复验为前提，搜索 learning rate、batch size、`tau`、network / LTC 容量等；目标函数以中后期 deterministic eval 与训练成本共同定义，不能只选单次 best reward。

### P0: 先把当前 fixed-window 对照实验做扎实

- [ ] 完成四组默认 variant 的 500k steps 对比：`mlp`、`ltc`、`ltc_residual`、`ltc_residual_action`。
- [ ] 明确区分并保存 `final_model_eval` 和 `best_model_eval`，不要只依赖训练结束后的 final model。
- [ ] 从 `evaluations.npz` 汇总每个 variant 的 learning curve、best eval、last-N eval mean/std。
- [ ] 对 best / final model 分别生成 GIF，用于检查落地姿态、推力抖动和失败模式。
- [ ] 记录 wall-clock time、FPS、参数量和推理复杂度，避免只看 reward。

### P1: Multi-seed 统计

- [ ] 把单 seed 配置扩展为 multi-seed runner，例如 `seeds: [0, 1, 2, 3, 4]`。
- [ ] 每个 seed 使用独立 `run_tag` 或独立输出目录，避免 TensorBoard 和模型文件互相覆盖。
- [ ] 汇总 `best_eval_mean_reward`、`final_eval_mean_reward`、`last10_eval_mean`、`last10_eval_std`。
- [ ] 输出 `mean ± std` 表格，并保留每个 seed 的原始结果。
- [ ] 只有当某个 variant 在多 seed 上稳定提高 reward 或收敛速度时，才把它视为有效结构，而不是单次训练偶然结果。

### P2: Capacity-matched LTC / MLP 对照

当前公式版 circuit LTC 的默认规模可能偏大：`liquid_hidden_dim=128`、`features_dim=256`、`raw_features_dim=128`、`fusion_hidden_dim=256`、`ode_unfolds=4`。由于 circuit LTC 包含近似 `H × H` 的 pairwise gate，当前对比可能混合了两个因素：一是 LTC 是否提供有用的时序归纳偏置，二是 LTC 系列是否只是因为网络容量和计算量更大而表现不同。因此后续需要加入 capacity-matched 对照，让带时序网络的算法和基准 MLP 尽量处于相近规模。

- [ ] 增加 `capacity_matched_ltc` 配置，例如 `liquid_hidden_dim=32`、`features_dim=64`、`raw_features_dim=32`、`fusion_hidden_dim=64`、`ode_unfolds=1`。
- [ ] 增加稍强的 `capacity_matched_ltc_mid` 配置，例如 `liquid_hidden_dim=48`、`features_dim=96`、`raw_features_dim=48`、`fusion_hidden_dim=96`、`ode_unfolds=1`。
- [ ] 保持 SAC actor / critic 后端 `net_arch=[400, 300]` 不变，优先只控制 feature extractor 的规模，避免同时改动太多变量。
- [ ] 记录并比较 MLP 与 capacity-matched LTC 的参数量、`time/fps`、wall-clock time、`eval/mean_reward`、`last10_eval_std` 和 `critic_loss`。
- [ ] 如果 capacity-matched LTC 仍能接近或超过 MLP，说明时序结构可能有独立价值；如果只有大规模 LTC 有优势，则需要谨慎区分结构收益和容量收益。

### P3: Dense H×H circuit LTC 的稀疏化

当前 circuit LTC 使用近似全连接的 `H × H` 液态连接。`hidden_dim=128` 时仍可接受，但计算复杂度和参数规模会随 hidden size 二次增长。因此后续需要加入 sparse / circuit mask 做结构消融。

- [ ] 增加 `connection_mask`，让 gate 和 reversal potential 只在 mask 指定的连接上生效。
- [ ] 比较 dense LTC、random sparse LTC、block sparse LTC、local/ring sparse LTC。
- [ ] 保留 NCP-style sparse wiring 作为参考路线，但不要一开始就依赖 `ncps`，优先保持自实现结构可解释。
- [ ] 统计不同 sparsity ratio 下的 reward、FPS、参数量和显存占用。
- [ ] 检查 sparse mask 是否改变训练稳定性，而不仅仅是降低计算量。
- [ ] 如果 sparse circuit 的性能接近 dense circuit，但速度更快，则优先保留 sparse 版本作为后续 recurrent 主线候选。

### P4: Action-history 与时序建模消融

- [ ] 对 `ltc_residual_action` 增加更细的 action-history ablation。
- [ ] 比较 previous action 只进入 LTC branch、同时进入 raw branch、完全不进入网络三种设置。
- [ ] 比较不同 frame stack 长度，例如 2、4、8。
- [ ] 检查动作历史是否主要改善 early learning，还是改善最终 best reward。
- [ ] 关注失败模式：动作历史可能帮助建模输入惯性，也可能让网络过拟合短窗口相关性。

### P5: Recurrent SAC / full recurrent LTC-SAC

当前实现本质上仍是 fixed-window encoder：用 frame stack 伪造短时序输入，然后输出一个 feature vector。真正的 recurrent 版本需要改变 replay 和训练逻辑，不能只把 feature extractor 换成 RNN。

- [ ] 调研 SB3 SAC 是否适合直接扩展 recurrent policy；如果不适合，考虑 SB3-Contrib、CleanRL、Tianshou、TorchRL 或自写最小 recurrent SAC。
- [ ] 实现 sequence replay buffer，支持按 episode 采样连续片段。
- [ ] 处理 hidden state carry、episode reset、done mask 和 truncated mask。
- [ ] 加入 burn-in，让 hidden state 先用历史片段预热，再在后续片段上计算 loss。
- [ ] 明确 actor、critic、target critic 的 recurrent state 如何同步和截断反传。
- [ ] 比较 fixed-window LTC-SAC 与 recurrent LTC-SAC 的收益，判断 recurrent 是否真的必要。
- [ ] 在 LunarLander 验证稳定后，再迁移到更符合控制背景的任务，例如带输入延迟、执行器滞后或部分可观测状态的 USV / UAV 控制环境。

### P6: 最终可能形成的研究问题

- [ ] LTC 是否只是增加了网络容量，还是确实提供了有用的动态记忆？
- [ ] residual branch 是否是 SAC 中使用 LTC 的必要稳定化结构？
- [ ] previous action history 是否能显著改善连续控制中的输入-状态动态建模？
- [ ] sparse circuit mask 能否在基本不损失 reward 的情况下提高训练和推理效率？
- [ ] fixed-window LTC 和 full recurrent LTC-SAC 的收益边界在哪里？
