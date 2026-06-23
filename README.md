# SAC + LTC 控制实验记录

这个仓库用于学习和验证 Stable-Baselines3 SAC 在连续控制任务上的训练流程，并进一步探索把 LTC（Liquid Time-constant）时序特征引入 SAC policy / critic 的效果。当前重点环境是 `LunarLanderContinuous-v3`，`Pendulum-v1` 保留为快速入门 demo。

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

快速检查：

```bash
MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
conda run -n sac_sb3_demo python sac_lunarlander_ltc_compare.py \
  --config configs/smoke.yaml
```

正式配置在：

```text
configs/lunarlander.yaml
```

当前推荐正式实验是四组并行跑，并让 TensorBoard 指向单独 run tag，避免曲线混乱。TensorBoard run 名应保持扁平：

```text
mlp_1
ltc_1
ltc_res_1
ltc_act_1
```

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
  sac_lunarlander_ltc_compare.py \
  render_sac_lunarlander_gif.py \
  sac_experiments/lunarlander_compare.py \
  sac_experiments/lunarlander_common.py \
  sac_experiments/variants.py \
  sac_experiments/ltc_features.py
```

```bash
MPLCONFIGDIR=/tmp/matplotlib-sac-demo \
conda run -n sac_sb3_demo python -c \
  "from pathlib import Path; from sac_experiments.lunarlander_compare import load_config; c=load_config(Path('configs/lunarlander.yaml')); print(c.variants)"
```

期望：

```text
['mlp', 'ltc', 'ltc_residual', 'ltc_residual_action']
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
outputs/lunarlander/<run_tag>/<variant>/best_model/best_model.zip
outputs/lunarlander/<run_tag>/<variant>/final_model.zip
outputs/lunarlander/<run_tag>/<variant>/eval_summary.json
outputs/lunarlander/<run_tag>/<variant>/eval_logs/evaluations.npz
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

动作历史模型需要加：

```bash
conda run -n sac_sb3_demo python render_sac_lunarlander_gif.py \
  --model-path outputs/lunarlander/<run_tag>/ltc_residual_action/best_model/best_model.zip \
  --output-path outputs/lunarlander/<run_tag>/visualizations/ltc_act_best.gif \
  --action-history
```

GIF 用于直观看落地姿态、主发动机和侧向控制是否稳定，不能替代多 episode evaluation。

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

## 后续路线

短期：

- 完成四组 500k 对比。
- 汇总 `evaluations.npz`，生成多组 summary。
- 对 best/final model 生成 GIF。
- 跑多 seed，统计均值和方差。

中期：

- 加入 `ltc_residual_action` 的 action-history ablation。
- 比较 dense LTC、random sparse LTC、NCP-style sparse wiring。
- 评估参数量和推理耗时，避免只看 reward。

长期：

- 从 fixed-window LTC 进入 full recurrent LTC-SAC。
- 需要 sequence replay buffer、hidden state carry、episode reset、burn-in、sequence batch loss。
- 在 LunarLander 上验证后，再考虑迁移到 USV / UAV 等存在执行器延迟和输入惯性的控制任务。
