# 单卡超参数搜索：TPE、ASHA、统计复验与正式训练

本章完整说明本仓库 `experiments/hyperparam` 分支中的超参数搜索基础设施。重点不是罗列
几条命令，而是解释每个阶段为什么存在、实际执行了什么、产生什么证据、何时停止，以及
上一阶段的结果如何进入下一阶段。

本文所说的“候选”都是**同一算法、同一 variant、同一网络结构下的不同超参数配置**。
例如，三个 SAC 候选可能具有不同的学习率、折扣因子和 batch size，但它们不是三个不同
算法。一个 study 不混合 SAC 与 PPO，也不同时比较 MLP、RBF 和 LTC。

## 1. 系统最终要解决什么问题

给定超参数配置 $\theta$ 和训练 seed $s$，一次强化学习训练得到的最终回报可写成：

$$
R(\theta, s) = \mu(\theta) + \epsilon(\theta, s).
$$

$\mu(\theta)$ 是真正关心的期望性能，$\epsilon(\theta,s)$ 则包含网络初始化、环境随机性、
采样轨迹和优化过程带来的噪声。单次训练只能观测一个带噪声的样本，因此不能把“某一条
曲线最高”直接解释成“该超参数更好”。

另一方面，如果对搜索空间里的每个配置都立即运行九个完整 seed，单张 RTX 4070 的成本
会非常高。系统因此把任务拆成四个职责清晰的阶段：

| 阶段 | 主要功能 | 是否完整训练 | 是否保存模型 | 能否作为最终结论 |
|---|---|---:|---:|---:|
| TPE 采样 | 决定下一组值得尝试的超参数 | 取决于 ASHA | 否 | 否 |
| ASHA 剪枝 | 尽早停止当前比较中明显落后的 trial | 落后者否 | 否 | 否 |
| 自动统计复验 | 用独立的共同 seed 比较少量优秀配置 | 是，多次 | 否 | 可以选出配置 |
| 正式训练 | 使用冠军配置训练需要长期保留的模型 | 是 | 是 | 生成最终模型产物 |

最重要的边界是：**TPE 和 ASHA 负责经济地找候选，自动复验负责验证候选的稳定性，正式
训练负责产出要保存和使用的模型。**

## 2. 基础配置与搜索配置各管什么

普通训练 YAML 描述一个完整、可直接运行的实验，包括：

- 算法是 SAC 还是 PPO；
- variant、网络结构和环境包装；
- 环境 worker 数；
- 训练总 transition 数；
- 过程评估频率和评估 episode 数；
- 算法超参数；
- 输出、checkpoint 和 TensorBoard 位置。

搜索 YAML 不重新定义整个实验，而是引用一份已经合法的普通训练 YAML，然后只声明允许
改变的算法标量参数。例如：

```yaml
base_config: ../sac/parallel/parallel_8env.yaml

study:
  name: sac_mlp_4070_balanced
  output_root: outputs/hyperparameter_search
  n_trials: 24
  seed: 42
  n_startup_trials: 8

pruner:
  min_resource: 100000
  reduction_factor: 2

parameters:
  sac.learning_rate:
    type: float
    low: 0.0001
    high: 0.001
    log: true
  sac.batch_size:
    type: categorical
    choices: [128, 256, 512]

revalidation:
  top_k: 3
  seed_base: 1000
  min_seeds: 5
  max_seeds: 9
  evaluation_episodes: 50
```

这种分层有两个目的。第一，普通训练与搜索 trial 走相同的 `load_config()` 校验和统一训练
入口，搜索器不会形成第二套隐蔽训练逻辑。第二，一个 study 中保持算法、网络、worker 数
和训练预算不变，trial 之间的差异才能主要归因于搜索参数。

参数路径只能指向算法配置中的标量值，例如 `sac.tau`、`sac.ent_coef`、
`ppo.clip_range` 或 `ppo.n_epochs`。环境、总预算、输出目录、variant 和网络结构不能在
同一个 study 中漂移。若确实要比较 MLP 与 LTC，或比较 8 worker 与 16 worker，应创建
两个独立 study，再使用共同实验协议比较，而不是把它们塞进一次 TPE 搜索。

## 3. 正式搜索开始前发生什么

执行以下命令时：

```bash
conda run -n sac_sb3_demo python main.py \
  --search-config configs/search/sac_mlp_4070_balanced.yaml
```

程序在创建第一个 trial 前完成以下工作。

### 3.1 解析并验证两层 YAML

搜索器先读取搜索 YAML，解析 `base_config` 的相对路径，再加载基础训练 YAML。随后检查：

1. study 名称和输出根目录是否合法；
2. trial 数、TPE 启动 trial 数、ASHA 参数和复验参数是否为有效正整数；
3. 每个搜索参数的路径是否存在、类型是否匹配；
4. 基础配置是否只包含一个 variant；
5. 训练总 transition 是否满足原有 SAC/PPO 约束；
6. ASHA 资源节点是否与评估频率、环境数和 PPO rollout 对齐。

任何一项不满足都会在耗费训练资源前报错。

### 3.2 建立不可混用的 study 身份

新 study 会写入 `study_manifest.json`，其中包括：

- schema 版本；
- study 名称；
- 基础 YAML 和搜索 YAML 的路径及 SHA-256；
- 当前 Git HEAD；
- Python 和 Optuna 版本；
- TPE seed 与启动 trial 数；
- ASHA 的最小资源、缩减因子和实际资源节点；
- 搜索目标的定义。

恢复运行时会比较 schema、study 名称及两份 YAML 的哈希。配置发生变化后，程序拒绝把
新结果写入旧 study，避免“同名数据库中实际包含两套实验条件”。

### 3.3 记录目标机器信息

程序写入 `hardware_snapshot.json`，记录：

- 可见逻辑 CPU 核数；
- 配置要求的环境 worker 数；
- 请求的设备；
- CUDA 是否可见；
- GPU 名称、数量与总显存；
- PyTorch 和 CUDA 版本。

如果环境 worker 过多，导致 learner 和操作系统几乎没有 CPU 余量，程序给出警告，但不会
偷偷修改 YAML。硬件选择仍由实验者控制。

## 4. TPE 的功能与一次采样过程

TPE 是**参数建议器**。它回答的问题是：“在还剩一次 trial 预算时，下一组参数应该取
什么值？”

它不负责训练模型，不决定 trial 在中途是否停止，也不能证明最终候选具有统计优势。

### 4.1 启动探索

默认配置中的 `n_startup_trials: 8` 表示前八个 trial 主要用于建立初始观察。此时历史
数据不足，TPE 还不能可靠判断哪些区域更有希望，因此先覆盖不同的参数组合。

启动 trial 不是正式复验，也不要求每个 trial 都完成全预算；它们仍受 ASHA 管理。

### 4.2 历史引导采样

有了足够的已完成或已上报 trial 后，TPE 根据“参数取值”和“目标值”的关系，把搜索历史
划分为相对优秀和相对普通的观察，并优先建议更可能落入优秀区域的参数。

直观上，它不是在固定网格中逐点穷举，而是一边训练一边更新对有希望区域的判断。它仍会
保留探索性，因此不会永远重复当前最好的一组数值。

### 4.3 参数类型

当前搜索器支持：

- `float`：连续浮点区间；
- `int`：离散整数区间；
- `categorical`：从显式候选列表中选择；
- 对正数跨度较大的参数使用 `log: true`，例如学习率。

学习率从 $10^{-4}$ 到 $10^{-3}$ 时，使用对数尺度比在线性坐标中采样更符合优化问题的
数量级特征。`batch_size`、`ent_coef` 模式等不适合连续插值的参数则使用 categorical。

### 4.4 TPE 的输出

TPE 对每个 trial 产生一组具体参数，例如：

```text
sac.learning_rate = 0.00037
sac.tau           = 0.0081
sac.gamma         = 0.991
sac.batch_size    = 256
sac.ent_coef      = auto_0.1
```

程序将这些值覆盖到基础 YAML 的副本中，同时写入统一的搜索训练 seed 和该 trial 独立的
输出目录，生成：

```text
trials/trial-0007/resolved_config.yaml
```

这份文件随后再次经过普通 `load_config()` 校验，并交给统一训练函数执行。也就是说，
TPE 只提出参数，真正的环境构造、模型构造、训练和评估仍由仓库原有训练架构负责。

## 5. ASHA 的功能与剪枝过程

ASHA 是**训练预算分配器**。它回答的问题是：“当前 trial 已经训练到某个资源节点，
根据目前表现，是否值得继续投入后续预算？”

它不决定下一组参数，也不负责最终统计复验。

### 5.1 资源的含义

本实现中的 resource 是模型已经采样的总 transition 数，而不是 wall-clock 秒数。这样做
可以避免因为 CPU 抢占、磁盘抖动或不同机器速度而改变比较口径。

对于：

```yaml
pruner:
  min_resource: 100000
  reduction_factor: 2
```

如果基础训练预算足够大，资源节点会依次是 100000、200000、400000……直到不超过完整
训练预算。这里的数值来自 YAML 和训练预算，并没有把某组历史文档数字硬编码为标准答案。

### 5.2 trial 如何上报中间成绩

训练过程使用确定性评估回调。每到 `evaluation.frequency`，评估环境执行配置规定数量的
episode，得到一次平均回报。搜索器维护最近三次评估均值：

$$
J_t = \frac{1}{3}\sum_{k=t-2}^{t}\bar R_k.
$$

只有同时满足以下条件时才向 ASHA 上报：

1. 已经积累至少三次有限的评估结果；
2. 当前总 transition 数恰好位于一个 ASHA 资源节点。

使用最近三次的均值，是为了降低一次偶然高峰或低谷对剪枝决定的影响。它仍然只是廉价的
搜索指标，不等于独立多 seed 结论。

### 5.3 为什么资源节点必须严格对齐

所有节点必须是 `evaluation.frequency` 的倍数，否则到达节点时可能根本没有新评估；
也必须是 `environment.n_envs` 的倍数，从而保持总 transition 口径。

PPO 还有额外约束：节点必须是 `n_envs * ppo.n_steps` 的倍数。PPO 先收集完整 rollout，
再计算 GAE 并进行 minibatch 更新；在半个 rollout 中比较会破坏算法自身的更新边界。

### 5.4 剪枝决定

trial 到达资源节点并上报 $J_t$ 后，Optuna 的 `SuccessiveHalvingPruner` 将它与 study
中可比较的历史中间结果进行比较。相对落后的 trial 会抛出 `TrialPruned` 并立即停止，
状态记为 `PRUNED`；值得继续的 trial 进入下一个资源区间。

最后一个完整预算节点不会再触发剪枝，因为此时训练预算已经消耗完毕。

### 5.5 三种 trial 终态

| 状态 | 含义 | 是否可进入复验候选 |
|---|---|---:|
| `COMPLETE` | 完成基础 YAML 规定的全部训练预算，并产生有限目标值 | 是 |
| `PRUNED` | 在 ASHA 节点被认为暂不值得继续 | 否 |
| `FAIL` | 配置、环境、数值或运行时异常导致失败 | 否 |

搜索目标值是 trial 结束时最近三次确定性过程评估的均值。它用于搜索排序，但不是最终
champion 的选型分数。

## 6. TPE 与 ASHA 如何共同完成粗筛

二者是串行协作关系：

1. TPE 为一个新 trial 建议参数；
2. 统一训练入口从头创建模型并开始训练；
3. 评估回调持续收集最近三次评估；
4. 到资源节点后，ASHA 判断是否继续；
5. trial 结束后，其参数、状态和成绩写入 study；
6. TPE 在下一次建议时使用更新后的 study 历史。

因此，“TPE + ASHA 是自动复验前的粗筛选”这个理解是正确的，但还可以更精确：

- TPE 提高**参数采样效率**；
- ASHA 提高**单个 trial 的预算使用效率**；
- 二者共同输出一小组完成全预算、搜索指标较高的候选；
- 它们不会替代独立 seed 的统计复验。

默认单张 GPU 同时只运行一个 trial。多 trial 并发会让环境 worker、GPU allocator 和磁盘
I/O 互相竞争，也会让 wall-clock 与显存数据难以解释。

## 7. 搜索阶段保存什么

搜索过程中故意不保存模型权重、checkpoint、TensorBoard 和 monitor 日志，以避免大量
被剪枝 trial 占满磁盘。每个 trial 保留足以审计搜索行为的轻量产物：

```text
trials/trial-0007/
  resolved_config.yaml
  trial_result.json
  outputs/.../eval_logs/evaluations.npz
  outputs/.../eval_summary.json
  outputs/.../experiment_summary_*.json
```

`trial_result.json` 会记录：

- trial 编号和终态；
- TPE 采样的参数；
- ASHA 节点及上报值；
- 完成 trial 的最终搜索目标；
- 失败或剪枝原因；
- CUDA 可用时的峰值显存统计。

study 根目录还保存 SQLite 数据库、sampler 状态、JSON 汇总和 CSV 表格。每个 trial
结束后都会保存 sampler 状态，便于中断后继续。

## 8. 中断恢复时发生什么

使用：

```bash
conda run -n sac_sb3_demo python main.py \
  --search-config configs/search/sac_mlp_4070_balanced.yaml --resume
```

恢复过程会：

1. 读取并验证原 study manifest；
2. 打开原 SQLite study；
3. 恢复保存的 TPE sampler 状态；
4. 统计数据库里已经存在的 trial 数；
5. 继续创建 trial，直到数据库中的 trial 总数达到 `study.n_trials`；
6. 重写当前 `study_summary.json` 和 `study_trials.csv`。

这里的 trial 总数包含 `COMPLETE`、`PRUNED` 和 `FAIL`。恢复不是“保证凑够 24 个完整
trial”，而是“把预定的 24 次 trial 尝试执行完”。如果失败 trial 异常偏多，应先查明
环境或配置错误，再用新的 study 名称开展正式搜索。

直接对非空 study 目录运行而不加 `--resume` 会被拒绝。修改 YAML 后也不能恢复旧 study；
应修改 `study.name`，让新实验使用新的目录和数据库。

## 9. 自动统计复验的完整过程

搜索结束后，显式执行：

```bash
conda run -n sac_sb3_demo python main.py \
  --search-config configs/search/sac_mlp_4070_balanced.yaml --revalidate
```

复验不是继续训练搜索中的模型。搜索阶段没有保留那些模型权重，而且搜索 seed 与搜索
过程指标本来就不适合作为最终证据。复验会把候选配置逐个**从头完整训练**。

### 9.1 前置条件

程序以恢复模式打开原 study，并要求至少存在 `revalidation.top_k` 个合法 `COMPLETE`
trial。默认 `top_k: 3`，因此至少需要三个完成全预算且目标值有限的 trial。

`PRUNED` 和 `FAIL` trial 不会进入候选集。三个候选按搜索目标值从高到低选出，但这个
排名只负责确定谁值得复验，不直接决定冠军。

### 9.2 共同独立 seed

默认训练 seed 从 `seed_base: 1000` 开始。前三个候选都先使用完全相同的五个 seed：

```text
1000, 1001, 1002, 1003, 1004
```

“共同”意味着每个候选面对相同的 seed 条件，便于做配对比较；“独立”意味着这些 seed
没有用于搜索阶段，降低搜索过程对最终评估的污染。

每个“候选 × seed”都是一次从随机初始化开始的完整训练，不会继承搜索 trial 的参数状态、
优化器状态或 replay buffer。

### 9.3 完整预算训练

复验训练使用基础 YAML 中原定的全部 transition 预算。ASHA 不再参与，候选不会因为早期
曲线较差而被中途停止。

这样做是因为复验目标已经从“节约大规模搜索成本”变成“公平比较少数候选的最终表现”。
早期学习速度较慢但最终表现较好的配置，不应在这个阶段被剪掉。

### 9.4 held-out 最终评估

每次完整训练结束后，程序创建一个单独的最终评估环境。其 seed 为：

$$
s_{\text{eval}} = s_{\text{train}} + 1{,}000{,}000.
$$

默认执行 50 个 deterministic episode，记录其平均回报作为该候选在该训练 seed 下的
最终分数。这个评估环境没有参与训练，也不同于训练过程中周期性使用的评估环境。

因此，一条复验分数对应：

```text
固定候选参数
+ 一个独立训练 seed
+ 完整训练预算
+ 一个隔离的 held-out 环境 seed
+ 50 个 deterministic episode 的平均回报
```

增加评估 episode 数主要降低“最终测量本身”的噪声；增加训练 seed 数主要降低“训练过程
随机性”的不确定性。两者不能互相替代。

### 9.5 为什么使用单侧 95% LCB

对候选 $i$ 的 $n$ 个复验分数，程序计算样本均值 $\bar R_i$、样本标准差 $s_i$ 和单侧
95% Student-t 下置信界：

$$
\operatorname{LCB}_{0.95}(i)
=
\bar R_i
-
t_{0.95,n-1}\frac{s_i}{\sqrt n}.
$$

LCB 可以理解为“考虑当前样本量与波动后，对候选稳定性能的保守估计”：

- 均值高会提高 LCB；
- seed 间波动大，会降低 LCB；
- seed 数增加会降低标准误，缩小不确定性；
- 小样本使用 Student-t，而不是假装总体方差已知。

冠军按最大 LCB 选择，而不是按最大单次 reward、最大曲线峰值或最高样本均值选择。

### 9.6 配对比较如何决定是否已经区分

因为候选使用相同 seed，可以对领先者与挑战者逐 seed 计算差值：

$$
D_s = R(\theta_{\text{leader}},s)-R(\theta_{\text{challenger}},s).
$$

程序再计算差值样本的单侧 95% 下置信界。如果下界大于零，说明在当前规则下，领先者相对
该挑战者已经具有明确的正向差异；该挑战者无需继续增加 seed。

如果下界小于或等于零，只能说现有证据尚不能明确区分，不能靠肉眼观察曲线宣布领先。

### 9.7 从五个 seed 自适应增加到九个

程序先让全部三个候选完成五个共同 seed，然后：

1. 选出当前 LCB 最高的候选作为临时领先者；
2. 对它与其他候选执行配对下置信界比较；
3. 已被明确压过的候选停止增加 seed；
4. 临时领先者和仍无法区分的候选共同增加一个新 seed；
5. 重新计算 LCB 与配对差异；
6. 重复直到只剩一个未被排除的候选，或达到九个 seed。

这是一种自适应计算分配：明显较差的候选不继续消耗完整训练预算，接近的候选得到更多
证据。默认最大训练次数的上界是：

$$
\text{top\_k}\times\text{max\_seeds}=3\times9=27
$$

次完整训练；如果较早区分，实际次数会更少。

### 9.8 `statistically_separated` 的含义

如果在达到最大 seed 数之前，临时领先者相对所有仍在比较的候选，其配对差值下置信界都
大于零，则：

```json
"statistically_separated": true
```

如果九个 seed 后仍有多个候选无法明确区分，系统仍需要给后续自动化一个唯一配置，因此
选择剩余候选中 LCB 最高者写入 `champion.yaml`，同时标记：

```json
"statistically_separated": false
```

`false` 不表示冠军一定不好，也不表示所有候选完全相同。它只表示：在当前参数空间、
训练预算、seed 上限和 95% 规则下，证据不足以声明冠军与所有近邻候选已经明确分离。

### 9.9 复验保存什么、不保存什么

每个候选、每个 seed 会保留：

- 解析后的完整 YAML；
- 训练与 held-out 评估摘要；
- `revalidation_result.json`；
- 该 seed 的最终分数和评估条件。

复验阶段依然不保存模型、checkpoint、TensorBoard 或 monitor 日志。原因是复验需要执行
多次完整训练，其目标是**选择配置**，不是保留 15 到 27 份中间模型。

根目录最终生成：

- `revalidation_summary.json`：候选参数、搜索分数、逐 seed 分数、均值、标准差、上下
  置信界、最终活跃状态、冠军编号和显著区分标记；
- `champion.yaml`：按上述规则选出的冠军超参数配置。

如果 `revalidation_summary.json` 已经存在，再次执行 `--revalidate` 会直接返回现有结果。
如果上一次复验中途终止并留下部分目录，程序拒绝覆盖，以免把两次不完整 seed 集悄悄
拼接。当前策略是保留现场以便检查，并使用新的 study 名称重新执行复验。

## 10. 复验之后：正式训练发生什么

自动复验完成时，三个候选中的若干配置已经被多 seed 完整训练过，但这些训练只用于统计
比较，模型权重没有保存。此时系统不会自动继续占用 GPU。

默认后续动作是显式使用 `champion.yaml` 启动普通训练：

```bash
conda run -n sac_sb3_demo python main.py \
  --config outputs/hyperparameter_search/<study_name>/champion.yaml
```

这次训练回到普通持久化模式，会：

1. 使用冠军超参数和 YAML 中指定的训练 seed；
2. 创建正式输出目录；
3. 执行训练前评估、完整训练和周期评估；
4. 保存 best model、checkpoint 和 final model；
5. 保存 TensorBoard、评估曲线和实验摘要；
6. 在训练结束后重新加载 final model 做最终评估。

所以“后续就是几个优秀算法的直接训练”需要稍作修正：

- 它们是几个优秀的**超参数配置**；
- 自动复验期间，它们已经被多 seed 直接完整训练，但不保存模型；
- 复验结束后，系统默认只推荐 `champion.yaml` 做一次可持久化正式训练；
- runner-up 不会自动再训练，也不会自动保存；
- 如果研究目标要求保留前三名模型，需要为每个候选生成独立普通 YAML，并设置不同的
  seed、输出目录或 `output.run_tag`，再逐个显式启动。

同一份 `champion.yaml` 的正式输出目录非空后不能直接重复运行。若要做多次正式模型训练，
应复制配置并设置唯一 `output.run_tag` 或输出位置，避免覆盖已有权重和日志。

## 11. 默认 RTX 4070 平衡档

`configs/search/*_4070_balanced.yaml` 面向 Ubuntu 单张 RTX 4070。它不是最大吞吐档，
而是 GPU 使用、CPU worker、搜索覆盖、复验成本和磁盘占用之间的默认起点。

| 项目 | SAC | PPO |
|---|---:|---:|
| 基础配置 | 8 环境 SAC MLP | 16 环境 CUDA PPO MLP |
| 并行 trial | 1 | 1 |
| 默认 trial 尝试数 | 24 | 24 |
| TPE 启动 trial | 8 | 8 |
| 搜索 seed | 42 | 42 |
| 复验候选 | top 3 | top 3 |
| 初始/最大复验 seed | 5 / 9 | 5 / 9 |
| 每个 seed 的 held-out episode | 50 | 50 |
| 搜索与复验模型持久化 | 否 | 否 |

SAC 的 8 个环境与 8 次梯度更新保留约 1:1 的梯度更新/transition 口径。PPO 使用现有
CUDA 大网络配置，并保持完整 rollout 和 minibatch 约束。4070 只是该档位的目标硬件；
worker 数和网络容量仍应根据 Ubuntu 主机的实际 CPU、显存占用及吞吐数据复查。

## 12. 完整命令工作流

### 12.1 准备隔离环境

```bash
conda create -n sac_sb3_demo --clone cybernetic_env
conda run -n sac_sb3_demo python -m pip install -r requirements-sac-demo.txt
```

确认 Ubuntu 主机能够看到目标 GPU：

```bash
nvidia-smi
conda run -n sac_sb3_demo python -c \
  "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

### 12.2 运行 SAC 搜索

```bash
conda run -n sac_sb3_demo python main.py \
  --search-config configs/search/sac_mlp_4070_balanced.yaml
```

### 12.3 中断后恢复

```bash
conda run -n sac_sb3_demo python main.py \
  --search-config configs/search/sac_mlp_4070_balanced.yaml --resume
```

### 12.4 显式启动统计复验

```bash
conda run -n sac_sb3_demo python main.py \
  --search-config configs/search/sac_mlp_4070_balanced.yaml --revalidate
```

### 12.5 使用冠军配置正式训练

```bash
conda run -n sac_sb3_demo python main.py \
  --config outputs/hyperparameter_search/sac_mlp_4070_balanced/champion.yaml
```

PPO 使用相同命令结构，只需把搜索配置替换为：

```text
configs/search/ppo_mlp_4070_balanced.yaml
```

## 13. 产物目录与推荐阅读顺序

```text
outputs/hyperparameter_search/<study>/
  hardware_snapshot.json
  study_manifest.json
  study.sqlite3
  sampler_state.pkl
  study_summary.json
  study_trials.csv
  trials/
    trial-0000/
      resolved_config.yaml
      trial_result.json
      outputs/...
  revalidation/
    candidate-0003/
      seed-1000/
        resolved_config.yaml
        revalidation_result.json
        outputs/...
  revalidation_summary.json
  champion.yaml
  champion_training/
```

建议按以下顺序阅读：

1. `hardware_snapshot.json`：确认实验确实运行在预期硬件和 CUDA 条件下；
2. `study_manifest.json`：确认搜索协议、配置哈希和资源节点；
3. `study_summary.json`：检查 COMPLETE、PRUNED、FAIL 数量和 trial 排名；
4. `trial_result.json`：检查特定 trial 的参数、节点成绩和终止原因；
5. `revalidation_summary.json`：查看各候选逐 seed 分数、LCB 和是否明确区分；
6. `champion.yaml`：确认最终正式训练要使用的完整配置；
7. 正式训练输出：检查最终模型、TensorBoard、评估曲线和摘要。

`study.sqlite3` 和 `sampler_state.pkl` 属于本地可变恢复状态，不应作为人工阅读的主要
结论，也不应在运行中手工编辑。

## 14. 常见误解与判断边界

- **TPE 不是最终裁判。** 它只根据已有噪声观测提高下一次采样效率。
- **ASHA 剪掉不等于理论上永远较差。** 它表示该 trial 在当前 study、当前节点和当前
  比较规则下不值得继续占用预算。
- **搜索排名不是冠军排名。** 搜索使用一个共同搜索 seed 和过程评估指标；冠军使用独立
  多 seed held-out 结果的 LCB。
- **50 个 episode 不是 50 个训练 seed。** 它们降低单个模型的测量噪声，不能替代多个
  随机初始化。
- **复验已经进行了完整训练，但没有最终模型。** 这些训练用于估计配置稳定性，模型权重
  被有意丢弃。
- **`statistically_separated: false` 不是失败。** 它是对证据强度的诚实描述；此时
  `champion.yaml` 仍是当前规则下的保守首选。
- **最高 best reward 不是选型规则。** 它容易受到多次观察后的选择偏差影响。
- **smoke reward 不是研究结论。** smoke 只验证 YAML、回调、SQLite、训练调度和资源
  清理路径。
- **LCB 最高不等于理论全局最优。** 结论只适用于已定义的参数空间、预算、seed、环境、
  软件版本和目标硬件。
- **更多 worker 或并发 trial 不必然更快。** CPU、进程通信、GPU learner、显存和磁盘
  都可能成为瓶颈，且竞争会降低 trial 间的可比性。
