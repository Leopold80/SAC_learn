# LTC 时序特征提取器

本文只说明仓库中 LTC 相关代码的实际结构、数学形式和实验边界。项目首页不展开这些细节。

## 1. 当前定位

仓库中的 LTC 是 **fixed-window temporal feature extractor**：

```text
最近 K 帧 observation
        ↓
窗口内 LTC 状态递推
        ↓
最终 liquid state 映射为 feature
        ↓
SB3 SAC / PPO actor-critic
```

它不是跨环境步持续携带 hidden state 的 recurrent policy，也没有 sequence replay、burn-in 或 NCP sparse wiring。每次调用 feature extractor 时，内部状态都在当前 observation window 内重新初始化和递推。

对应代码：

- `sac_experiments/ltc_features.py`
- `sac_experiments/variants.py`
- `configs/sac/ltc_comparison.yaml`

配置校验要求 LTC variant 使用

\[
K=\texttt{frame\_stack}\ge 2.
\]

## 2. 输入张量

环境提供的 stacked observation 在 extractor 内统一整理为

\[
O\in\mathbb{R}^{B\times K\times d_o},
\]

其中：

- \(B\) 是 batch size；
- \(K\) 是 frame stack 长度；
- \(d_o\) 是单帧 observation 维数。

`_as_stacked_observations()` 同时接受 SB3 可能给出的展平形式

\[
O_{\mathrm{flat}}\in\mathbb{R}^{B\times Kd_o}
\]

和显式时序形式 \(B\times K\times d_o\)，并统一为后者。

## 3. Legacy simple LTC

`LTCTemporalFeaturesExtractor` 是早期兼容分支，对每一帧计算

\[
\tilde h_k
=
\tanh\!\left(W_u o_k+W_h h_k+b\right),
\]

再用显式 Euler 更新

\[
h_{k+1}
=
h_k+\Delta t\,\frac{-h_k+\tilde h_k}{\tau},
\qquad
\tau=\operatorname{softplus}(\ell_\tau)+10^{-3}.
\]

该分支对应 variant `ltc_simple`，仅作为历史兼容和调试参考，不是当前正式主线。

## 4. Circuit LTC 连续时间形式

`CircuitLTCTemporalFeaturesExtractor` 对应

\[
\dot{x}_i
=
-\frac{x_i}{\tau_i}
+
\sum_{j=1}^{H}
f_{ij}(x_j,u;\theta)\left(A_{ij}-x_i\right),
\]

其中：

- \(x_i\) 是第 \(i\) 个 liquid unit 的状态；
- \(H\) 是 `liquid_hidden_dim`；
- \(u\) 是当前窗口中的 observation；
- \(A_{ij}\) 是可学习 reversal potential；
- \(f_{ij}\in(0,1)\) 是连接 \(j\to i\) 的门控强度；
- \(\tau_i\) 是正时间常数。

代码中

\[
\tau_i
=
\operatorname{softplus}(r_i)+\tau_{\min}.
\]

对每个来源节点 \(j\)，`gate_layer` 读取 \([x_j;u]\)，同时输出其到全部目标节点的门控。可以写成

\[
f_{ij}(x_j,u;\theta)
=
\sigma\!\left(w_i^{\mathsf T}[x_j;u]+b_i\right).
\]

初始 gate bias 设为 \(-5\)，使初始连接大多接近关闭，避免随机初始化时出现过强耦合。

## 5. 半隐式 Euler 更新

将连续方程整理为

\[
\dot{x}_i
=
-\left(rac{1}{\tau_i}+\sum_j f_{ij}\right)x_i
+
\sum_j f_{ij}A_{ij}.
\]

仓库对与 \(x_i\) 成比例的衰减项使用隐式处理，对 reversal drive 使用显式处理。设子步长

\[
\delta t
=
\frac{\texttt{dt}}{\texttt{ode\_unfolds}},
\]

则代码中的更新恰好是

\[
x_i^{(m+1)}
=
\frac{
 x_i^{(m)}
 +\delta t\sum_j f_{ij}^{(m)}A_{ij}
}{
 1+\delta t\left(	au_i^{-1}+\sum_j f_{ij}^{(m)}\right)
}.
\]

这对应 `numerator / denominator` 的实现。每一帧 observation 内执行 `ode_unfolds` 个子步，然后处理下一帧。

窗口首帧用于初始化：

\[
x_0=\tanh(W_{\mathrm{in}}o_0).
\]

窗口结束后通过

\[
z_{\mathrm{ltc}}
=
\operatorname{ReLU}\!\left(W_z\operatorname{LayerNorm}(x_K)+b_z\right)
\]

得到送入 SB3 policy 的 feature。

## 6. Residual LTC

`ResidualCircuitLTCFeaturesExtractor` 同时保留原始 Markov 信息和 LTC 时序特征：

\[
z_{\mathrm{raw}}
=
\rho\!\left(\operatorname{vec}(O_{1:K})\right),
\]

\[
z_{\mathrm{ltc}}
=
\psi_{\mathrm{LTC}}(O_{1:K}),
\]

\[
z
=
F\!\left([z_{\mathrm{raw}};z_{\mathrm{ltc}}]\right).
\]

其中 \(\rho\) 是 `raw_projection`，\(F\) 是 `fusion` MLP。该结构对应 variant `ltc_residual`。

## 7. Action-history variant

`ltc_residual_action` 使用增强帧

\[
\tilde o_k=[o_k;a_{k-1}]
\]

作为 LTC 分支输入，但 raw residual 分支只截取原始 observation：

\[
z_{\mathrm{raw}}
=
\rho\!\left(\operatorname{vec}(o_{1:K})\right),
\qquad
z_{\mathrm{ltc}}
=
\psi_{\mathrm{LTC}}(\tilde o_{1:K}).
\]

这是代码中 `raw_obs_dim` 的用途。该约束用于分离“动作历史改善时序建模”与“所有网络都多看了动作”两种效应。

## 8. Variant 与代码映射

| Variant | Feature extractor | 输入 | 定位 |
|---|---|---|---|
| `mlp` | SB3 `FlattenExtractor` | stacked 或单帧 observation | 基线 |
| `ltc_simple` | `LTCTemporalFeaturesExtractor` | stacked observation | legacy |
| `ltc` | `CircuitLTCTemporalFeaturesExtractor` | stacked observation | circuit LTC |
| `ltc_residual` | `ResidualCircuitLTCFeaturesExtractor` | stacked observation | raw + LTC fusion |
| `ltc_residual_action` | `ResidualCircuitLTCFeaturesExtractor` | observation + previous action | 动作历史只进入 LTC 分支 |

`variants.py` 负责把 variant 名称映射到 extractor class 和 kwargs；训练器本身不包含 LTC 专属训练循环。

## 9. 计算复杂度

Dense circuit LTC 为每个目标—来源节点对计算 gate，因此核心张量大小约为

\[
B\times H\times H.
\]

忽略输入投影和输出层后，单次 extractor forward 的主要计算复杂度约为

\[
\mathcal O\!\left(BKUH^2\right),
\]

其中 \(U=\texttt{ode\_unfolds}\)。reversal potential 的参数量也是

\[
\mathcal O(H^2).
\]

因此比较 LTC 与 MLP 时必须同时报告 reward、优化参数量、wall-clock 和吞吐量，不能只看单次最高回报。

## 10. 实验边界

当前 LTC 实验只回答：在相同 SB3 训练框架下，fixed-window 时序特征是否改善 LunarLander 的学习表现。

它不直接证明：

- LTC 必然优于 MLP；
- NCP wiring 有效；
- 网络具有闭环稳定性保证；
- fixed-window 结果能够自动迁移到 recurrent SAC；
- LunarLander 结果能够直接推广到 USV/UAV。

后续结构研究只在多 seed、同训练预算和明确容量口径下推进。统一评价协议见 [`experiment_protocol.md`](experiment_protocol.md)。
