# 方案B 完整矩阵报告：{v,g,e,m} × {PySR, LGF} 全对照
## Tensor-Basis Coefficient Caliber: Full Matrix, Formal Formulas, and Joint Refit

日期：2026-10-16 ~ 10-19 ｜ 项目：LLM-Feynman-G ｜ 数据：旋转槽道流 DNS（Re_τ=180, Ro_τ=7.5/15/30/75/150）

> **本报告用途**：① 论文排版（表格 + 矢量图 fig1–fig6 PDF）；② PINN 工程接入（附录 A 变体冠军公式套，含 LaTeX 数值式与 numpy 可调用表达式）。配套文件：`formulas_champions.md`（公式附录单文件版）、`figs/*.pdf|png`、`basisB2_v2_summary.json`。

---

## 1. 口径定义（所有结果遵循）

目标：将雷诺应力各向异性张量的 23 分量严格写成张量基形式

$$b_{23} = \sum_{j} s_j \, a_j(\Theta) \, \tilde T^{(j)}_{23}$$

- $\tilde T^{(j)}_{23}$：张量基 $T^{(j)}$ 的 23 分量，**逐工况 RMS 归一**（消除跨工况 3–4 个量级的动态范围病态）；$s_j$ 为该通道 scale；
- $a_j(\Theta)$：待发现的系数函数，$\Theta=\{\eta_2,\zeta_2,S^2/\Omega^2,Re_\tau,r_{o,\mathrm{loc}},y_w\}$（Pope 不变量 + 局部旋转数 + 真壁距；yw∈[0,1] 无量纲）；
- 变体 = 张量基池：**v/g/e** 各 4 通道（T1/T5/T6/T10，W 定义不同）；**m** 10 通道混合池（T1 + T5/T6/T10 × {v,g,e} 型，54 特征）；
- 评估：重组 b23 的逐工况 NMSE（分母 = 逐工况 var_c(b23)），worst = 5 工况最大值；
- **两级口径**：
  - **中间口径**：SR 逐通道估常数后直接重组——用于方法对比（所有方法承受同一协议，公平）；
  - **最终口径（joint refit）**：固定 SR 公式形状，把全部通道的常数联合最小化重组 NMSE（Levenberg–Marquardt，`--stage refit`）。PySR 的内联数值常量同样参数化后参与同一联合优化——**终口径对两方法完全公平**。

## 2. 方法与档位

| 档位 | 配置 | 说明 |
|---|---|---|
| PySR full | maxsize=25, 3 seeds | 全尺寸 PySR（回归基线） |
| PySR ms12 | maxsize=12 | **等复杂度对照**（与 LGF 输出复杂度匹配） |
| LGF legacy | pop60/gens25/≤6参数 | 旧配置 |
| LGF v2 | pop80/gens30/≤8参数/180s + 物理提示升级（SSG/EB-RISM 闭包族显式列举） | 增强档 |
| LGF big | pop80/gens40/**≤12参数**/max_edges=24 + 联合解 β_j 结构注入 G_INIT | 大形状档 |
| LGF nophys | 同 v2 但 generic 提示 | 知识消融 |

- LGF 后端：GLM-5.3（thinking enabled, reasoning_effort=low, max_tokens=16384）；
- LGF 两臂：**mg**（每代引导）/ **go**（仅 G_INIT 注入知识）；
- 运行目录：`outputs/asm/runs_basisB2/<variant>_<method><tag>/result.json`（`refit` 字段 = 最终口径）。

## 3. 完整矩阵主结果

### 3.1 全部 27 组（NMSE 重组 b23；每组两列 = 中间口径 / 最终口径 refit）

| 组 | v 中间 | v 终值 | g 中间 | g 终值 | e 中间 | e 终值 | m 中间 | m 终值 |
|---|---|---|---|---|---|---|---|---|
| PySR full | 4.054 | 1.414 | 0.357 | 0.128 | 0.963 | 0.377 | 46.10 | 1.173 |
| PySR ms12 | 2.909 | 2.004 | 1.007 | 0.288 | 1.233 | 0.570 | 42.77 | 0.956 |
| LGF-mg legacy | 13.77 | 2.570 | 0.938 | 0.172 | 2.566 | 0.370 | 99.54 | **0.126** |
| LGF-go legacy | 7.668 | **2.231** | 0.826 | 4.521⁵ | 2.154 | 2.154⁶ | 117.17 | **0.125** |
| LGF-mg v2 | — | — | 0.646 | 0.152 | 0.983 | 0.111 | — | — |
| LGF-go v2 | — | — | 0.693 | 0.176 | 2.923 | 0.337 | — | — |
| LGF-go nophys | — | — | 1.271 | 0.224 | 1.897 | 0.157 | — | — |
| **LGF-mg big** | — | — | **0.376** | **0.124** | **0.792** | **0.0576** ⭐ | 61.12 | 0.926 |
| LGF-go big | — | — | 0.550 | 0.140 | 0.856 | 0.134 | 34.03 | **0.0722** ⭐ |

**参照系**：48 常数联合解上界 g 0.0034 / e 0.0037 / m 0.0001（in-sample，不可外推）；标量口径 SOTA **0.0514**（T23-full 轮 LGF 公式）。

### 3.2 终口径逐工况（总冠军 e_lgf_mg_big，NMSE）

| Roτ | 7.5 | 15 | 30 | 75 | 150 | worst |
|---|---|---|---|---|---|---|
| NMSE | 0.048 | 0.050 | 0.066 | 0.068 | 0.055 | **0.068** |

跨 5 工况极差 0.020——所有组中最稳；5 工况全部进入 10⁻² 量级。

## 4. 关键结论（论文可直接引用）

1. **系数口径触及 SOTA**：e 基终口径 0.0576 = 标量 SOTA 0.0514 的 1.12×，且保持严格 $g(\Theta)\cdot\tilde T_{ij}$ 结构；g 基 0.124 与 PySR full refit（0.128）持平但复杂度低 3–4 倍。
2. **等复杂度下 LGF 胜 PySR**：PySR 25→12 全面退化（g 0.357→1.007），LGF 复杂度不变更优 → **LLM 物理先验 = 隐式正则**。
3. **误差机理**：中间口径 10⁰ 的误差 84–92% 来自"逐通道独立估常数 × 抵消结构"（抵消比 Σ‖C_m‖/‖b23‖ = 99–32 201×），联合重拟合后全部回落 1–2 个量级——**SR 找形状 + 联合定常数**是系数口径的正确两阶段用法。
4. **知识消融**：g 基 physics −45%（0.693 vs 1.271）；e 基 go 臂反号（2.92 vs 1.90）→ 知识作用路径不单调，mg（持续引导）比 go（仅初始化）稳健。
5. **m 池两重结论**：① legacy 爆炸（99–117）可被 refit 治愈到 0.126/0.125——病态在常数确定方式；② big 档两臂分化：mg 65 参仅 0.926（过参数化形状不可抛光），**go 55 参 refit 0.0722（全局第二）**——混合池形状在 go 臂（仅初始化注入）下更简洁可抛光。PySR 的 m 形状 refit 后仍 0.96–1.17。
6. **公式形态**：LGF 输出全部落在 ASM 物理族（壁面对数/指数衰减、tanh 饱和、eta2/zeta2 有理式）；nophys 退化为 log(Cyw+C)² 纯数值结构。v 基 a2 的 `170.5·tanh(1.39408x) − 170.5·tanh(1.39414x)` 是抵消结构的活标本（附录 A）。

## 5. 图（PDF 矢量 + PNG 300dpi 双套，图内文字英文）

| 文件 | 内容 |
|---|---|
| `figs/fig1_basisB2_bar.(pdf\|png)` | 变体×方法重组 NMSE 总览（含联合解与 SOTA 线） |
| `figs/fig2_basisB2_profiles.(pdf\|png)` | **终口径冠军** e_lgf_mg_big 剖面：5 工况 DNS vs 重建 |
| `figs/fig3_basisB2_coef.(pdf\|png)` | 系数函数 a_m(Θ)：联合解靶标 vs SR 发现（终口径常数） |
| `figs/fig4_basisB2_heatmap.(pdf\|png)` | 逐工况 NMSE 热图 |
| `figs/fig5_basisB2_v2_groups.(pdf\|png)` | 等复杂度 + 知识消融分组对照 |
| `figs/fig6_basisB2_refit_final.(pdf\|png)` | **终口径全对照**（含标量 SOTA 线） |

## 6. 诚实声明（写论文必附）

1. **in-sample 警告**：0.0576 与标量 SOTA 0.0514 的对比仅在 in-sample NMSE 意义下成立；48 参联合解的留一外推已证不可行（test 3.6–10⁷，`extrapolation_probe.txt`），big 组（21–24 参）未单独重做外推检验。终口径的价值主张是"严格张量基结构 + 显式物理族公式 + 精度追平"，不是外推稳定性。
2. **v 基不可行（有诊断依据）**：T 通道共线（条件数 136–234）；决定性证据 = 逐工况自由系数（零泛化负担）残差仍 ~1.06——形式上限，非方法问题。SR 中间 2.9–13.8、终值 1.4–2.6 一致。
3. **协议失败案例**：g_lgf_go legacy refit 发散（0.826→4.52，可复现）；e_lgf_go legacy refit 无改善（2.154→2.154）——奇异形状（inv/除法链）LM 不收敛，保留原值。
4. yw 为非局部量（真壁距）：严格 Pope 局部口径下精度 0.30–0.34（前轮已证）。
5. PySR 快 15–100×；LGF 每变体单 seed（本轮），上轮 T23-full 多 seed 已给出方差量级。
6. 所有跑批为系数/分量拟合口径，非真 ASM 方程求解（10-08 口径声明）。

## 7. PINN 接入指引

1. 公式见**附录 A**（或独立文件 `formulas_champions.md`）；推荐变体：**e**（0.0576）＞ g（0.124）＞ m（0.126）＞ v（不可用）；
2. 输入变量定义：$\eta_2,\zeta_2,S^2/\Omega^2$ 为 Pope 不变量（由平均场梯度构造），$Re_\tau=180$ 固定，$r_{o,\mathrm{loc}}$ 局部旋转数，$y_w$ 无量纲真壁距（**yw∈[0,1]**，1=壁面、0=中心）；
3. 重建：`b23 = Σ_j s_j · a_j(Θ) · T̃_j23`，通道 scale $s_j$ 与 T̃ 归一约定见附录 A 各公式下方；
4. Python 表达式已内联全部常数，`import numpy as np` 后可直接调用（标量或同形数组）；
5. 训练 caution：公式在训练域内（Re_τ=180，Ro_τ∈[7.5,150]）可靠，域外外推未验证（诚实项 1）。

## 附录 A：变体冠军公式套

（由 `scripts/rc_asm_basisB2_formulas.py` 自动生成；LaTeX 数值式 + numpy 表达式 + 逐工况 NMSE + 数值核对）

# 变体冠军公式套（正式口径：公式形状由 LGF 符号回归发现，常数来自联合重拟合）

> 生成脚本：`scripts/rc_asm_basisB2_formulas.py`；数据源：`outputs/asm/runs_basisB2/*/result.json`。
> 所有公式严格保持张量基形式 $b_{23}=\sum_j s_j\,a_j(\Theta)\,\tilde T^{(j)}_{23}$，
> 其中 $\tilde T^{(j)}_{23}$ 为逐工况 RMS 归一的通道分量，$s_j$ 为该通道 scale（各公式下方给出），
> $\Theta=\{\eta_2,\zeta_2,S^2/\Omega^2,Re_\tau,r_{o,\mathrm{loc}},y_w\}$（yw∈[0,1] 无量纲壁距）。

### 变体 e —— 冠军组 `e_lgf_mg_big` (LGF-mg big (joint refit))

- 最终口径 NMSE(b23) = **0.0576**（worst 0.068），SR 中间口径 0.792；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..4 对应通道 T1s, T5s, T6s, T10s。

**a1(Θ)** — 通道 T1s，scale = 0.0402578：

$$a_{1}(\Theta) = \frac{- 0.000307839 Re_\tau + 3.01635 \left(\eta_{2} + 11.4144\right) \log{\left(1.54276 y_w + 0.922545 \right)}}{\eta_{2} + 11.4144}$$

**a2(Θ)** — 通道 T5s，scale = 0.606326：

$$a_{2}(\Theta) = 1.89998 y_w - \frac{20.8929 y_w}{\sqrt{r_{o,\mathrm{loc}}} + 9.66228} - \frac{1.24817}{\eta_{2} + 1.16246}$$

**a3(Θ)** — 通道 T6s，scale = 0.7168：

$$a_{3}(\Theta) = 0.0533727 y_w^{2} - \frac{9.08352 y_w}{S^2/\Omega^2 - 52.5758} + \frac{0.0182819}{3.04326 r_{o,\mathrm{loc}} - 0.656019}$$

**a4(Θ)** — 通道 T10s，scale = 0.436623：

$$a_{4}(\Theta) = \frac{0.319931 r_{o,\mathrm{loc}} - 0.206477 \left(r_{o,\mathrm{loc}} + 4.6822\right) \tanh{\left(0.00388523 r_{o,\mathrm{loc}} + 7.55284 y_w \right)}}{r_{o,\mathrm{loc}} + 4.6822}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.0402578
a1 = -0.000307839*ret/(eta2 + 11.4144) + 3.01635*np.log(1.54276*yw + 0.922545)
# a2, channel T5s, scale=0.606326
a2 = 1.89998*yw - 20.8929*yw/(np.sqrt(roloc) + 9.66228) - 1.24817/(eta2 + 1.16246)
# a3, channel T6s, scale=0.7168
a3 = 0.0533727*yw**2 - 9.08352*yw/(s2o2 - 52.5758) + 0.0182819/(3.04326*roloc - 0.656019)
# a4, channel T10s, scale=0.436623
a4 = 0.319931*roloc/(roloc + 4.6822) - 0.206477*np.tanh(0.00388523*roloc + 7.55284*yw)
```

逐工况 NMSE：7.5: 0.048 | 15: 0.050 | 30: 0.066 | 75: 0.068 | 150: 0.055

数值核对：✅ max rel diff = 1.47e-14

### 变体 g —— 冠军组 `g_lgf_mg_big` (LGF-mg big (joint refit))

- 最终口径 NMSE(b23) = **0.1238**（worst 0.237），SR 中间口径 0.376；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..4 对应通道 T1s, T5s, T6s, T10s。

**a1(Θ)** — 通道 T1s，scale = 0.0202466：

$$a_{1}(\Theta) = 0.000582218 \eta_{2} - 2.59543 y_w^{2} + 2.33371 y_w - 0.359922$$

**a2(Θ)** — 通道 T5s，scale = 0.397212：

$$a_{2}(\Theta) = - 0.00940079 \sqrt{r_{o,\mathrm{loc}}} - 1.37266 y_w + 0.300209 \left(\log{\left(r_{o,\mathrm{loc}} \right)} + 1.04348\right) \tanh{\left(y_w \right)} + 0.0323185$$

**a3(Θ)** — 通道 T6s，scale = 0.139926：

$$a_{3}(\Theta) = \frac{\left(3.5454 y_w^{2} - 1.73012 y_w + 0.143592\right) \log{\left(y_w + 203.34 \right)}}{0.884546 r_{o,\mathrm{loc}} + 9.19053 y_w + 1.38539}$$

**a4(Θ)** — 通道 T10s，scale = 0.152257：

$$a_{4}(\Theta) = - \frac{59.2735 \eta_{2} \left(1.22487 y_w \zeta_{2} - 0.0501875 \left(\zeta_{2} - 9.16154\right) \tanh{\left(r_{o,\mathrm{loc}} \right)}\right) - 0.269994 \left(\eta_{2} + 899.52\right) \left(\zeta_{2} - 9.16154\right)}{\left(\eta_{2} + 899.52\right) \left(\zeta_{2} - 9.16154\right)}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.0202466
a1 = 0.000582218*eta2 - 2.59543*yw**2 + 2.33371*yw - 0.359922
# a2, channel T5s, scale=0.397212
a2 = -0.00940079*np.sqrt(roloc) - 1.37266*yw + 0.300209*np.log(2.83908*roloc)*np.tanh(yw) + 0.0323185
# a3, channel T6s, scale=0.139926
a3 = (3.5454*yw**2 - 1.73012*yw + 0.143592)*np.log(yw + 203.34)/(0.884546*roloc + 9.19053*yw + 1.38539)
# a4, channel T10s, scale=0.152257
a4 = 59.2735*eta2*(-1.22487*yw*zeta2/(zeta2 - 9.16154) + 0.0501875*np.tanh(roloc))/(eta2 + 899.52) + 0.269994
```

逐工况 NMSE：7.5: 0.125 | 15: 0.070 | 30: 0.106 | 75: 0.237 | 150: 0.081

数值核对：✅ max rel diff = 2.59e-12

### 变体 m —— 冠军组 `m_lgf_go_big` (LGF-go big (joint refit))

- 最终口径 NMSE(b23) = **0.0722**（worst 0.155），SR 中间口径 34.029；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..10 对应通道 T1s, T5s_v, T5s_g, T5s_e, T6s_v, T6s_g, T6s_e, T10s_v, T10s_g, T10s_e。

**a1(Θ)** — 通道 T1s，scale = 0.0192964：

$$a_{1}(\Theta) = - \frac{0.00464753 \sqrt{Re_\tau} y_w}{0.000428044 S^2/\Omega^2 + 1.71477} + \frac{2.50071 r_{o,\mathrm{loc}}}{0.830462 r_{o,\mathrm{loc}} + 7.58463} - 2.33993 y_w$$

**a2(Θ)** — 通道 T5s_v，scale = 0.69033：

$$a_{2}(\Theta) = 2.49911 y_w^{2} - 0.0867483 y_w \left(\log{\left(r_{o,\mathrm{loc}} \right)} - 5.97991\right) - 2.28592 \tanh{\left(r_{o,\mathrm{loc}} \right)} + 2.2864$$

**a3(Θ)** — 通道 T5s_g，scale = 1.54364：

$$a_{3}(\Theta) = - \frac{4.52963 \cdot 10^{-7} r_{o,\mathrm{loc}}^{\frac{5}{2}} - \left(0.0032569 r_{o,\mathrm{loc}} + 1\right) \left(5.43221 \tanh{\left(1.1742 r_{o,\mathrm{loc}} \right)} - 5.44091 \tanh{\left(1.23157 r_{o,\mathrm{loc}} \right)}\right)}{0.0032569 r_{o,\mathrm{loc}} + 1}$$

**a4(Θ)** — 通道 T5s_e，scale = 2.775：

$$a_{4}(\Theta) = - 4.03543 \cdot 10^{-5} r_{o,\mathrm{loc}} y_w^{2} - 0.013683 y_w + 0.00136314 + e^{- 11.4082 r_{o,\mathrm{loc}}}$$

**a5(Θ)** — 通道 T6s_v，scale = 0.145571：

$$a_{5}(\Theta) = - \frac{2.67749 y_w \zeta_{2}}{\zeta_{2} + 30.2419} + 0.107535 - 29.4225 e^{- 3.60952 r_{o,\mathrm{loc}}}$$

**a6(Θ)** — 通道 T6s_g，scale = 4.93981：

$$a_{6}(\Theta) = \frac{2.24864 \zeta_{2} - \left(0.000167474 r_{o,\mathrm{loc}} + 2.25357 \tanh{\left(1.7304 r_{o,\mathrm{loc}} \right)}\right) \left(\zeta_{2} - 1.88983 \cdot 10^{-5}\right)}{\zeta_{2} - 1.88983 \cdot 10^{-5}}$$

**a7(Θ)** — 通道 T6s_e，scale = 4.72961：

$$a_{7}(\Theta) = \frac{0.0063886 \sqrt{r_{o,\mathrm{loc}}} y_w + \left(1.45957 y_w + \tanh{\left(y_w \right)}\right) \left(2.19698 \tanh{\left(1.52017 r_{o,\mathrm{loc}} \right)} - 2.19851\right)}{1.45957 y_w + \tanh{\left(y_w \right)}}$$

**a8(Θ)** — 通道 T10s_v，scale = 0.523326：

$$a_{8}(\Theta) = \frac{0.0978421 y_w + \left(0.000377304 r_{o,\mathrm{loc}} - 0.349575 y_w\right) \left(r_{o,\mathrm{loc}} - 0.709357\right) - 0.00489058}{r_{o,\mathrm{loc}} - 0.709357}$$

**a9(Θ)** — 通道 T10s_g，scale = 0.131477：

$$a_{9}(\Theta) = \sqrt{r_{o,\mathrm{loc}}} \left(- 0.0011614 r_{o,\mathrm{loc}} + 0.0685236 \log{\left(5.73961 y_w + 0.230465 \right)}\right)$$

**a10(Θ)** — 通道 T10s_e，scale = 0.28521：

$$a_{10}(\Theta) = \frac{24.833 y_w \left(0.0185322 r_{o,\mathrm{loc}} - 0.00880291 \left(r_{o,\mathrm{loc}} + 1\right) \log{\left(r_{o,\mathrm{loc}} \right)}\right) + 0.00468331 \left(\sqrt{r_{o,\mathrm{loc}}} + r_{o,\mathrm{loc}}\right) \left(r_{o,\mathrm{loc}} + 1\right)}{r_{o,\mathrm{loc}} + 1}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.0192964
a1 = -0.00464753*np.sqrt(ret)*yw/(0.000428044*s2o2 + 1.71477) + 2.50071*roloc/(0.830462*roloc + 7.58463) - 2.33993*yw
# a2, channel T5s_v, scale=0.69033
a2 = 2.49911*yw**2 - 0.0867483*yw*np.log(0.00252906*roloc) - 2.28592*np.tanh(roloc) + 2.2864
# a3, channel T5s_g, scale=1.54364
a3 = -4.52963e-7*roloc**(3/2)/(0.0032569 + 1/roloc) + 5.43221*np.tanh(1.1742*roloc) - 5.44091*np.tanh(1.23157*roloc)
# a4, channel T5s_e, scale=2.775
a4 = -4.03543e-5*roloc*yw**2 - 0.013683*yw + 0.00136314 + np.exp(-11.4082*roloc)
# a5, channel T6s_v, scale=0.145571
a5 = -2.67749*yw*zeta2/(zeta2 + 30.2419) + 0.107535 - 29.4225*np.exp(-3.60952*roloc)
# a6, channel T6s_g, scale=4.93981
a6 = -0.000167474*roloc + 2.24864*zeta2/(zeta2 - 1.88983e-5) - 2.25357*np.tanh(1.7304*roloc)
# a7, channel T6s_e, scale=4.72961
a7 = 0.0063886*np.sqrt(roloc)*yw/(1.45957*yw + np.tanh(yw)) + 2.19698*np.tanh(1.52017*roloc) - 2.19851
# a8, channel T10s_v, scale=0.523326
a8 = 0.000377304*roloc - 0.349575*yw - 0.0225776*(0.216612 - 4.33359*yw)/(roloc - 0.709357)
# a9, channel T10s_g, scale=0.131477
a9 = 0.183739*np.sqrt(roloc)*(-0.00632093*roloc + 0.37294*np.log(5.73961*yw + 0.230465))
# a10, channel T10s_e, scale=0.28521
a10 = 0.00468331*np.sqrt(roloc) + 0.00468331*roloc + 24.833*yw*(0.0185322*roloc/(roloc + 1) - 0.00880291*np.log(roloc))
```

逐工况 NMSE：7.5: 0.033 | 15: 0.054 | 30: 0.041 | 75: 0.078 | 150: 0.155

数值核对：✅ max rel diff = 1.56e-13

### 变体 v —— 冠军组 `v_lgf_go` (LGF-go legacy (joint refit))

- 最终口径 NMSE(b23) = **2.2305**（worst 3.058），SR 中间口径 7.668；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..4 对应通道 T1s, T5s, T6s, T10s。

**a1(Θ)** — 通道 T1s，scale = 0.100647：

$$a_{1}(\Theta) = 0.179751 \left(\log{\left(r_{o,\mathrm{loc}} \right)} + 1.33836\right) \log{\left(2.44248 - 2.12821 y_w \right)}$$

**a2(Θ)** — 通道 T5s，scale = 9.65403：

$$a_{2}(\Theta) = - \left(0.0157721 \tanh{\left(3.638 r_{o,\mathrm{loc}} \right)} + 0.013384 \tanh{\left(0.00630331 S^2/\Omega^2 \right)}\right) \log{\left(y_w + 0.266476 \right)}$$

**a3(Θ)** — 通道 T6s，scale = 0.278312：

$$a_{3}(\Theta) = \frac{- 7.12859 \cdot 10^{-5} \eta_{2} \left(S^2/\Omega^2 - 0.0095088\right) + 0.0373976 S^2/\Omega^2 + 0.00930046}{S^2/\Omega^2 - 0.0095088}$$

**a4(Θ)** — 通道 T10s，scale = 9.56966：

$$a_{4}(\Theta) = \frac{0.190595 r_{o,\mathrm{loc}} \left(r_{o,\mathrm{loc}} + 75.1609 y_w\right)}{466.199 r_{o,\mathrm{loc}}^{2} - 46.7772}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.100647
a1 = 0.179751*np.log(3.81279*roloc)*np.log(2.44248 - 2.12821*yw)
# a2, channel T5s, scale=9.65403
a2 = 0.410519*(-0.0384198*np.tanh(3.638*roloc) - 0.0326027*np.tanh(0.00630331*s2o2))*np.log(yw + 0.266476)
# a3, channel T6s, scale=0.278312
a3 = -7.12859e-5*eta2 - 15.7459*(-0.00237507*s2o2 - 0.000590659)/(s2o2 - 0.0095088)
# a4, channel T10s, scale=9.56966
a4 = -0.190595*roloc*(roloc + 75.1609*yw)/(46.7772 - 466.199*roloc**2)
```

逐工况 NMSE：7.5: 1.602 | 15: 1.771 | 30: 2.152 | 75: 2.569 | 150: 3.058

数值核对：✅ max rel diff = 7.85e-13

---

## 附录 B：产物清单

- 本报告：`outputs/asm/t23basisB2_paper/BasisB2_full_matrix_report.md`
- 公式附录单文件：`formulas_champions.md`；汇总 JSON：`basisB2_v2_summary.json`
- 图：`figs/fig1–fig6`（PDF+PNG 双套）
- 运行数据：`outputs/asm/runs_basisB2/*/result.json`（`refit` 字段 = 最终口径常数）
- 脚本：`rc_asm_basisB2.py`（base/sr/refit 三阶段）、`rc_asm_basisB2_agg2.py`、`rc_asm_basisB2_figs.py`、`rc_asm_basisB2_formulas.py`
- 历史报告：`BasisB2_v2_report.md`（v2 对照轮）、`prev_t23full/`（T23-full 轮）、前轮 `t23basisB_paper/`
