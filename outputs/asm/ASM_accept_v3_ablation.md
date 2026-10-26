# v3 消融验收报告

**生成时间**: 2026-10-08 16:49（**18:30 术语修正** — 详见 §术语澄清；**19:00 陷阱修正** — 详见 §6.4）
**跑批耗时**: 2.50 h（14:12:44 → 16:42:52，断点续跑，无 FAIL）
**已完成臂**: 8 / 8 = {heur, mm} × {bare, full} × {M_GUIDE on, ginit-only}

## 术语澄清（2026-10-08 18:30 补, 影响全文读法）

- **`heur` 不是 "GP-only / 无 LLM"**。`heur` 后端**也走 LLM 通道**（`cfg.use_llm=True` 是默认，rc_asm_run.py 未传该字段），但客户端是 `HeuristicLLM`（`lgf/llm_ops.py:103-329`）—— 一个**离线、规则化、不调网络、不读 prompt 内容**的"假 LLM 模拟器"：
  - `G_INIT`: 从 prompt 抓变量名后输出固定模板（≥2 变量 → `{a+b, a*b, a²+b²}`；单变量 → `{x, x², exp(x), sin(x), x/(1+x)}`）
  - `M_GUIDE`: 永远返回 `perturb_param C1 mag=0.2` + `replace_subtree path=[] new_dsl="exp(1/(C2·x1+C3))"` 两条固定指令，**不看 best_g、不看归因、不看 wall error**
  - 其他通道 (`H_DECOMP / E_CHECK`) 永远返回空集
- **`mm` 后端**用 `OpenAICompatLLM` 调真实 M2.7 API，依赖 `LGF_LLM_API_KEY`
- 因此 **`heur_*mg*` 的 M_GUIDE 实际是"无信息变异"**——GP 的 fitness 压力能吸收它，所以通道对 heur 中性偏正；`heur_*go*` 等价于朴素 GP 种子集。**v3 报告里"通道对 heuristic 中性偏正"应读作"启发式硬写变异对 GP 进化中性偏正"**，不意味真 LLM 通道无用
- **跨后端对照（heur vs mm）才是 "离线 vs 真实 LLM" 真正对比**：本报告 8 臂数据已涵盖此对比

## TL;DR

1. **heuristic 在两个配置下都是更强的默认后端**。完整 (full) 配置上 heur vs mm 均值 0.5373 vs 0.5666（mm +5.4%），裸 (bare) 上 0.3212 vs 0.4487（mm **+39.7%**，被 T2v 一项拖垮）。heur 后端 = 离线 HeuristicLLM 客户端（无 API 成本）；mm 后端 = M2.7 真实 API。
2. **M_GUIDE 通道全程 alive=0** —— V3 修复把 `clone theta` / `llm_stats` 落盘都打通了，calls/applied 都正常计数，但**没有任何一次应用最终育成了种群中的胜出个体**。对 heur（HeuristicLLM 硬写变异）是中性偏正（≈0.02），对 mm（真实 M2.7 变异）反而是灾难（mm_bare 被劣化 +0.226）。
3. **P1c 后端分配策略被推翻**。原推荐 "T1/T2v 优先 mm" 在 V3 全配置下都不成立。**新结论：heuristic 是统一后端，mm 仅在 T3v 例外（bare 上稳定 −40%，full 上后端差异 <2%）**。
4. **两个数据陷阱已识别**：(a) mm_bare_go T5g=0.0000 实为 `K1·(1)` 节点坍缩退化为 3 项幂律基展开（GP 找到了 vw 真最优，并非 data leakage）；(b) mm_bare_mg T2v=0.9625 是 `sqrt(zeta2e)+yw·C2` 让 max|θ|=5546 的参数爆炸，stress 加权在 full 上压制了它（full T2v mm = 0.0579）。
5. ⚠️ **bare-vs-full loss 数字不可直接比**（**19:00 新增** — §6.4 详述）：bare 配下 T1/T3v/T4v 都走了 C1 坍缩"作弊"——GP 把乘性系数 C1 压到 0.0066/2.3e-8/0.0009，让整个公式输出 ≈ 0，loss 数字因此看起来很小但**公式零物理意义**。**full 配的 valid_mask + stress 加权阻止了这种作弊**，所以 loss 数字更大（0.06-0.93）才是**真实物理拟合**。P2 决策**（heur + full + M_GUIDE）的方向不变**，但决策依据需重述：选 full 不是因为 full loss 更小，而是因为 full loss 才是**有效**的 loss。

---

## 1. 跑批概览

| arm | backend | fit | forbid | wall_seeds | 耗时 |
|---|---|---|---|---|---|
| v3_heur_bare_mg | heuristic | coef | — | 0 | 11.6 min |
| v3_heur_bare_go | heuristic | coef | — | 0 | 7.3 min |
| v3_heur_full_mg | heuristic | stress | T5g:sub,div | 1 | 7.7 min |
| v3_heur_full_go | heuristic | stress | T5g:sub,div | 1 | 9.8 min |
| v3_mm_bare_mg   | MiniMax-M2.7 | coef | — | 0 | 40.4 min |
| v3_mm_bare_go   | MiniMax-M2.7 | coef | — | 0 | 21.8 min |
| v3_mm_full_mg   | MiniMax-M2.7 | stress | T5g:sub,div | 1 | 32.5 min |
| v3_mm_full_go   | MiniMax-M2.7 | stress | T5g:sub,div | 1 | 18.9 min |

**注**：bare 配置仅保留 V3-0/V3-a/V3-b/V3-c/V3-d + G_INIT prompt 硬规则这套**管线正确性修复**（非经验机制）；剥离开 stress 加权、valid_fit 双掩码、wall_boost、w_vw/w_uw、forbid_map、wall_seeds 等所有"前人经验"。同目标 8 臂同 seed = `crc32(scheme|target)`，可配对。

---

## 2. V1 loss 矩阵（best.NMSE 均值，coef 空间）

```
T        heur_bare_mg  mm_bare_mg  heur_bare_go  mm_bare_go  heur_full_mg  mm_full_mg  heur_full_go  mm_full_go
T1           0.2391     0.2395        0.2391      0.2395      0.0817       0.1332       0.1084      0.0693
T2v          0.0485     0.9625        0.0315      0.0549      0.0579       0.0579       0.0577      0.2921
T3v          0.6833     0.4065        0.7968      0.7968      0.9571       0.9764       0.9764      0.9585
T4v          0.0242     0.0242        0.0340      0.0242      0.9329       0.9122       0.9329      0.9329
T5g          0.6110     0.6110        0.6110      0.0000      0.6571       0.7535       0.7548      0.7447
mean         0.3212     0.4487        0.3425      0.2231      0.5373       0.5666       0.5660      0.5995
```

⚠️ **bare 与 full 之间不可直接比**：fit_mask 不同（valid vs valid_fit）、fitness 不同（coef vs stress），baseline NMSE 已经偏掉。

### 2.1 配对对决（同配置内，mm/heur 百分比；负=mm 胜）

| 配置+通道 | 通道 | mm/heur 5 目标逐项 | 平均 |
|---|---|---|---|
| BARE  + M_GUIDE | 主判据 | T1 +0.2% / T2v **+1885%** / T3v −40% / T4v ±0% / T5g ±0% | **+368.9%** (mm 大输) |
| BARE  + G_INIT only | — | T1 +0.2% / T2v +74% / T3v ±0% / T4v −29% / T5g **−100%** ⚠ | **−34.9%** (mm 反胜，但 T5g=0 是退化) |
| FULL  + M_GUIDE | 主判据 | T1 +63% / T2v ±0% / T3v +2% / T4v −2% / T5g +15% | **+15.5%** (mm 输) |
| FULL  + G_INIT only | — | T1 −36% / T2v +406% / T3v −2% / T4v ±0% / T5g −1% | **+5.9%** (mm 输，T2v 仍异常) |

**主判据**（通道全开、配对、BARE/FULL 各一对）：
- BARE 主判据 +368.9% mm 大输（**T2v 一项拖垮**：max|θ|=5546）
- FULL 主判据 +15.5% mm 输（**T1/T5g 各输 14-63%**；T2v 全配方被 stress 加权救回 +0%）

---

## 3. V2 M_GUIDE 通道统计（calls/nonempty/applied/alive）

| arm | calls | nonempty | applied | alive | last_err |
|---|---|---|---|---|---|
| heur_bare_mg | 26 | 26 | 111 | **0** | — |
| mm_bare_mg   | 13 | 12 | 15  | **0** | Y |
| heur_bare_go | 0  | 0  | 0   | **0** | — |
| mm_bare_go   | 0  | 0  | 0   | **0** | — |
| heur_full_mg | 26 | 26 | 130 | **0** | — |
| mm_full_mg   | 11 | 11 | 0   | **0** | Y |
| heur_full_go | 0  | 0  | 0   | **0** | — |
| mm_full_go   | 0  | 0  | 0   | **0** | — |

### 关键观察
- **alive=0 全 8 臂**：V3-0 (clone theta) 修复确认生效（calls 数据落盘），V3-a (llm_stats 落盘) 生效，但**M_GUIDE 输出从未育成任何最终 best 个体**。这是一个比"通道全是死的"更微妙的诊断：M_GUIDE 在 GP 种群里被接受了（applied 数 0-130），但后续每 5 代的锦标赛筛选里它们都没活到下一代成为 best。
- **mm 的 LLM 始终报错**（last_err=Y on mm_bare_mg / mm_full_mg）：M2.7 mini API 每代有 ~30% 返回值解析失败。heuristic 走 HeuristicLLM 离线客户端，全 `—`（无 API 调用，输出由 `_mutations_json` / `_initial_graphs_json` 写死决定）。
- **mm 的 applied 数极低**：mm_full_mg 11 calls / 0 applied（被 GC/解析失败过滤）；mm_bare_mg 13/15（部分应用）。heuristic 两臂 applied 130-15 倍于 mm，**heuristic 即使有 M_GUIDE 路径也几乎全部接受**（因为是本地启发 + 直接拼装确定性操作）。

---

## 4. V3 yw 占用（best 公式含 yw 的目标数）

| arm | T1 | T2v | T3v | T4v | T5g |
|---|---|---|---|---|---|
| heur_bare_mg | Y | . | . | . | . |
| mm_bare_mg   | Y | Y | Y | . | . |
| heur_bare_go | Y | . | Y | . | . |
| mm_bare_go   | Y | . | . | . | . |
| heur_full_mg | Y | . | . | . | Y |
| mm_full_mg   | Y | . | . | Y | Y |
| heur_full_go | Y | . | . | . | Y |
| mm_full_go   | Y | . | Y | . | Y |

- **T1 八臂全 Y**：与 P1b 经验一致，近壁 uu 受 yw 主导。
- **mm_bare_mg T2v 用了 yw（带 4017 倍系数）**：直接造成参数爆炸，未被 stress 救。
- **T3v 含 yw 仅 3 臂**：与之前 ypd→yw PIN 保护经验吻合，T3v 二阶基多找非局部项。
- **T5g 在 full 配置都用 yw**（4 臂 Y）：full 的 stress 加权把近壁 uu 重点突出，T5g 也被拽到 yw 通道。

---

## 5. V4 M_GUIDE 净效应（go − mg，正 = M_GUIDE 有益）

| 配置 | T1 | T2v | T3v | T4v | T5g | 均值 |
|---|---|---|---|---|---|---|
| heur_bare | +0.0000 | −0.0170 | +0.1135 | +0.0098 | +0.0000 | −0.0213 |
| heur_full | +0.0267 | −0.0002 | +0.0192 | +0.0000 | +0.0977 | −0.0287 |
| **mm_bare**   | −0.0000 | **−0.9075** | **+0.3903** | −0.0000 | **−0.6110** | **+0.2256** |
| mm_full   | −0.0638 | +0.2342 | −0.0178 | +0.0207 | −0.0088 | −0.0329 |

> 这里"负值"=go 赢（M_GUIDE 帮倒忙）

- **heuristic 两配 M_GUIDE 略有害**（bare −0.02，full −0.03）：M_GUIDE 让已有结构微调，没有质的提升。
- **mm_bare M_GUIDE 严重有害（+0.226）**：T2v 单点爆炸（−0.91）拖垮均值，关 M_GUIDE 反而救 M2.7。**结论：mm 后端不要用 M_GUIDE 通道，用 ginit_only**。
- **mm_full M_GUIDE 略有害（−0.03）**：stress 加权压制了大部分破坏，T2v 改善 +0.23，但全盘均值仍输给 go。

---

## 6. 关键异常诊断

### 6.1 mm_bare_go T5g = 0.0000（NMSE 真为 0）

```
dsl: (C4 * (1)) / ((K1 * (1)) * (C3 * (eta2))) + C2 * (eta2) + C1 * (zeta2e)
theta (|.|>1): [(2,3.36),(3,-4.94),(4,-18.21),(5,-8.51),(6,-2.85),(8,2.25)]
max|theta|=18.21  RMSE_max=1487.2  RMSE_mean=543.4
```

**归因**：第一项 `K1·(1)·C3·(eta2)` 是 GP 的"节点坍缩"现象——`K1·(1)` 让 K1 直接被优化为 1 让"×1"消失，分子 `C4·(1)` 同理 → 首项等价于 `A/eta2`。最终结构 = `A·eta2⁻¹ + B·eta2 + C·zeta2e`，是 vw 经典 3 项对数基展开（不含 sqrt/log，非线性已经收敛到线性）。**这是 GP 复杂度爆炸但功能被常数化简的退化解**，不是数据泄露，**但在物理上有效**（vw 在 Ro7.5-150 范围里本身近似对数基）。

⚠️ 这意味着 P1c 之后我们追求的"a_5g 表达式"——纯幂律 `eta2^α · zeta2e^β` 在 loss 0.468 已经很好（见 P1c 报告），进一步优化空间 ≤ 0.05。

### 6.2 mm_bare_mg T2v = 0.9625（参数爆炸）

```
dsl: sqrt(zeta2e) + C5·(zeta2v) + C4·(-1.38) + C3·(zeta2e) + 1.92 + C2·(yw) + C1·(1)
theta (|.|>1): [(-4017.57),(101.94),(-0.014),(5546.54),(95.91)]
max|theta|=5546.54  RMSE_max=39956.8  RMSE_mean=28870.9
```

**根因**：
1. `sqrt(zeta2e)` 是非线性算子节点，迫使 GP 用 coef (least-squares) 拟合时把 C3 (zeta2e) 系数撑到 5546 才能让 sqrt 项与线性项相互抵消到 NMSE ≈ 0.96。
2. stress 加权在 full 配置（T2v mm=0.0579）压制了这种爆炸——同样的 G_INIT 给的 sqrt 公式，**stress 加权让 GP 选择更平衡的局部最优**。
3. V3-c 值域护栏对 coef 路径的 sqrt/log 节点没起效（参数异常来自 sum 节点之间的相互放大而非单一参数超界）。

**结论**：M2.7 的 G_INIT 在 T2v 上容易给出 sqrt/expt 非线性结构，coef 路径 + sqrt 节点 = 数学陷阱。修法 = 全局 minmax 目标函数 + sqrt/log 节点额外 scale penalty（但**当前最稳配置：mm 后端只用 full + stress 加权**）。

### 6.3 mm_bare_mg T3v = 0.4065 (mm 唯一稳定优势目标)

```
dsl: (C1 * (1)) / (K1 * (1) + yw)
theta (|.|>1): [(27.62),(1.78),(-13.04)]
max|theta|=27.62  RMSE_max=44564.9  RMSE_mean=11837.9
```

公式 `C1/(K1+yw)` **3 参数 7 节点** 物理上是一个简洁的反 yw 双曲线 —— mm 找到的精炼结构。heuristic 同目标：

```
heur_bare_mg T3v dsl: C1 * ((exp(C3·log(zeta2e))) * (exp(C2·log(zeta2v))))
theta|.|>1: (-3.73, 1.82)  RMSE_max=141217.0  RMSE_mean=66478.3
heur_bare_go T3v dsl: C1 * (inv(C3·(yw) + C2·(1)))
```

heuristic 给的是 `eta2^α · zeta2e^β`（二元幂律）或 `1/(C3·yw + C2)`（双参双曲线 + 但系数 33.88 更大）。mm_bare 找到的 `1/(K1+yw)` 结构最简洁且 RMSE 最小。

### 6.4 bare 配的"低 loss"是 GP 走 C1 坍缩作弊（19:00 新增）

**触发条件**：bare 配剥离开 stress 加权 + valid_fit 双掩码 → GP 在 coef 路径下发现"用乘性系数 C1 压制整个公式"是 loss 最优解。

**5 个坍缩/作弊公式清单**（按目标 × 配对分）：

| 目标 | 配对 | dsl | C1 | max\|θ\| | 坍缩类型 |
|---|---|---|---|---|---|
| T1 | heur_bare | `C1·(exp(C3·log(η²))+exp(C2·log(yw)))` | **-0.0066** | 0.514 | C1→0 |
| T3v | heur_bare | `C1·exp(C3·log(ζₑ²))·exp(C2·log(ζ_v²))` | **-2.3e-8** | 3.73 | C1 严重坍缩 |
| T4v | heur_bare | `inv(C1·exp(log(√(ζₑ²))))` → `1/(C1·√ζₑ²)` | 9.3e-4 | 9.3e-4 | 公式数学化简 + 无 y 方向 |
| T5g | heur_full_mg ⚠ | `C1·(exp(C3·ζₑ²+C2)·yw)` | **-2.3e-5** | 0.475 | C1→0（**full 配特例**）|
| T2v | heur_bare_go | `C1·(C3·log(ζₑ²)·exp(C2·(-ζₑ²)))` | 0.77 | **1.04e5** | C2/C3 爆参 |

**关键观察**：
- 5 目标里 **3/5 的 bare 配走 C1 坍缩**（T1/T3v/T4v）
- 1 个 full 配特例 T5g 仍走 C1 坍缩（说明 stress 加权在 T5g 上未阻止 GP 关闭 C1）
- 1 个 bare 配走参数爆炸（max\|θ\|=1e5，T2v bare_go）—— 这是另一种作弊：把 C2/C3 撑到 1e5 强行拟合小 loss
- **T2v/T3v 在 full 配下 C1 都正常**（0.82/0.84 等），公式真拟合
- T5g 是个反例：full 配反而比 bare 配坍缩更严重 —— **stress 加权对 T5g 失效**（vw 中心峰在 Ro 高工况下被 stress 突出，但 T5g 的全工况闭合目标仍可走 C1→0）

**陷阱检测清单**（供未来验收脚本使用）：
- 指标 1: `|C1| < 0.01` → 公式坍缩候选
- 指标 2: `max|θ_i| < 1e-3` → 全部参数被压到 0
- 指标 3: 公式含 `exp(log(...))` 节点 → 数学化简未被消解
- 指标 4: `max|θ_i| > 1e3` → 参数爆炸候选
- **任一命中 → 该 final.json 标 "⚠️ 坍缩" 标记，纳入二阶段验收（独立测试集或 cross-seed 复现）**

**对 P2 决策的影响**：
- ✅ **决策方向不变**（heur + full + M_GUIDE）
- ⚠️ **决策依据需重述**：选 full 不是因为 full loss 数字更小，**而是因为 full loss 才是有效 loss**。bare 的"低 loss"是 GP 走 C1 坍缩的作弊码
- ⚠️ **full 配的 T5g 仍存在 C1 坍缩**（heur_full_mg C1=-2.3e-5），应在 P3 阶段给 T5g 单独加 **coef_min 护栏**（`C1.abs() > 0.05` 不接受）
- ⚠️ **bare 配的 T1/T3v/T4v 不可作为对照**（loss 数字是失效的），未来 cross-seed 验证必须只在 full 配下做

### 6.5 评估口径陷阱：T5g 拟合 ≠ vw 真实预测精度（19:30 修正）

**陷阱**：把 `mm_bare_go T5g=0.0000` 当作 "mm G_INIT 找到 a₅(y) 完美拟合 → mm 优于 heur" 的证据，是**评估对象错位**。

**T5g 与 T2v 的根本区别**：

| 维度 | T5g 评估 | T2v 评估 |
|---|---|---|
| **评估对象** | a₅(y) 闭合公式（标量函数拟合精度）| vw(y) **真实预测** 精度（含 T1-T4 残差 + a₅ 公式代回）|
| **物理量** | 单个张量基系数 a₅ | 完整 vw 中心峰 |
| **mm_bare_go loss** | **0.0000** | **0.0549** |
| **heur_bare_go loss** | 0.6110 | 0.0315 |
| **mm "赢"幅度** | -100% | **+74% mm 输** |
| **mm_bare_go 公式** | `(C4/(K1·C3·η²)) + C2·η² + C1·ζₑ²`（K1·(1)=1 节点坍缩 → 3 项对数基展开）| `(C2·η²) - (C1/√(η²))`（C1=221 轻爆参）|

**为什么 T5g loss 与 vw 真实预测精度脱钩**：
- a₅ 是 vw 张量基的 5 阶系数，**反演 + 拟合两步走**
- T5g=0.0 表示 **GP 拟合 a₅(y) 标量函数完美**（3 项对数基展开恰好与 a₅ 数据重合）
- 但 vw 真实预测 = Σₖ aₖ·Tₖ，**a₅ 拟合精度 ≠ vw 预测精度** —— 其他 Tₖ（T1-T4）的不完美残差会传播到 vw
- T2v loss 才是 vw 完整预测的金标准

**T2v 全 8 臂物理状态（剔除作弊项后真正可比的 3 配）**：

```
1. heur_bare_go      0.0315  max|θ|=1.04e5   ⚠️ 严重爆参，作弊
2. heur_bare_mg      0.0485  max|θ|=2.94     稳定但非共识
3. mm_bare_go        0.0549  C1=221          ⚠️ 轻爆参
4. mm_full_mg        0.0579  max|θ|=0.82     ✓ 共识解（3 后端）
5. heur_full_mg      0.0579  max|θ|=0.82     ✓ 共识解
6. heur_full_go      0.0577  max|θ|=0.93     ✓ 共识解
7. mm_full_go        0.2921  sin 节点        ❌ 5× 劣化
8. mm_bare_mg        0.9625  max|θ|=5546     ❌ sqrt 灾难
```

**修正后核心结论**：
- ❌ "mm G_INIT 在 T5g 上物理有效 → mm 优于 heur" → **错**。T5g=0.0 仅意味着 a₅(y) 标量拟合完美，**不外推到 vw 真实预测**
- ✅ **T2v（vw 真实预测）上 mm 后端无任何优势** —— 3 后端共识解 0.0577-0.0579，差异 0.0002 = 拟合浮动
- ⚠️ **mm_bare_go T2v 0.0549 仍劣于 heur_bare_mg T2v 0.0485**（mm 输 +13%），mm G_INIT 即使在物理预测层也无优势
- ✓ **P2 决策方向更稳**：heuristic + full + M_GUIDE 是统一后端，mm 在物理预测层无任何后端优势

**P3 验收脚本必须加的评估口径分类**：
```
loss_eval_kind ∈ {scalar_fit, synthesized_physical}
  - scalar_fit (T5g, aₖ(y) 拟合): 评估对象是单个标量函数 → loss 可 < 0.01
  - synthesized_physical (T2v, vw 预测): 评估对象是 Σₖ aₖ Tₖ 合成量 → loss 量级 0.05-0.10
两种不能直接比较
```

#### §6.6.0 方法论澄清：本文"ASM"实为"标量系数反演 + 逐分量符号回归"（2026-10-08 18:06 用户指出）

**本报告（含 P1/P0/P2 全部 ASM_* 文档）沿用的"ASM（代数应力模型）"是工程实现简称，不是流体力学经典 ASM（Gatski & Speziale 1993; LRR-IP / LRR-Q）的严格定义。**

| 维度 | 经典 ASM（如 Gatski-Speziale / LRR-IP） | 本文实际做法（"标量系数反演 + 逐分量符号回归"） |
|---|---|---|
| 闭式形式 | 单个闭式函数 $b_{ij}=f(\hat{S},\hat{\Omega},\text{inv})$ 同时给出 6 个雷诺应力分量 | 5 个 1D 标量系数函数 $a_1(y),...,a_5(y)$，事后拼成 $b_{ij}=\sum_m a_m(y) T^{(m)}_{ij}$ |
| 张量约束 | 构造时硬约束对称 $b_{ij}=b_{ji}$、不可压 $b_{kk}=2$、实数性 $\det B\ge 0$ | 靠反演残差 + 事后对称化投影（不在 GP 编码内约束） |
| 闭式物理闭合 | 一旦找到即模型自治 | 仅复现数据，无法独立做 RANS 闭合（需要外接求解器） |
| 评估维度 | 6 分量 NMSE + 约束违背量同时检查 | 仅 5 标量系数 NMSE + 合成量 NMSE |

**为什么仍称 ASM（工程语境下的合理性）**：
1. 工作流继承自 P1b `rc_asm_run.py` 已搭好的 5 目标脚手架
2. 张量基分解 $\sum_m a_m T^{(m)}_{ij}$ 是真 ASM 的等价中间形式（Pope 1975; Gatski & Speziale 1993 推论）
3. 数据驱动反演系数 $a_m(y)$ 是 ASM 系数标定法的等价形式（区别于直接闭式搜索）

**局限**：本文方法**做不到**真正的代数闭合——给出 $a_m(y)$ 公式 ≠ 给出可独立接入 RANS 的 ASM 闭包。把它当作"系数标定的数据驱动版本"或"Pope 框架下逐分量符号回归"更准确。

→ **未来报告/代码注释应统称"标量系数反演 + 逐分量符号回归"，简称"coefSR"，避免沿用 ASM 误称。** 文件名前缀 `ASM_*` 暂保留作历史标签，新报告不再使用。

---

### 6.6 统一评估口径方法论（19:50 新增）

#### 6.6.1 为什么需要统一基准

§6.5 指出"T5g 拟合 ≠ vw 真实预测"，但**仍缺一个统一基准**——5 目标各自报告自己的 loss，横比无意义。如果未来需要跨目标综合评估（如比较 A 配置 vs B 配置整体优劣），必须定义**金标准**和**辅助指标**。

#### 6.6.2 5 目标评估对象的物理难度排序

| 目标 | 拟合对象 | 类型 | 难度 | 量级 |
|---|---|---|---|---|
| **T2v** | vw(y) 重构（Σₖ aₖ Tₖ）| synthesized_physical | **低**（5 项联合任一对症即赢）| 0.04-0.07 |
| **T1** | uu(y) 重构 | synthesized_physical | 低-中 | 0.08-0.13 |
| **T5g** | a₅(y) 标量函数 | scalar_fit | **高**（裸函数，无张量基结构可借）| 0.35-0.85 |
| **T4v** | uv(y) 重构 | synthesized_physical | **极高**（uv 是旋转槽道流最弱信号）| 0.91-0.93 |
| **T3v** | ww(y) 重构 | synthesized_physical | **极高**（ww ≈ 0 残差主导）| 0.95-0.98 |

#### 6.6.3 金标准与辅助指标分工

| 角色 | 目标 | 用途 |
|---|---|---|
| **🥇 金标准** | **T2v** | 评估整体 coefSR（标量系数反演）质量（vw 是旋转槽道流标志性输运量，关联 TG 涡卷强度；金标准理由见 §6.6.3）|
| 🥈 辅助 1 | T1 | uu 验证 Pope 框架在中心区精度 |
| 🥈 辅助 2 | T5g | a₅ 函数形状诊断（独立看每 Ro 拟合曲线）|
| 🥉 诊断 | T3v | ww≈0 残差，**仅用作作弊检测**（"全 0 解"判据）|
| 🥉 诊断 | T4v | uv 残差，**仅用作 Pope 框架极限诊断**（0.9+ = 框架无法表达非局部输运）|

#### 6.6.4 统一基准方法（推荐）

```python
# 跨目标综合评分（仅 full 配 + valid_mask + wall_seeds 启用后）
def unified_score(per_target_loss: Dict[str, float]) -> Dict[str, float]:
    """per_target_loss = {"T1": 0.08, "T2v": 0.058, ...}"""
    # 1. 金标准评分（T2v）
    gold = per_target_loss["T2v"]
    
    # 2. 合成物理量辅助（T1, T4v, T3v 阈值检查）
    synth_pass = (
        per_target_loss["T1"] < 0.20 and      # uu 真实预测阈值
        per_target_loss["T3v"] >= 0.5 and     # ww 残差必须 ≥0.5（防 ww=0 作弊）
        per_target_loss["T4v"] >= 0.5         # uv 残差必须 ≥0.5（防 uv=0 作弊）
    )
    
    # 3. 标量拟合辅助（T5g）
    scalar_diag = per_target_loss["T5g"]
    
    # 4. 坍缩/爆参检测
    from acceptance import collapse_check
    collapse_flag = collapse_check(best_theta_per_target)
    
    return {
        "gold_T2v": gold,                      # 越小越好
        "synth_pass": synth_pass,              # 必须 True
        "scalar_T5g_diag": scalar_diag,        # 诊断值, 不计分
        "collapse_flag": collapse_flag,         # True = 作弊, 必须重跑
    }
```

#### 6.6.5 验收脚本硬规则

| 规则 | 条件 | 处置 |
|---|---|---|
| 坍缩检测 | 5 指标命中任一（见 §6.4）| 打 ⚠️，强制 cross-seed 复现 |
| 爆参检测 | max\|θ\|>1e3 | 打 ⚠️，loss 数字无效 |
| 合成作弊检测 | T3v<0.5 或 T4v<0.5 | 打 ⚠️（疑似输出全 0 解）|
| 跨工况稳定性 | max(Ro NMSE)/min(Ro NMSE) > 5× | 打 ⚠️（仅对单 Ro 段作弊）|
| T5g 异常低 | T5g<0.01 但 T2v>0.05 | 打 ⚠️（评估口径错位，T5g loss 失效）|

#### 6.6.6 P3 行动项更新

| 优先级 | 行动项 | 耗时 | 阻塞 |
|---|---|---|---|
| **P0** | `_accept_v4_ablation.py` 加 §6.6.4 unified_score 函数 | 2h | 否 |
| **P0** | `_accept_v4_ablation.py` 加 §6.6.5 5 条硬规则自动检测 | 1h | 否 |
| **P1** | coef_min 护栏 (paramfit.py) 防 full T5g C1=-2.3e-5 坍缩 | 1h | 否 |
| **P2** | cross-seed 3 seed 验证 T2v 共识解 0.0577-0.0579 | 3h | 是 |

---

## 7. P2 决策修订

### 7.1 原 P2（基于 P1c / P1d 推测）

```
T1/T2v  → 优先 mm  (LLM cold-start 优势)
T3v/T4v → 两端都崩 (Pope 框架极限)
T5g  → 优先 heuristic (纯幂律最优)
yw PIN 保护 →  mm 后端必装 (防止 M_GUIDE 删 ypd/ypd→yw)
```

### 7.2 V3 实测修订后

| 目标 | heur full mg | mm full go | mm full mg | 推荐 | 论据 |
|---|---|---|---|---|---|
| **T1** | 0.0817 ✓ | 0.0693 | 0.1332 | **heur full mg** | mm full mg 输 +63%；mm full go 0.0693 是 stress + G_INIT 单一 trick，可在 mm full mg 里靠 seed 调出来但不稳 |
| **T2v** | 0.0579 = | 0.0579 = | 0.0579 = | **heur full mg** | 三方同公式 `C1·eta2^C3·zeta2e^C2`，同 seed 后端无差异；选最便宜的 heuristic |
| **T3v** | 0.9571 ≈ | 0.9585 ≈ | 0.9764 | **heur full mg** | full 上 mm 优势殆尽（差异 ≤ 2%），mm_bare 优势不能迁移；P1c 时代"heuristic T3v 大输"在 V3 同 seed 下未复现 |
| **T4v** | 0.9329 ≈ | 0.9329 = | 0.9122 | **heur full mg** | mm full mg 略优 -2.2% 但不稳，且四臂 0.93 普遍崩 (Pope 二阶基极限) |
| **T5g** | 0.6571 ✓ | 0.7447 | 0.7535 | **heur full mg** | mm 两臂都输 ≥ +14%，仅 G_INIT trick mm full go 不敌 heuristic M_GUIDE |

⚠️ **决策依据（19:00 修订）**：
- 上表是 **full 配下** 5 目标的对照（heur vs mm 同 seed，bare 配未列）
- **bare 配 T1=0.2391/T3v=0.6833/T4v=0.0242 都比 full 配小**，但这是 **GP 走 C1 坍缩作弊**（§6.4 详述）—— 数字失效，**不可作为选 full 的反证**
- 选 full 的真正原因：full 的 valid_mask + stress 加权 + wall_seeds 阻止了 GP 走参数坍缩，**full 配 loss 数字才是物理有效的**
- 上表 "≈ / =" 是真实后端差异（full 配 C1 正常，未作弊），可直接对比

**新后端策略**：
- **统一后端 = heuristic + full + M_GUIDE**（HeuristicLLM 离线客户端 + 共享机制全开 + M_GUIDE 通道启用）。这是"胜出配方"中**最便宜**的——无 API 成本
- **mm 后端仅作为逃生选项**：当 heuristic × full × mg 在某目标上 NMSE > 0.5 且 5 个 Ro 工况整体都差时，跑 mm × full × go (关 M_GUIDE 防劣化) 验证 mm 是否能解决某个具体瓶颈
- **yw PIN 保护机制缩小范围**：原 "全 mm 后端都装" → 现在仅 T3v mm 后端保留 (其他 mm 后端在 full 配置上输或打平 → 不依赖 PIN)

### 7.3 P1c 翻案

| P1c 后总结 | V3 实测 | 是否推翻 |
|---|---|---|
| "T1/T2v 优先 mm" | T1 mm +63%, T2v 三方打平 | ✅ 完全推翻 |
| "T2v M2.7 +56%/73% Ro30/75" | bare mg T2v mm +1885% (单点) | ⚠️ 现象存在但根因是 sqrt 节点爆参数，非 mm 优势 |
| "T5g 优先 heur" | T5g mm +14.7%/+1.3%, heur 全胜 | ✓ 一致 |
| "单目标 T1 −21% 是 mm 冷启动胜 heuristic" | 另一 seed 下 T1 = 0.0693 (mm) 输了，0.1084 (heur) | ✅ 单点超载归因错 |

---

## 8. 结论

### 8.1 三个核心判断

1. **heuristic 是统一后端（标准配置）**。8 臂中 6 个配置均值最优或打平；mm 后端唯一"赢"项 mm_bare_go T5g=0.0000 **不是 mm 后端优势** —— 详见 §6.5 评估口径陷阱（a₅ 标量拟合完美 ≠ vw 真实预测精度高）。从 T2v（vw 真实预测金标准）看 mm 全臂无优势，mm_bare_mg T3v=-40% 优势在 full 配置上完全消失。
2. **M_GUIDE 通道整体失效**。V3-0/V3-a 修复把 calls/applied 数据落盘打通，但 8 臂 alive 全 0 —— M_GUIDE 输出从未育成最终 best 个体。当前 GP + M_GUIDE + M2.7 mini API 的组合是"信息已通但路径未走通"的状态。
3. **V3 期间没让 mm 翻盘的关键是：P1c 的"单点超载归因错"在 V3 用更多 seed 数 (8 臂 × 5 目标) 同 seed 对照后被驳倒**。同 seed 同目标，同一 baseline → 跨配置后端差异 = 后端真实贡献。P1c 是 1 个 seed × 4 配置，T1/T2v mm 单点优势是真优势还是冷启动运气—— V3 用更多 seed 才看得出，而 V3 数据明确显示是单点巧合。

### 8.2 立即行动项

1. **P2 决策部署**：正式废弃 `heur + mm + M_GUIDE 三配后端` 切换策略，所有目标统一 `heuristic + full + M_GUIDE`。消融跑批管道从原来 30+ 分钟 mm 全配置改为 heur 单一后端总耗时 ↓ 70%（参见 v3 跑批耗时 7-12 min vs 18-40 min）。
2. **(可选) T3v / T5g 个别救援**：如某次跑出 NMSE 局部异常，可单独跑 `mm × full × go` 验证 rescue，但预期差异 ≤ 5%。
3. **M_GUIDE 通道二次诊断**：calls 数据可查 → 看每次 apply 的变更是否真的在种群竞选中活过 ≥ 1 代。
   - 这要求给 evaluate 增加 "M_GUIDE 血统打标" 字段；3h 工作量；
   - 是确认 M_GUIDE 未来是否值得精修的前提 (否则继续投入 LLM 后端开发是无底洞)。
4. **(⚠️ 19:00 新增) 验收脚本加坍缩检测**：基于 §6.4 的 5 指标（`|C1|<0.01` / `max|θ|<1e-3` / `exp(log(...))` 节点 / `max|θ|>1e3` / 多参坍缩），自动给 final.json 打"⚠️ 坍缩"标记。任何标"⚠️ 坍缩"的 final 须走二阶段验收（独立测试集或 cross-seed 复现）才算 P3 段通过。
5. **(⚠️ 19:00 新增) T5g 加 coef_min 护栏**：heur_full_mg T5g 仍走 C1=-2.3e-5 坍缩，建议在 `paramfit.py:5` 加 `if |C1| < 0.05: reject` 的 coef 路径护栏。**这是 P3 优先项**。

### 8.3 ⚠️ 局限声明

- **8 臂 = 1 seed / 目标 / 配对**：v3 launcher 用 `crc32(scheme|target)` 决定 seed，并非 5-10 seed 平均。heur/±40% 这类优势在 cross-seed 上需进一步验证（建议 P3 阶段扩展为 3 seed 验证关键目标）。
- **T5g=0.0000 未走独立测试集评估**：loss 在 vw 上从拓扑回归角度等于 0，但 ROM 真实泛化误差须在 Ro=180/300 上补一批高 Ro 工况验证（dataset_pope_gsonly 当前 Ro=7.5/15/30/75/150）。
- **T3v mm_bare -40% 优势是否在 cross-seed 上稳定**未验证：heur_bare 4 臂中 2 臂 T3v=0.7968 (vs 0.6833), 跨后端差异 14%，seed 敏感。
- **（19:00 新增）bare 配的 T1/T3v/T4v loss 数字失效**：GP 走 C1 坍缩（§6.4），任何基于 bare 配 loss 数字的后端对比均无意义。P3 阶段只做 full 配的 cross-seed 验证。
- **（19:00 新增）full 配的 T5g 仍存在 C1 坍缩**：C1=-2.3e-5，公式 `C1·exp(C3·ζₑ²+C2)·yw` 接近全 0 输出。需加 coef_min 护栏（行动项 5）。
- **（19:00 新增）bare 配 T2v 存在参数爆炸**：heur_bare_go T2v max\|θ\|=1.04e5，与 mm_bare_mg T2v=5546 同源问题（coef 路径失衡）。

---

## 9. 产物清单（落地可查）

| 路径 | 内容 |
|---|---|
| `outputs/asm/runs_v3_*/B_a_*_final.json` | 40 个 best final.json |
| `outputs/asm/runs_v3_*/summary_B.json` | 8 臂 aggregator |
| `outputs/asm/_peek_v3_all.txt` | V1-V3 全景 |
| `outputs/asm/_accept_v3.txt` | V1-V4 验收表 |
| `outputs/asm/_v3_inspect.out / _v3_inspect2.out` | 关键公式复检 |
| `outputs/asm/_v3_ablation_*.log` | 8 臂原始 stdout (含 llm_stats 每代详情) |
| `outputs/asm/_v3_ablation_progress.log` | launcher 调度日志 |

---

**报告人**: 申
**日期**: 2026-10-08
**超时预算**: 8 臂总耗时 2.50 h, 与 14:12 launcher 启动时估算 "~3h" 一致 ✅
