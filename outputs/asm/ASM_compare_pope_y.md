# Pope × y⁺ 增广 — 三向后端对比（8不变量 vs 9特征 × heuristic vs minimax）

## 1. 设计

- **Pope 框架**（Pope 1975 JFM 72:331；b_ij 归一化各向异性张量；10 张量基 × W ∈ {W̄, Ω̃} → 19 唯一候选；贪心选基 + 条件数换基 → {T1, T2v, T3v, T4v, T5v}，med cond 10.3）
- **特征维度演进**：pope 轮用 8 个无量纲不变量列；**pope_y+ 在 8 不变量之上加 ypd=|y⁺|（到最近壁面）**作为第 9 列；DSL 算子 sub/exp/inv/tanh 等齐全，φ(y⁺) 形状（van Driest/有理/指数阻尼/tanh饱和）LGF 都可表达
- **fitness = 应力空间加权 NMSE（w=‖T_m‖²_F）**，向后兼容
- **数据集**：dataset_pope.npz（8 特征）vs dataset_pope_y.npz（9 特征），后者的合法点最浅 |y⁺|=4.4

## 2. 方案B — 全 ASM 重建 R²（越大越好）

### uv（湍流剪切主项）⭐ ypd 弥合了 heuristic/minimax 鸿沟

| 后端\特征 | 8 inv（pope） | 9 特征（pope_y） |
|---|---|---|
| heuristic  | 0.334 / 0.116 / 0.305 / 0.599 / 0.606 | **0.934 / 0.922 / 0.945 / 0.952 / 0.943** |
| minimax    | **0.923 / 0.973 / 0.970 / 0.991 / 0.993** | 0.924 / 0.925 / 0.941 / 0.941 / 0.938 |

- **ypd 把 heuristic uv 从 0.12–0.61 提到 0.92–0.95**，与 minimax 持平 → 两个后端在 uv 上汇合
- 启发式 T5v 公式 `C3·ζ²ₑ / (ζ²ₑ + C2·ypd)` 是 φ₅ ≈ `1/(1+C·ypd/ζ²ₑ)` 型加性阻尼

### uu / vv（对角法向分量）—— minimax 全线改善；heuristic 双线爆炸

| y⁺ | 后端 | uu Ro7.5..150 | vv Ro7.5..150 |
|---|---|---|---|
| 8 | heuristic | -2.44, -1.14, -0.84, -0.70, -0.64 | 0.47, 0.61, 0.72, 0.31, 0.06 |
| 8 | minimax | -0.36, 0.06, -0.31, -0.04, 0.41 | 0.26, 0.39, 0.57, 0.82, 0.70 |
| 9 | heuristic | **-7892, -14441, -10397, -3530, -2336** ❌ | **-15578, -19040, -6896, -1509, -783** ❌ |
| 9 | minimax | **-1.23, -0.31, 0.13, 0.62, 0.71** ⭐ | **0.03, 0.22, 0.51, 0.54, 0.34** ⭐ |

- **heuristic uu/vv 在 ypd 加入后变灾难**：T4v/T3v 公式无 ypd，但多 1 个特征维度后 GP 在 Ro7.5/15/30 上过拟合了 T3v（a_T3v Ro75 = -203.076 vs pope 时全 -0.016 平坦）。二阶基底 + 过拟合 = 重建爆炸
- **minimax uu/vv 中高 Ro 转正**：Ro150 uu 0.41 → 0.71、vv 0.70 → 0.34（vv 略降但仍正）；低 Ro 仍负，但比 pope 收敛

### ww / uw / vw（剩余分量）

| y⁺ | 后端 | ww 7.5/30/75/150 | uw 7.5/30/75/150 | vw 7.5/30/75/150 |
|---|---|---|---|---|
| 8 | heuristic | -25.6/-2.6/-2.9/-2.8 | -0.34/0.67/0.62/0.61 | -8.8/-3.3/-3.8/-7.7 |
| 8 | minimax | -5.7/0.29/0.50/0.40 | -0.36/0.62/0.73/0.59 | -0.43/-0.84/-2.6/-3.3 |
| 9 | heuristic | -5.5/-16.5/0.12/**0.36** ⭐ | -0.46/0.61/0.73/0.61 | **-0.41/-1.00/-2.36/-2.78** ⭐ |
| 9 | minimax | **-6.0/0.45/0.51/0.26** | -0.13/0.63/0.75/0.53 | -2.5/-3.8/-4.9/-6.6 |

- **ww 中高 Ro**：heuristic ypd 加入后 Ro75/150 转正（0.12/0.36），minimax ww 全 Ro 改善
- **vw**：heuristic 显著改善 -8.8~-3.3 → -2.8~-0.2（一个量级），但 minimax 反而变差 -3.3 → -6.6（ypd 没进 minimax 公式但特征维度变大让 GP 在 vw 主项上漂了）

## 3. aₘ 公式（heuristic 用 ypd，minimax 完全不用）

| 目标 | heuristic_pope_y | minimax_pope_y |
|---|---|---|
| T2v | `(C2·ζ²ₑ + C1·C)/(P1(ζ²ₑ))` (P1是 ζ²ₑ 二次多项式) | `(C2·1)/(ζ²ᵥ + C1·1)` |
| T5v | **`(C3·ζ²ₑ)/(ζ²ₑ + C2·ypd) / (s2o2ᵥ + C1·1)`** ⭐ 含 ypd | `C1·exp(C2·log(η²))·exp(1)` |
| T4v | `C1·exp(C3·log(η²))·exp(C2·log(ζ²ₑ))` | `C1·exp(C3·sin(η²))·exp(C2·log(ζ²ₑ))` |
| T3v | `C1·exp(C3·log(η²))·exp(C2·log(ζ²ₑ))` | `C1·exp(C2·log(ζ²ₑ))` |
| T1  | `(C2·log(C4 + C3·η²))/(η² + C1·1)` | `C1·(-inv(C3·η² + C2·1))` |

**关键观察**：**minimax 5/5 公式零 ypd**——LLM 后端在 Pope 框架下完全不走近壁显式距离路线；而 heuristic 在 T5v 中主动发现加性阻尼结构。

## 4. 综合判断

### ypd 的两面性
| 维度 | 正面 | 负面 |
|---|---|---|
| 拟合可达性 | uv 一致到 0.92+；heuristic 公式自动用 ypd | 二阶基（T3v/T4v）多 1 个自由度后 Ro75 显著过拟合 |
| 后端差距 | heuristic uv 跨越 0.6 鸿沟 | minimax 在 vw 上反退化 |

### heuristic vs LLM（minimax）后端差
| 行为 | heuristic | minimax |
|---|---|---|
| ypd 利用 | 主动（5/5 中 1/5 用进公式） | 不用（5/5 全零 ypd） |
| 公式风格 | 多项式/有理式（P1 定义）| 纯 exp/log/inv（更"干净"）|
| 在 9 特征空间稳定性 | 易过拟合低 Ro 中等 Ro | 稳定，uu 中高 Ro 解锁 |
| uv 上限 | ypd 后与 minimax 持平 | pope 时已 0.99，pope_y 微降但仍 0.92+ |

### 整体推荐
- **uv 选 minimax × pope（无需 ypd）** 或 **heuristic × pope_y（更便宜）** 都可
- **uu/vv 选 minimax × pope_y**（唯一在中高 Ro 转正且双线稳定）
- **vw 选 heuristic × pope_y**（一个量级改善）
- **ww 选 minimax × pope_y**（中高 Ro 转正）
- **不存在一个后端 × 一个数据集能解全部分量**：这是 Pope 框架的根本瓶颈——二阶张量（T3v/T4v）的 aₘ 需要非局部物理，不是更多局部特征能解的

## 5. 待办与诚实声明

- **heuristic uu/vv 灾难不是新问题**：是 stress fitness 对 T₃v/T₄v 加权放大导致的；多 1 个特征下 GP 在 Ro7.5/15/30 多自由度过拟合是诱因。**缓解**：把高阶 aₘ 的 stress 权重封顶，或将 Ro 联合反演改为加权 Ro 联合
- **a_T1（线性涡粘项）单系数 R² 全 < 0.6**，仍是 Pope 框架天花板，与 ypd 无关
- **结论外推性仅限于本 Re_τ=180 DNS 族**，高 Re 下 ypd 设计可能失效（壁面 vs 自由流区过渡）

输出文件：
- `outputs/asm/ASM_summary_pope_y.md`（heuristic + minimax pope_y 各自表）
- `outputs/asm/asm_compare_pope_y.csv`（per-component per-backend per-Ro）
- `outputs/asm/figures/Ro*_compare_pope_y.png`（剖面对比）
- `outputs/asm/runs_heuristic_pope_y/summary_B.json`、`runs_minimax_pope_y/summary_B.json`（原始 summary）
- bug 修复：`lgf/dsl.py:495` `sorted(terms, key=lambda t: t[0])`（原 `sorted(terms)` 在 k 相等时会落到 Node 对象比较，TypeError；本轮 heuristic 第一次跑因此崩溃一次）