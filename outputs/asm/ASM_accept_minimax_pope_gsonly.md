# P1c 验收报告 — minimax × pope_gsonly vs heuristic × pope_gsonly

**日期**: 2026-10-08
**实验**: 用户指令"在 gsrt/gsonly 基础上用 minimax 大模型 LGF 再做一次，希望结果比 heuristic 好"
**对照**: minimax M2.7 × `dataset_pope_gsonly.npz` (9 列, T5g 基底) vs heuristic × 同数据集
**种子**: 同 (rc_asm_run.py:152 `crc32("B|Tx")`, 由 target 名哈希, 单因子公平)
**预算**: 150 s/目标 × 5 目标 = 28 分钟总
**总耗时**: 11:03:51 → 11:31:44 = **27.9 分钟**

---

## 0. 实验设计

| 项 | 值 |
|---|---|
| 后端 | minimax (MiniMax-M2.7), key 从 `C:\Users\lenovo\.config\evoscientist\config.yaml` 注入 |
| 数据集 | `dataset_pope_gsonly.npz` (9 列: 8 v-不变量 + yw, T5g 基底) |
| Tag | `minimax_pope_gsonly` |
| 目标 | T1 / T2v / T3v / T4v / T5g (5 个) |
| Fitness | `stress` (vw=5 / uw=3 / 其它=1), `wall_boost=8`, `fit_mask=valid_fit` |
| Launcher | `scripts/_run_minimax_gsonly.py` (复刻 P1b `_run_minimax_wy.py v2`, 加防休眠 + UTF-8 日志) |

**与 heuristic × gsonly 唯一差异**: 后端 (minimax vs heuristic). 其它全部对齐.

---

## 1. 5 目标 stress loss 对比 (越小越好)

| T | heuristic | minimax | delta | yw_h | yw_m | n_h | n_m |
|---|---|---|---|---|---|---|---|
| T1 | 0.0817 | **0.0647** | **-0.0170 (-21%)** ✅ | Y | N | 9 | 13 |
| T2v | 0.0579 | 0.0583 | +0.0004 (+0.7%) 平 | N | N | 10 | 13 |
| T3v | 0.9780 | 0.9807 | +0.0027 (+0.3%) 输 | N | N | 8 | 10 |
| T4v | 0.9329 | 0.9329 | +0.0000 (=) 同 | N | N | 10 | 10 |
| T5g | **0.3579** | 0.6057 | **+0.2478 (+69%)** ❌ | N | Y | 10 | 13 |
| **mean** | **0.4817** | **0.5284** | **+0.0468 (+9.7%)** ❌ | | | | |

(n = Graph 节点数; yw = best 公式含 yw; **粗体 = 该指标下胜出方**)

---

## 2. 5 Ro NMSE 详情 (best['metrics'])

```
T    后端      Ro7.5   Ro15    Ro30    Ro75    Ro150   mean
T1   heur      0.106   0.075   0.036   0.058   0.133   0.082
T1   minx      0.118   0.059   0.065   0.028   0.053   0.065    ← 中高 Ro 大赢
T2v  heur      0.042   0.052   0.060   0.068   0.068   0.058
T2v  minx      0.120   0.065   0.026   0.018   0.062   0.058    ← 高 Ro 赢低 Ro 输
T3v  heur      0.948   0.906   0.978   1.067   0.990   0.978    ← 双方都崩
T3v  minx      0.953   0.913   1.007   1.061   0.970   0.981
T4v  heur      0.985   0.911   0.906   0.977   0.885   0.933    ← 完全相同
T4v  minx      0.985   0.911   0.906   0.977   0.885   0.933
T5g  heur      0.160   0.306   0.447   0.301   0.576   0.358
T5g  minx      0.767   0.855   0.770   0.244   0.393   0.606    ← Ro7.5/15/30 爆炸
```

**T1 minx 高 Ro 减半**: Ro75 NMSE 0.058→0.028 (-52%), Ro150 0.133→0.053 (-60%).
**T5g minx 低 Ro 爆炸**: Ro7.5 NMSE 0.160→0.767 (+379%), Ro15 0.306→0.855 (+179%), Ro30 0.447→0.770 (+72%).

---

## 3. V2 vw 中心峰 (|y/h|<0.15, T5g 公式驱动; 阈值 > 0.03)

| Ro | DNS | heuristic | minimax | h_pass | m_pass |
|---|---|---|---|---|---|
| Ro7.5 | 0.037 | 0.052 | 0.027 | ✓ | **✗** |
| Ro15 | 0.045 | 0.067 | 0.048 | ✓ | ✓ |
| Ro30 | 0.059 | 0.068 | 0.053 | ✓ | ✓ |
| Ro75 | 0.054 | 0.051 | 0.030 | ✓ | ✓ |
| Ro150 | 0.034 | 0.055 | 0.044 | ✓ | ✓ |
| **达标** | | | | **5/5** | **4/5** |

---

## 4. V3 / V4 验收

| 验收项 | 要求 | heuristic | minimax |
|---|---|---|---|
| V3 yw 占用 (best 公式含 yw) | ≥3/5 (旧标准) | **1/5** | **1/5** |
| V4 反演健康度 max\|a\|@fit | < 1e3 | **187** ✅ | **187** ✅ (同数据集) |

V3 双方都 1/5 实际上是 gsonly 实验的接受现实 (gsrt 报告已声明: T1 best 含 yw 已足够驱动近壁);
V4 数据集层面的健康度, 与后端无关.

---

## 5. best 公式对比 (DSL 形式)

| T | heuristic | minimax | 节点数比 |
|---|---|---|---|
| T1 | `C1·(exp(C3·√η²)·C2·log(yw))` | `(C2·ζ2e/(C4·ζ2e+C3))/(η²+C1)` | 9 → 13 |
| T2v | `C1·exp(C3·log(η²))·exp(C2·log(ζ2e))` | `(C5·η²+C3·C4·s2o2v+C2·s3)/(C1·s2o2v+s2o2e)` | 10 → 13 |
| T3v | `C3 - C1·exp(C2·log(ζ2e))` | `C1·exp(C3·log(η²))·exp(C2·log(ζ2e))` | 8 → 10 |
| T4v | `C1·exp(C3·log(η²))·exp(C2·log(ζ2e))` | (与 heur 完全相同) | 10 = 10 |
| T5g | `C1·ζ2v^C3·ζ2e^C2` (纯幂律) | `(C4·η²)/(η²+C3) - (C2·s2o2e - C1·yw)` (减法) | 10 → 13 |

**minimax 公式普遍更复杂** (平均 12.4 节点 vs heuristic 9.4 节点, +32%).

---

## 6. 决定与解读

### 6.1 用户期望"比 heuristic 好" — **整体未达成, 但 T1 显著赢**

| 维度 | 评估 |
|---|---|
| 5 目标 loss 均值 | **输 9.7%** ❌ |
| vw 中心峰 5/5 | **输 1/5** (Ro7.5 失分) ❌ |
| yw 占用 | 持平 1/5 (gsonly 实验已接受此现实) |
| T1 单独 | **赢 21%** (中高 Ro 减半) ✅ |
| T2v 单独 | 持平 0.7% |
| T3v/T4v | 持平 (双方都崩在 Pope 框架天花板) |
| T5g 单独 | **输 69%** ❌ |

### 6.2 三个核心发现

**发现 1: minimax 公式普遍复杂化, 但损失不一定改善**
- 平均节点数 +32% (9.4 → 12.4)
- 复杂化主要来自 LLM 物理先验鼓励分数结构 (分子/分母) 和多分支加法
- 但在 a_m 这种**单调正 + 多工况外推**问题上, 纯幂律往往最优
- T5g 是典型反例: 减法结构在 Ro7.5/15/30 弱旋转下值域不对, 5 Ro NMSE 0.77-0.86 爆炸

**发现 2: minimax 对中高 Ro (强旋转) 有结构偏好**
- T1: Ro75/150 NMSE 减半, Ro7.5 略输
- T2v: Ro30/75 NMSE 减半, Ro7.5 翻倍
- 物理上 T5g 已经显式含旋转, minimax 似乎把"旋转偏好"扩散到其他目标
- 后果: 高 Ro 工况 (Ro75/150) 显著赢, 低 Ro (Ro7.5) 显著输 — **不公平地牺牲低 Ro 换高 Ro**
- 这与 stress-weighted fitness (vw=5/uw=3 隐含高 Ro 占比大) 协同放大

**发现 3: M_GUIDE 保 yw 机制在 gsonly 上又失守**
- P1b (pope_wy) 修过 `M_GUIDE` 注入 PHYSICAL CONTEXT + FORBIDDEN ypd — T1 公式含 `exp(C2/yw)`
- P1c (gsonly): T1 公式 `(C2·ζ2e/(C4·ζ2e+C3))/(η²+C1)` **不含 yw**; T5g 公式才含 yw (`... - C1·yw`)
- 物理: T5g 显式含旋转 → minimax 误判"基已对症" → 主动删 yw
- 影响: 削弱了 T1 主导的 uu/vv/ww 近壁阻尼 (V1 fit 区 worst 预期 uu/vv 比 heuristic 略输)

### 6.3 与 P1b 教皇对比的同构性

P1b (minimax × pope_wy, T5v 基底, 2026-10-04) 的结论:
- M_GUIDE 不再删 yw (T1 best 含 `exp(C2/yw)`)
- 但 V1 近壁 uu/vv 仍劣于 heuristic (71.6/361 vs 13.1/19.8)
- 瓶颈在 Pope 框架非后端

P1c 重复了同构现象:
- M_GUIDE 在 T1 失守 (无 yw) — 但 T5g 含 yw (净效果 1/5 vs 1/5, 持平)
- T5g 是核心胜负手, 输给 heuristic 69% — 这次瓶颈仍在 Pope 框架, 但**额外是 T5g 公式选择错了**

**Pope 框架的硬约束**:
- T3v/T4v 双方都崩 (loss 0.93-0.98) — 二阶张量基的 aₘ 需要非局部输运, 8 不变量解不开
- T1/T2v/T5g 是 Pope 框架内能优化的目标
- T1 minx 赢 21% 证明: 在 LGF 可解的目标上, minimax 后端确实有优势
- T5g minx 输 69% 证明: 在 a_m 形式敏感的物理目标上, 物理先验**引导错了方向**

---

## 7. 局限声明 (诚实口径)

- **本验收的最终 NMSE 重构**有 RuntimeWarning 干扰 (np.log 在 yw→0 处的 -inf 被 numpy warning 触发 PS 错误状态);
  v3 版已用 `warnings.filterwarnings("ignore")` 屏蔽, V2 5 Ro 中心峰数据已正确产出
- **gsonly npz 不含 B_T5g 列** (rc_pope_gsonly.py:84-89 的 pop 漏写, 反演时用内存值), 验收脚本从
  `dataset_pope3_wy.npz` 取 T5g 列兜底 — 不影响 GP 训练, 只影响外部验收. **建议在 rc_pope_gsonly.py 加
  `out[p + "B_T5g"] = basis6["T5g"]` 写盘** (下次实验前修)
- **minimax 仅跑 1 个 seed** (target 名 hash 决定 seed, 与 heuristic 同), 单 seed 报告平均改善方向,
  多次 seed 才能确认显著性 — 但 v1b 报告已是单 seed, 同口径可比
- **minimax API 调用约 5-6 次/目标** (1 LLM/GUIDE 周期), 28 分钟总时长符合预算
- **5 目标 loss 均值是最严口径**, V1/V2 是更细致的诊断, 整体均值输赢是最终结论

---

## 8. 采纳决定与下一步

| 项 | 决定 |
|---|---|
| minimax × pope_gsonly | ❌ **整体不采纳** (5 目标均值输 9.7%, vw 中心峰 4/5) |
| 但 T1 单独 formula | ✅ 可考虑作为 T1 单独 baseline (中高 Ro 减半, 低 Ro 略输) |
| T5g 公式攻坚 | 仍从 heuristic T5g `C1·ζ2v^C3·ζ2e^C2` loss 0.358 出发; **不要再让 minimax 推 T5g** |
| 后端选择策略 (更新) | **T1/T2v 优先 minimax** (中高 Ro 改善, 但要警惕 yw 失守); **T5g 优先 heuristic** (纯幂律最优); T3v/T4v 无论后端都崩, **Pope 框架无解** |

### 下一步可选方向

1. **minimax 限定 yw PIN** (在 `apply_mutations` 强制要求 T1/T2v/T5g best 含 yw, 否则注入惩罚项)
2. **T1 用 minimax + T5g 用 heuristic 的混合策略** — 后端分配到目标级
3. **Pope 框架突破**: 引入非局部项 (压力应变, TG 涡卷输) — 这是结构创新, 后端帮不上
4. **GLM5.1/DeepSeek V4 复跑 P1c** (P2 路线, LGF_LLM_API_KEY 在 evoscientist config 已有 key 但未启
   用) — 验证 minimax 退化是否 minimax 专属

---

## 附: 文件清单

- 数据集: `dataset_pope_gsonly.npz` (9 列, 已就绪)
- 跑批产出: `runs_minimax_pope_gsonly/B_a_{T1,T2v,T3v,T4v,T5g}_{final.json,log.jsonl,jsonl}` + `summary_B.json`
- Launcher: `scripts/_run_minimax_gsonly.py` (防休眠 + UTF-8 日志 + key 注入)
- 验收: `scripts/_accept_p1c_v3.py` + `outputs/asm/_accept_p1c.log`
- 日志: `runs_minimax_pope_gsonly_stdout.log` / `_stderr.log` (~870KB stderr, 全是 lambdify RuntimeWarning, 无关)
