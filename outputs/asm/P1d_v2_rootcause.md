# P1d v2 根因诊断：为什么 4 臂改进"全部无效"

日期：2026-10-08 13:40（14:20 增补 V3-0 更正与 V3 实施记录）｜ 对象：`runs_minimax_pope_gsonly_v2/` + `runs_heuristic_gsonly_v2/`（4 臂 heur / minx / h_v2 / m_v2）
方法：逐代轨迹对比（`*_log.jsonl`）+ n_pop 种群规模审计 + 掩码挂载点代码定位 + gen0 初始种群指纹

---

## ⚠ 重大更正（14:20，证据优先）：真根因是 Graph.clone 不拷贝 theta

本文初版推断"M_GUIDE 调用失败静默返回 {}"——**不成立**。后续三层复现推翻了它：

1. **heuristic 臂链路单独复现完全正常**：analyze OK → M_GUIDE 返回 2 条 canned mutations → apply_mutations 返回 clone（`scripts/_repro_mguide_silent.py`）
2. **runlog.err 零条 `[LGF-LLM]` 失败日志**（launcher 开了 `LGF_LLM_DEBUG=1`）——MiniMax API 调用没有失败过
3. **微型冒烟（当前盘面代码 + heuristic）gen4 n_pop 仍不增长**——问题在管线内部，与后端无关

**真根因**：`lgf/graph.py:176-180` 的 `Graph.clone()` 只拷贝 `loss`/`metrics`，**不拷贝 `theta`**（初始 None）。`pipeline_lgf.py:108` `best_g = g.clone()` → gen4 触发 M_GUIDE 时 `attribution.analyze(best_g, best_g.theta=None, ...)` → `theta[:P_g]` 抛 TypeError → 被 `pipeline_lgf.py:155-156` 的 `except: pass` 静默吞掉。

**推论**：M_GUIDE 在 P1c/P1d 全部 6+ run（含 P1c 旧代码 run）**从未执行过一次**，与后端无关。本文原"结论 1（M_GUIDE 全灭因 API 失败）"作废；"G_INIT 是 LLM 唯一生效通道"的结论**反而被加固**——它不是"M_GUIDE 失效的副产物"，而是 M_GUIDE 从未运行这个事实的直接结果。n_pop 恒 40 在旧顶替制代码下本来就不可区分，但 B2 wall-seed 打印（`runlog.txt` 显示种群到过 43/44/45）证明 v2 进程用的确是新 pipeline，n_pop 回落 40 只因 select 在同代收缩。

---

## 一、证据链（全部对代码可追）

| # | 证据 | 来源 | 含义 |
|---|------|------|------|
| E1 | 6/6 run（3 目标 × P1c/v2）n_pop 全程恒 40 | `*_log.jsonl` GenLog.n_pop | ~~B1 追加制下 M_GUIDE 变异若存活 n_pop 必 >40~~ **已被更正**：真因是 clone 不拷贝 theta 导致 M_GUIDE 整块从未执行（见顶部更正） |
| E2 | T5g 25 代 best_loss 与 P1c 逐代完全相同，best DSL 字节级相同 | 轨迹对比 | T5g 两轮都是纯 GP 演化，LLM 中途引导零影响 |
| E3 | T1/T2v gen0 best DSL 即分叉（P1c T2v: 复杂分数 vs v2: `C2*zeta2v+C1`） | gen0 日志 | **G_INIT 初始种群两轮不同 → G_INIT 是 LLM 唯一真实生效通道** |
| E4 | forbidden_ops 只挂在 M_GUIDE→apply_mutations 路径 | `pipeline_lgf.py:138,147-148`、`llm_ops.py:594` | GP 自身路径（`random_graph` :81-84、crossover/mutate :114-127、templates）**完全不受掩码约束** |
| E5 | heuristic 臂 h_v2 vs heur 也分叉（T1 0.084 vs 0.0817） | V1 表 | pipeline 改动（B2 wall seeds / rng 流）本身改变轨迹，heuristic G_INIT 不调 API |
| E6 | T2v m_v2 top3 全是 `C3*exp(C4*eta2/yw)+...`，loss 0.5608 | v2 final.json | M_GUIDE 全灭（E1）→ 爆炸结构只能从 **G_INIT 初始种群**进入 |

## 二、四条结论

### 结论 1：M_GUIDE 中途引导在 P1c 和 P1d 上零存活（最重磅）

v2 的 n_pop 恒 40 直接证明 M_GUIDE 返回空 mutations——MiniMax 的 M_GUIDE 调用失败后**静默返回 "{}"**
（`llm_ops.py:77-89`，`llm_debug=False` 时无任何日志），或返回 JSON 中无 mutations 键。
P1c 的 T5g 与 v2 完全重放 → P1c 的 M_GUIDE 对 T5g 同样零影响。

**影响**：P1b 时代"30 次 M_GUIDE 调用"的结论已不适用于 P1c/P1d 跑批配置；P0/A1（per-case 归因）、A2（值域硬规则）对 M_GUIDE 的改进**根本没有被测到**——不是无效，是没进场。

### 结论 2：G_INIT 是 LLM 唯一生效通道（含副作用）

- T1/T2v 的 minx vs m_v2 差异全部来自 G_INIT 初始种群（E3）
- T2v m_v2 的 exp(eta2/yw) 爆炸：新 prompt 的"单调形式引导"（`llm_ops.py:502-503`）在 G_INIT 阶段就引入了 `exp(eta2/yw)`——yw→0 于壁面，指数爆炸；无 eval 层值域检查拦截（E6）
- 即：**A2 prompt 改进不是无效，而是通过 G_INIT 产生了新的失败模式**

### 结论 3：forbidden_ops 对 T5g 空转，P1c 归因需修正

掩码只过滤 M_GUIDE 建议路径，而该路径整体空转（结论 1）；T5g best 的 sub/div 结构来自 GP 自身（templates/random/mutate/crossover，E4）。
**修正**：P1c 报告中"T5g minx 减法结构爆炸 = minimax 误判/删 yw"的归因不成立——该公式与 LLM 中途引导无关，minx vs heur 的 T5g 差异全部来自 G_INIT。

### 结论 4：apply_mutations 存在未触发的隐藏缺陷

全部变异失败时返回**原图 clone**（非 None，`llm_ops.py:564,607`），`pipeline_lgf.py:149` 的 `ng is not None` 检查形同虚设。
若 mutations 非空但全失败，会把原图（可能带 perturb_param）以 loss=inf 追加进种群——污染而非引导。本轮因 mutations 恒空未触发。

## 三、V3 实施记录（14:20 完成，58 项测试全过）

| 项 | 改动 | 位置 | 冒烟验证 |
|---|------|------|------|
| **V3-0 根因修复** | `Graph.clone()` 补 `theta` 拷贝 | `lgf/graph.py:176-186` | M_GUIDE n_calls=1/n_applied=5, gen4 n_pop 12→17, best 改善 0.1739→0.1611 |
| V3-a 统计落盘 | `llm_stats`（ginit n_candidates/n_parsed；mguide n_calls/n_nonempty/n_applied/n_alive/last_error/last_exception；forbidden_filtered）进 final.json + summary_B.json + stdout 单行；`except: pass` 改为记录 traceback 并打印 | `lgf/pipeline_lgf.py`、`scripts/rc_asm_run.py` | smoke 输出 mguide dict 完整 |
| V3-b 掩码下沉 eval 层 | 主循环 eval 前与最终精修前 `_has_op(g.root, FORBIDDEN)→loss=inf`，覆盖 GP/templates/G_INIT/M_GUIDE 全部来源 | `lgf/pipeline_lgf.py` | forbidden_filtered=5, best 无 sub/div |
| V3-c 值域护栏 | 拟合后 `\|pred\|>range_factor×max\|y\|`（默认 20）→ loss += 10+1e4×越界占比 | `lgf/paramfit.py` `_max_frac_out_of_range` | 单测全过 |
| V3-d 全失败返回 None | apply_mutations 追踪 applied_any（perturb 命中参数 / replace 成功挂载），全失败返回 None | `lgf/llm_ops.py` | n_applied 只计真实生效 |

**附带发现修正**：初版报告"`*_final.json` cfg 明文存 api key"属实（本地文件，共享/入库前需脱敏）。

## 四、对"经验归因"问题的分析（heuristic 为何显得不输 M2.7）

见对话回复。要点：(a) 共享机制（B2 wall seeds、stress fitness、valid_fit）对两臂同等生效，确实抬高了 heuristic 下限；(b) 但只给 LLM 的经验（A1/A2 prompt、B3 掩码）因 M_GUIDE 从未运行**全部未被测到**，而唯一生效的 G_INIT prompt 改动反而制造了 T2v exp 爆炸；(c) heuristic 后端本身零物理知识（G_INIT 只有 x1+x2/x1*x2/平方三条平凡公式），其全部"聪明"来自共享机制 + GP；(d) P1c T1 minx -21% 是"LLM 冷启动胜 heuristic"的直接证据（当时 B2 未上线）。干净回答需要 2×2×2 消融：{heur, M2.7} × {全套机制, 裸配置} × {M_GUIDE on}。

---

## 附：原 V3 方案（2026-10-08 14:20 已全部实施，上表为实施记录）

| 项 | 改动 | 位置 | 预期 |
|---|------|------|------|
| V3-a | M_GUIDE 调用统计落盘：每次调用记录 成功/失败/mutations 数/存活数 + LAST_ERROR 进 final.json | `llm_ops.py` + `pipeline_lgf.py` | 先确认 v2 M_GUIDE 是否全挂（timeout？响应格式？），否则后续改进仍在盲跑 |
| V3-b | forbidden_ops 下沉到 **eval 层**：主循环 eval 前 `_has_op(g.root, FORBIDDEN) → g.loss=inf` | `pipeline_lgf.py` ~3 行 | 覆盖 GP/crossover/G_INIT/templates 全路径，T5g 的 sub/div best 将被真正排除 |
| V3-c | 值域护栏：fit 后 `\|pred\| > 20×max\|y\|` → loss 罚 | `paramfit.py` | 堵 exp(eta2/yw) 类壁面爆炸（T2v 教训） |
| V3-d | apply_mutations 全部变异失败 → 返回 None | `llm_ops.py` | 消除原图 clone 混入种群的隐患 |

建议执行顺序：**V3-a 必须先行**（不改它，v3 重跑还是盲的）；V3-b + V3-c 随后一并上；V3-d 顺手修。

## 附：遗留风险

⚠ `*_final.json` 的 cfg 字段**明文保存 api key**（`llm_api_key: sk-...`）。本地文件风险低，但如需共享 run 目录或入库，应先脱敏。
