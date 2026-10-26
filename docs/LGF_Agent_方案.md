# LGF → LGF-Agent 升级方案

> 版本 v1.0 ｜ 2026-10-16 ｜ 目标读者：Wei
> 参考：SRHarness（清华 FIB Lab, arXiv:2609.35501）｜ Tacke et al., *LLM-driven design of physics-constrained constitutive models: two agents are better than one*（2026, DOI 10.1007/s44379-026-00099-3 / arXiv:2605.23754）｜ Du et al., *Large language models for automatic equation discovery of nonlinear dynamics*, Phys. Fluids 36(9), 2024, DOI 10.1063/5.0224297

---

## 0. 一句话结论

LGF 现在是**一条固定的双循环管线 + 2 个无状态 prompt 钩子**。把它升级成 LGF-Agent，本质不是"套一个 agent 壳"，而是把三件目前**隐式写死在 `pipeline_lgf.py` 里**的东西显式化：

| 显式化什么 | 现状 | 升级后 |
|---|---|---|
| **科学动作** | LLM 只能做两件事：生成初始候选（G_INIT）、改写候选（M_GUIDE） | 统一工具接口，LLM 可主动调用分析/拟合/评估/提交（≈SRHarness 的 9 动作） |
| **科学状态** | 内存种群 `pop`，进程结束即失；只有 top-8 落盘 | 候选-证据契约 + append-only archive + Pareto 投影视图 |
| **轨迹调度** | R=1, C=1, L=generations, K=1（硬编码） | R-C-L-K 显式可配，支持重启/分支/溯源/恢复 |

**再加一个独立审查者（Inspector）角色**，把你已经在用的"验收 5 硬规则"从**跑批后人工事后检查**提升为**入池前机器裁决**，并且裁决方式是**数值探针**而不是文本断言。

---

## 1. 现状诊断（逐条对照三篇文献）

### 1.1 动作空间：注册了 5 个 op，实际只接了 2 个

`lgf/llm_ops.py` 定义了 5 个外层算子（T_PROP / G_INIT / H_DECOMP / M_GUIDE / E_CHECK）。实际调用点全库扫描结果：

| 算子 | 设计意图 | 调用点 | 状态 |
|---|---|---|---|
| `T_PROP` | 提议新结构模板 | 无 | ❌ 死代码 |
| `G_INIT` | 初始候选生成 | `pipeline_lgf.py:120` | ✅ 活 |
| `H_DECOMP` | 可分性分解假设 | 无 | ❌ 死代码 |
| `M_GUIDE` | 归因驱动定向变异 | `pipeline_lgf.py:268` | ⚠️ 活但存疑（见 1.3） |
| `E_CHECK` | 等价类排序 | 无 | ❌ 死代码 |

对照 SRHarness 的 9 个 action（`statistics_analysis / relationship_analysis / read_skill / evaluate_formula / submit_formula / constant_fit / call_pysr / call_sindy / code_executor`），差的不只是数量：

- SRHarness 的动作接受**表达式视图**（`log x_i`、`x_i/x_j`、残差 `y − f(x)`），不限于原始数据列；
- 我们的 LLM 看不到任何数值执行结果 —— `G_INIT` 只看数据画像字符串，`M_GUIDE` 只看归因摘要字符串。

而 LGF **已经具备**这些能力，只是没暴露给 LLM：`paramfit.fit_joint`（拟合）、`compiler.CompiledGraph`（求值）、`priors.run_priors`（对称/可分/量纲）、`eq_compat`（符号等价）、`gp.py`（爬山的全套算子）。

> **结论**：动作层升级的 80% 工作是"接线"，不是"造新轮子"。

### 1.2 状态层：没有持久状态，代价已经显现过

现状：候选存在内存 `pop` 里，跨代存活但**跨进程消失**；落盘只有 `{name}_final.json` 的 top-8 + `{name}_log.jsonl` 的逐代 best。

后果（这是最有说服力的诊断）：

- **T23 口径争议**（2026-10-16）：要判断"最优公式是否真的用了 T23 通道"，只能事后写 `_tmp_t23_contrib.py` 手工反算净贡献（0.0418→0.0617，~6% RMS）。
- **V4b 的"M_GUIDE alive=0"**：`llm_stats` 是事后打的补丁（`pipeline_lgf.py:74-79`），因为在状态层缺失的情况下，**"这次 LLM 干预到底有没有生效"根本不可观测**。
- **跨 run 复用为零**：T23-full 8 条 LGF + 12 条 PySR 的候选明明同族，却无法在状态层做统一 Pareto 排序 —— 只能靠 `rc_asm_t23full_agg.py` 离线拼表。

SRHarness 消融给了硬数字：**把 Pareto 视图换成"单一最优公式视图"，SA 从 74.47% → 66.67%**（去掉 best-formula 显式视图反而只掉到 72.97%）。我们的 `M_GUIDE` 恰好就是**只喂 `best_g`**（`pipeline_lgf.py:268`，`[(best_g, best_g.loss)]`）—— 正是文献证明会**让精炼偏向早期解**的配置。这条能解释 M_GUIDE 为什么弱。

### 1.3 轨迹层：R-C-L-K 被写死成 1-1-N-1

现状映射：

| SRHarness | LGF 现状 | 问题 |
|---|---|---|
| R（重启轮数） | 无（想换假设只能改配置重跑） | 无法在**一次运行内**用历史 Top-k 状态重建上下文 |
| C（独立分支） | 无 | 假设族之争（阻尼族 vs 对数律族 vs 张量基族）从未被并行探索 |
| L（精炼深度） | `generations=40` | 有，但被 `time_budget_s=600` 掐断实际代数 |
| K（局部采样） | 无（`temperature=0.7` 单采样） | K>1 + best-of-K 从未试过 |

SRHarness 轨迹消融（LSR-Transform-Anon, DeepSeek-v4-flash，SA 基线 74.47%）：

| 配置 | SA | 时间 |
|---|---|---|
| R1-C1-L30-K1（默认） | 74.47% | 14.9 min |
| R2-C1-L15-K1（更多重启） | 66.67% | 27.6 min |
| R1-C2-L15-K1（更多分支） | 72.07% | 23.0 min |
| R1-C1-L15-K2（局部采样） | 73.87% | 23.7 min |

**规律：同样预算下，加深精炼 > 增加轨迹多样性。** 这条直接否决"多跑几个种子/多开几条分支"的直觉方案 —— 对我们尤其相关，因为多 seed 恰恰是我们目前唯一验证过的稳健性手段。

### 1.4 生成 vs 审查：没有审查者，错误物理断言直接进 prompt

现状：`lgf/attribution.py` 的 `FEATURE_ROLES` + `llm_ops.py` 的 `apply_mutations`（PIN 保护、forbidden_ops 硬拦截）构成了**半套**约束系统，但：

- 它只拦截，**不裁决**；没有"哪个物理约束被违反、证据是什么、最小修复方向"的回执；
- 约束以**文本断言**形式写进 prompt，而非数值验证。

代价的直接证据（2026-10-16 发现）：

> `scripts/rc_asm_vw_t23_full.py:125-126` 注入 LGF 的 physics context 声称
> "T1s = S₂₃ = ½dW̄/dy 关于槽道中心**反对称**" —— **实测 T1_23/T5_23/T6_23/T10_23 与 b23 同为偶对称（even 得分 ≈ +1.00）**。
> 这条错误断言进了 prompt 若干轮，没有任何机制核对。

Tacke et al. 的核心结果正是冲着这类问题：加入 Inspector 后，"真正满足全部物理约束的导出模型"占比 **91%→100%（Opus 4.7）/ 37%→56%（Kimi K2.5）**，精度基本不降。他们的关键手法是：

> *"Whether a generated model satisfies the physical constraints is assessed numerically, by probing each constraint across a broad sample of deformation states, rotations, and perturbation directions."*

即 —— **每条约束都要被数值探针在大量状态/旋转/扰动方向上逐一验证**。这正是我们应该给"偶对称/反对称"这类断言配上的东西。

### 1.5 与 PoF 2024（Du et al.）的关系

该文的框架 = **两个交替迭代策略**：① LLM 当黑盒优化器（基于历史样本 + 性能做自改进）；② LLM 执行进化操作做全局搜索。

LGF 的 G_INIT/M_GUIDE 就是这两策略的雏形，但：

- 交替节奏写死（`llm_every=5`、`stagnation_window=12`），不可按证据自适应；
- 策略 ① 所需的"历史样本及性能"没有持久载体 —— LLM 每轮只看当前 best；
- 策略 ② 的进化操作由 `gp.py` 在**图空间**盲抽，LLM 从未被告知"上一代哪些交叉/变异成功了"。

---

## 2. 目标形态：LGF-A 五层 + 三角色

```
┌──────────────────────────────────────────────────────────────┐
│ L4 角色层   Creator(A)  │  Inspector(B)  │  Archivist(可选)    │  ← 生成/审查分离
├──────────────────────────────────────────────────────────────┤
│ L3 轨迹层   R-C-L-K 调度器 + provenance + 恢复 + 预算会计       │
├──────────────────────────────────────────────────────────────┤
│ L2 状态层   候选-证据契约 · append-only archive · Pareto 投影    │  ← 跨轨迹复用
├──────────────────────────────────────────────────────────────┤
│ L1 动作层   profile/prior/fit/evaluate/attribute/guardrail/    │  ← 统一工具接口
│             equiv/gp_search/submit/read_skill/code_exec        │
├──────────────────────────────────────────────────────────────┤
│ L0 领域层   asm 适配器  │  shipman 适配器  │  (未来域)          │  ← 已有先例
└──────────────────────────────────────────────────────────────┘
      运行时：tool-call 解析 · observation 压缩 · 沙箱 · 审计日志
```

**设计原则（沿用你既有的口径纪律）**

1. **诚实分开物理与建模**：Inspector 只裁决**可数值检验**的约束；无法数值化的物理直觉一律标 `UNVERIFIED_ASSUMPTION`，不得当作约束使用。
2. **不牺牲可审计性换性能**：agent 的目标是"物理合规 + 跨工况泛化 + 可追溯"，**不是比 PySR 快**（我们已诚实记录 PySR 快 15–100×）。
3. **legacy 路径必须保留**：`pipeline_lgf.run_lgf` 一行不动，新引擎并列，同预算对照。任何结论都要能回答"是 agent 带来的，还是模型带来的"。

---

## 3. 关键设计（落到文件）

### 3.1 L1 动作层 —— `lgf/tools.py`（新建）

统一接口：

```python
@dataclass
class ToolResult:
    ok: bool
    observation: str          # 进上下文的紧凑文本 (≤400 字符)
    evidence: dict            # 落盘的全量机器可读证据
    cost: dict                # {seconds, tokens, calls}
```

| 工具 | 复用实现 | 输入 | 返回 observation |
|---|---|---|---|
| `profile_data` | `priors.data_profile` | — | 变量值域/量级/相关矩阵前 5 |
| `prior_check` | `priors.run_priors` | 变量对 | 对称/可分/组合/量纲票 |
| `fit_constants` | `paramfit.fit_joint` | dsl, budget_s | loss, theta, 是否收敛 |
| `evaluate_dsl` | `paramfit.evaluate`/`compile` | dsl, theta | 逐 case NMSE + 值域 |
| `attribute` | `attribution.analyze` | dsl | 节点敏感度 top5 + 形状标签 |
| `residual_view` | 新（compiler 加减法） | dsl | **表达式视图**：残差统计、分 bin 误差、`resid` 可作后续 fit 的 y |
| `guardrail_check` | 新 `lgf/verify/hard_rules.py` | dsl | 9 条约束逐条 pass/violate + 证据 |
| `symbol_equiv` | `eq_compat` | dsl, dsl | 等价判定（SA 口径） |
| `gp_search` | `gp.py` 全套算子 | 父代, 代数, 算子开关 | 新 top-k + loss 轨迹（**GP 降级为一个动作**） |
| `submit_formula` | 新 | dsl | 冻结 + 触发 final refit |
| `read_skill` | 新 `lgf/skills/*.md` | 主题名 | 领域知识卡（ASM 张量基代数 / MMG 结构 / 对数律） |
| `code_exec` | 沙箱包装 | python 片段 | stdout 截断（**默认关闭**，见风险 R6） |

**关键点：表达式视图**（SRHarness 消融：移除基于表达式的视图 → SA 74.47%→67.57%，掉幅最大的一档）。我们有天然优势：`attribution.analyze` 本来就在算每个节点的子树值 `z_v`，把 `z_v` 提升为一等公民（可作为 `fit_constants` 的 y、`attribute` 的输入）即可，几乎零成本。

### 3.2 L2 状态层 —— `lgf/store.py`（新建）+ `lgf/projector.py`

**候选-证据契约**：

```python
Candidate = {
  "id": "c0417", "dsl": "...", "var_names": [...],
  "theta": [...], "param_names": [...],
  "metrics": {"loss_vw": 0.0514, "r2": ..., "rho": ..., "nmse_per_case": [...]},
  "complexity": {"nodes": 15, "edges": 12, "params": 4},
  "guardrails": {"collapse": "pass", "param_blowup": "pass", "synth_cheat": "violate", ...},
  "diagnostics": {"bin_errors": [...], "node_sens": [...], "loso": {...}},
  "provenance": {"traj": "r0/c0", "parent": "c0398", "op": "gp_search",
                 "gen": 17, "prompt_hash": "…", "tokens": 8123},
  "status": "admitted"           # proposed|fitted|admitted|rejected|submitted
}
```

**持久化**：`outputs/<domain>/agent/<task>/archive.jsonl`（append-only，可审计、可 diff、天然断点续跑）。与现有 `runs_*/{name}.jsonl` 风格一致。

**投影** `P_ρ(S_t)`：两种视图，默认开 Pareto，**single-best 模式保留为消融臂**：

- `Pareto(loss, complexity)` —— 主视图（文献证明这是收益来源）
- `TopK(best_loss)` —— 次要信号
- `SingleBest` —— **只用于复现文献消融 / 作为 M_GUIDE 现状对照**

有界化：每次注入 prompt 的候选数上限（建议 ≤8），被省略的候选仍在 archive 中，后续可重新纳入。

### 3.3 L3 轨迹层 —— `lgf/agent/scheduler.py`（新建）

把 R-C-L-K 显式化，配置字段进 `LGFConfig`（`_FIELD_KIND` 同步登记）：

```python
# ===== LGF-A (agent 引擎) =====
agent_mode: bool = False        # False → 走 legacy pipeline_lgf (行为完全不变)
agent_R: int = 1                # 重启轮数
agent_C: int = 1                # 独立分支数
agent_L: int = 30               # 单分支精炼步数
agent_K: int = 1                # 每步局部采样数
agent_pareto: bool = True       # 投影：Pareto / single_best / topk
agent_inspector: bool = True    # 审查者开关 (消融臂)
agent_inspector_model: str = "" # 建议与 Creator 不同的底座
agent_token_budget: int = 2_000_000
agent_cost_budget_usd: float = 5.0
agent_skills: list_str = []     # 加载的领域技能卡
```

调度循环（对应 SRHarness Algorithm 1）：

1. `S ← archive`；对 r ∈ 1..R：取 `TopK(S)` 作为初始上下文，新建对话；
2. 对 c ∈ 1..C：复制 prompt 骨架 → 独立分支（**分支赋不同假设族先验**，见 3.5）；
3. 对 ℓ ∈ 1..L：`Diagnose`（首步强制跑一次 profile + prior）→ 组 prompt（注入 `P_ρ(S)`）→ LLM 采样 K 个候选 → 逐个过 Inspector 与动作 → `Admit` 进 S → 选最佳继续；
4. 早停：提交公式通过全部约束且跨工况 worst-case 达标时返回；
5. 全程 `Record(r,c,ℓ)` + provenance 落盘。

预算：`time_budget_s`（已有）+ `agent_token_budget` + `agent_cost_budget_usd` 三元硬上限。**成本必须先实测单题再定 R/C/L**（见 §5）。

### 3.4 L4 角色层 —— `lgf/agent/roles.py`（新建）

**Creator（Agent A）**：现有 G_INIT/M_GUIDE 合并升级为**带工具调用的提案循环**。输出契约从"DSL 列表"变成"动作序列 + 候选"：

```json
{"thought": "...", "actions": [
   {"tool": "fit_constants", "dsl": "...", "budget_s": 8},
   {"tool": "guardrail_check", "dsl": "..."}],
 "candidates": [{"dsl": "...", "rationale": "..."}]}
```

**Inspector（Agent B）**：**不产生新公式**，只裁决。输入 = 候选 + 全部探针证据；输出：

```json
{"verdict": "pass | return_for_refinement | reject",
 "violations": [{"constraint": "C1_parity", "evidence": {"even_score": -0.31},
                 "min_fix": "把 √roloc 换成 roloc·log(roloc)"}],
 "unverified_assumptions": ["..."]}
```

**ASM 域约束清单（9 条，全部数值探针）**

| # | 约束 | 探针 | 对应现有资产 |
|---|---|---|---|
| C1 | 奇偶对称性 | 对 y→−y 镜像后 even/odd score | **今天 T1s 断言 bug 的机器化** |
| C2 | 张量形式可表性 | 在 T 基上投影残差 + 基共线性 cond | `_tmp_t23_basis_diag*.py`（cond 136–234 的发现） |
| C3 | 值域 | b23 ≥ 0、量级带 | `attribution._target_semantics` |
| C4 | 跨工况常数共享 | 是否存在 `K#`（已有硬拦截） | `_force_global_params` |
| C5 | 跨 Ro 泛化 | LOSO mean / worst-case | 已有 LOSO 脚本（标量 0.80、T 基爆） |
| C6 | 量纲/无量纲一致性 | `priors.dimensional_analysis` | ✅ 已有 |
| C7 | 数值鲁棒性 | yw→0 / 中心 / 外推抽样，NaN·inf·负 log | `paramfit._count_finite` 等 |
| C8 | 复杂度 | 节点/参数上限 | `cfg.max_edges/max_params` |
| C9 | 反作弊 | T3v/T4v 通道异常低 = 合成作弊 | 验收 5 硬规则的现成一条 |

**shipman 域**（复用同一 Inspector 骨架）：V1 六系数逐位恢复精度、误吸收（`calibrate_ext` 占比）、量纲、符号、Z 形/TD 稳定性。

**为什么"两个 agent 优于一个"在这里成立**：文献的价值不在"多一个 LLM"，而在**生成与审查的激励分离** —— 生成者的目标是拟合，审查者的目标是合规，两者用不同的 system prompt、可选的**不同底座模型**，避免自证偏差。我们的历史证据（MIX 轮"存在 LLM 位置偏好未排除"）说明这个风险是真实的。

### 3.5 假设族先验：C（分支）应该分支在什么上

分支不能是"同样的 prompt 采样两次"。ASM 域建议的 4 个族（可配置）：

| 族 | 先验骨架 | 依据 |
|---|---|---|
| F1 阻尼族 | `C·exp(C/yw)` / `C/(C+yw)` | B2 wall-seeds 已证有效 |
| F2 对数律族 | `√roloc·(C·log(yw)+C'·roloc)` | V4b 冠军公式所在族 |
| F3 张量基族 | `Σ a_m(roloc,yw)·T_m23`（g/e 基） | 10-16 算术：混合形式 0.0081/0.0076 |
| F4 机制族 | 涡黏类比 `ν_t·S₂₃` 结构 | 物理可解释性优先 |

**必须在 prompt 中声明这是分支假设、不是事实**，并由 Inspector 判定该族是否 pass —— 否则就是把错误断言换了个地方又塞进 prompt。

### 3.6 运行时 —— `lgf/agent/runtime.py`（新建）

- **tool-call 解析**：现成 `_strip_reasoning` + `safe_json_loads`（已适配 GLM-5.3 thinking、```json 围栏、尾随逗号）直接复用；新增"动作序列"schema 校验与一次修复重试。
- **observation 压缩**：全量 evidence 落 archive，prompt 里只放 ≤400 字符紧凑视图 —— 对应 SRHarness"runtime 保留详细证据但不塞进上下文"。
- **沙箱**：`code_exec` 限工作目录子路径、禁网、限时（参考 SRHarness 的 sandbox commit 思路）。**默认关闭**。
- **provenance & resume**：每次工具调用 → `traj.jsonl`；中断后从 archive + 最后一步恢复（对齐你 `runner_sm_resume.ps1` 的实际需求）。
- **成本会计**：tokens/cost 逐调用记账（现在完全没有）。
- **审计视图**：可选本地 WebUI（SRHarness 有，对汇报/论文附图有利），非必需。

---

## 4. 落地路线（P0 → P4，每步带验收）

### P0 — 底盘：状态层 + 约束层（**不调 LLM**）
产出：`lgf/store.py`、`lgf/verify/hard_rules.py`、`lgf/projector.py`、`lgf/tools.py`（前 8 个工具）

**验收（这是全案第一个"值不值得继续"的判决点）**：

1. 把现有 `outputs/asm/runs_*/*_final.json` 与 `outputs/shipman/**` 全量回灌进 archive；
2. 用 `Pareto` 投影 + archive 查询**直接复现** 2026-10-16 的两个结论：
   - "T23 净贡献 ~6% RMS"（取代 `_tmp_t23_contrib.py`）；
   - "T23-full 8 条 LGF 冠军全为纯标量函数，零 T 通道"；
3. 对 C1 奇偶性探针，**机器判定** `T1_23/T5_23/T6_23/T10_23` 为偶对称（应复现 even ≈ +1.00），并自动标记 physics context 中那条错误断言。

> 若 P0 第 3 条不能自动跑出来，说明约束层设计有问题，**不要进入 P1**。

### P1 — 轨迹层：`agent_mode=True` 跑通 R-C-L-K
产出：`lgf/agent/scheduler.py` + config 字段
**验收**：同一任务上 legacy vs agent（`agent_inspector=False`）同 `time_budget_s` 对照；agent 必须能**在预算内**跑完 R-C-L-K 并落 provenance；复现"L 加深 > R/C 增加"的定性规律（若复现不出，则说明迁移假设不成立，如实报告）。

### P2 — 角色层：接入 Inspector
**验收**：三臂对照（legacy / agent-no-inspector / agent+inspector）；报告"入池公式通过全部 9 条约束的比率"（对标文献 91→100% 的口径）。**精度不得显著下降**（文献的硬要求）。

### P3 — 域适配与技能卡
`asm`/`shipman` 的 `read_skill` 卡；把 `physics_context` 拆成 **可验证断言** 与 **待验证假设** 两类字段（前者交由 Inspector 探针，后者标注为假设不得进 prompt 作为约束）。

### P4 — 评估与论文
可选消融臂（对齐 SRHarness 设计，便于论文对标）：
- 投影：Pareto / single-best / topk
- 轨迹：R1C1L30K1 / R2C1L15K1 / R1C2L15K1 / R1C1L15K2
- 动作空间：全套 / 去掉表达式视图 / 只留 code_exec
- 角色：±Inspector
指标：`loss_vw`、T2v NMSE、SA（`eq_compat` 对金标准）、跨 Ro worst-case、约束通过率、tokens/cost、多种子 σ。

---

## 5. 预算与风险

### 5.1 成本（先实测，别先承诺）

SRHarness 在 LLM-SRBench 单题约 **14.9 min / 1.354 M tokens / $0.0353**（DeepSeek-v4-flash）。我们的 GLM-5.3（`thinking=enabled, reasoning_effort=low, max_tokens=16384`）单价与推理开销不同，**P1 的第一步就是测单题实测成本**，再据此定 R/C/L/K。参考起点：R1-C1-L30-K1。

### 5.2 风险表

| # | 风险 | 严重度 | 缓解 |
|---|---|---|---|
| R1 | LLM 自证偏差 / 位置偏好（MIX 轮已见苗头） | 高 | Inspector 独立实例（可换底座）+ 数值探针 + 候选呈现顺序随机化 |
| R2 | 指标套利：agent 优化 loss 而非物理 | 高 | 9 条硬规则 + LOSO + C9 反作弊，全部机器判 |
| R3 | 上下文膨胀 / 成本失控 | 中 | observation 压缩 + 有界投影 + 三元预算硬上限 |
| R4 | 复现性下降（轨迹随机性） | 中 | 全量 provenance + 固定默认 R1C1L30K1 + 多种子 σ 报告 |
| R5 | 与 PySR 的效率差距被误读成性能劣势 | 中 | 评估口径明确写"可审计 + 物理合规 + 跨工况"，不比单点 loss |
| R6 | 过度工程：agent 壳很大，收益未证 | **高** | P0 验收做判决点；`code_exec` 默认关闭；legacy 路径永久保留 |
| R7 | 文献数字来自 LLM-SRBench / CANN 本构，**迁移是假设不是结论** | 高 | 每条机制都在我们数据上单独消融，不能引用文献数字代替本域证据 |

---

## 6. 与你当前研究的关系

1. **直接服务今天的 T23 口径争议**：C2（张量形式可表性）+ C5（LOSO）+ 完整 archive，能把"能否写成 G·T"从**事后手算**变成**机器裁决**；你 10-16 已发现的关键约束（单分量上"能否写成 G·T"无约束力，真正约束力来自"同一组系数同时闭合 6 分量"）天然就是 C2 探针的定义。
2. **服务 shipman**：V1 系数恢复、误吸收检验、量纲/符号/稳定性，全部是 Inspector 约束的现成实例；`shipman/seeds.py` 的 monkeypatch 手法已在证明"领域层可以零改内核接入"。
3. **论文素材**：ASM 与 shipman 两篇都可加一节 "agentic LGF"，并复现 SRHarness 的三机制消融（动作/状态/轨迹）—— 这是审稿人会追问的"harness 本身是否是一等贡献"。

---

## 7. 决策记录

| # | 事项 | 决定 | 状态 |
|---|---|---|---|
| 1 | 引擎归属 | 新建并列引擎（`agent_mode` 开关，`pipeline_lgf.run_lgf` 一行不改） | 待最终确认 |
| 2 | Inspector 底座 | 先同底座（GLM-5.3）跑通，P4 再加"换底座"对照臂 | 待最终确认 |
| 3 | 第一落地域 | **ASM b23** | ✅ 已定（2026-10-16） |

> 决策 3 的含义：P0 的 archive 回灌与 C1 奇偶性探针**只针对 ASM b23 做全量验收**；
> shipman V1 留到 P3 域适配阶段再接（它的 Inspector 约束清单已在 §3.4 备好）。

---

## 附录 A — 三个待确认事项的详细说明

### A.1 "并列引擎 vs 重写"是什么意思

| | 并列引擎（推荐） | 重写 |
|---|---|---|
| 做法 | 新增 `lgf/agent/`、`lgf/tools.py`、`lgf/store.py`、`lgf/verify/`；旧 `pipeline_lgf.py` 完全不动；`cfg.agent_mode` 选引擎 | 把 `pipeline_lgf.py` 直接改成 agent 驱动，GP 降格为一个工具动作 |
| 已有产物 | `runs_*/`、`t23full_paper/`、`outputs/shipman/**` 全部照常可读、可复算 | 需要逐个回归验证 |
| 跑批脚本 | `runner_sm.ps1`、`_run_v4b_matrix.py` 等无需改动 | 全部要跟着改 |
| 论文 AB 对照 | **同一次提交内** legacy / agent 同预算对照 | 只能靠 git 回退，做不到同一版本内对照 |
| 缺点 | 两套代码有少量重复逻辑（如候选解析），需偶尔同步 | 代码干净 |

**关键理由不是风格，是实验设计**：你要的结论形式是"agent 比 legacy 好多少"。这个结论要求两条路**同时活着、同预算可比**。重写后旧路径只存在于 git 历史里，没法做同版本对照，也就失去了这个结论。所以建议并列。

### A.2 "Inspector 底座换模型"是什么意思

"底座"= 跑这个 agent 的**大语言模型本身**（backbone）。我们当前 Creator 用的是 GLM-5.3（见 `scripts/rc_asm_run.py` 的 `BACKENDS["glm53"]`）。

- **同底座**：Creator 和 Inspector 都是 GLM-5.3，只是 system prompt 不同。
- **换底座**：Inspector 换成另一个模型（DeepSeek / Kimi K2.5 / MiniMax M2.7 任选）。

**为什么要考虑换**：Inspector 的职责是挑 Creator 的错。同一个模型容易有**系统性同盲区** —— 它认为对的结构，当审查者时也倾向认为对，这就是自证偏差。文献里"生成与审查分离"的收益在不同底座上差异很大（Opus 4.7: 91%→100%；Kimi K2.5: 37%→56%），说明审查有效性依赖两个模型**不共享同一个盲区**。

**为什么不急着换**：每次审查多一次 LLM 调用（成本翻倍），还要多配一个 key/配额；更重要的是一开始就把两个变量（有没有 Inspector / 换不换底座）混在一起，实验就说不清了。

**建议**：P2 先同底座把管线跑通；P4 再做三臂对照（无 Inspector / 同底座 Inspector / 异底座 Inspector），那时"换底座有没有用"才是一个干净的可测问题。

### A.3 优化器选型：现在该不该上量子退火

见附录 B。

---

## 附录 B — 为什么现在不该上量子退火

### B.1 先分清：你这里有**两个**优化问题，量子退火只对其中一个有理论意义

| 层级 | 优化什么 | 变量规模 | 现状实现 | 问题性质 |
|---|---|---|---|---|
| **连续层** | 给定结构，拟合参数 θ | P = n_g + 5·n_c ≤ **~30** 维；拟合点 ≈ 200×5 = 1000 | `paramfit.fit_joint`：L-BFGS-B + **解析梯度**（`jac=True`，`compiler` 反向传播）+ 多启动（base / 均匀扰动 / 对数尺度 ±10^U(-3,3)）+ callback 硬 deadline + `ftol=1e-14` 抛光步 | 低维、非凸、**梯度廉价且解析** |
| **离散层** | 选结构 / 选通道 | 图空间组合爆炸 | `gp.py`：crossover / mutate / select / extinction | 组合搜索 |

量子退火（QA）解的是 **QUBO**（二值、二次、无约束）。它只对**离散层**有理论意义。

### B.2 连续层：数学上不成立，不是"性能差"

1. QUBO 只有**二次**项；而 `loss = Σ(y − f(θ))²` 中 f 含 `exp / log / tanh / div`，展开是超越函数或高次有理式。要编码必须分段线性/多项式逼近 + 引入大量辅助变量。
2. 6 个连续参数要量化成二值，精度与比特数呈指数关系；即便编码成功，得到的也只是"比 L-BFGS-B 更差的局部解" —— 因为 loss 光滑、梯度解析可得，局部搜索这一层已经没有改进空间。

**结论：连续参数层不适用。** 把 L-BFGS-B 换成量子退火，等于给一台已经会爬山的机器换成只会跳格子的。

### B.3 离散层：这才是 QA 的适用域，但我们规模太小、且关键耦合是高阶的

**自然 QUBO 形式**（对我们很适配）：

```
变量 x_i = 是否选第 i 个候选通道/项
目标 Σ_i c_i x_i + Σ_ij c_ij x_i x_j
稀疏度约束 → penalty「恰好选 k 个」
```

其中 `c_i` 可由"线性化一次拟合后的残差贡献"给出 —— 在方案 B 里 `a_m` 本身就是线性系数，天然适配。

**已有工作**（说明这条路有人走通）：

- Wenxin Li et al. (2025)：用相干伊辛机（CIM）把三类稀疏优化（ℓ0 正则 / k-稀疏 / 固定稀疏度）统一映成 QUBO，配 Benders 分解做量子-经典耦合，几十到上百比特毫秒级求解。
- D-Wave 系 QaML（npj Comput. Mater. 2026）：特征选择、SVM 训练、网络剪枝写成 QUBO，在 QUBO 剪枝网络上观察到更好的泛化。

**但对我们的三点硬限制**：

1. **规模太小，经典可以精确**：候选字典只有 10–40 项（T 基 10 + 不变量 ~12 + 模板），2^20 ≈ 10⁶ 可以**穷举求最优**。CIM / D-Wave 有意义的规模是 10²–10⁴ 比特。在小规模上，用硬件是降级。
2. **QUBO 只能表达成对耦合，而我们的核心约束是高阶的**：2026-10-16 的诊断结论明确 —— "只拟合 b23 时张量基形式不识别，每点 2 个自由系数即可精确复现 b23；真正约束力来自**同一组系数同时闭合 6 分量**"。这是**高阶/全局耦合**，正是 QUBO 化最难、degree reduction 会让规模爆炸的部分。
3. **当前公开证据不支持在组合优化上期待优势**：arXiv:2509.03952（2026-02，平行时间 QUBO 基准，D-Wave 两代 vs 经典 SA / GPU 求解器 VeloxQ）结论是 —— Advantage2 稳定优于上一代，但**经典 VeloxQ 仍有最短绝对运行时间**（"reflecting the maturity of classical heuristics"）。D-Wave 的 beyond-classical 结果（Science）集中在**自旋玻璃 / 量子动力学模拟**，不是组合优化。

**结论：目前不存在"符号回归上用 QA 打赢经典"的稳健公开证据。小规模实例上应当明确预期经典更好。**

### B.4 我们的真实瓶颈不在求解器（有日志证据）

| 证据 | 症状 | 换求解器能修吗 |
|---|---|---|
| `paramfit.fit_joint` V4 注释 + `lgf_mg/m` elapsed 2834s ≫ budget 600s | 单次 L-BFGS 无上限 → 预算被拟合吃光，GP 实际代数远低于 `generations` | ❌ 预算调度问题 |
| V4b：`Graph.clone` 不重置 `loss/theta` | GP 进化完全失效（每代 0.7s，top_final 坍缩到 1–2 个结构家族） | ❌ 状态管理 bug |
| 2026-10-16 T23 算术：v 基 cond 136–234 | 基共线 → 任意系数都拟合不动（残差 ~1.06） | ❌ **问题构造**（换基），不是求解器 |

**四行里没有一行是"求解器不够强"。** 在这种情况下引入量子硬件，优先级明显低于把这三处修好（前两处已修）。

### B.5 真要在优化上加力，建议的顺序

1. **变量投影 / 可分离最小二乘（收益最大，且不需要新依赖）**：把 θ 分成"线性进入"与"非线性进入"两半，线性部分用 `lstsq` **精确求解**，非线性部分维度从 ~30 压到 1–3 维后再多启动。证据：`shipman` 的 V1 六系数就是这么做到**逐位精确恢复、0% 误差**的；而冠军公式族 `√roloc·(C·log(yw) + C'·roloc)` 本身就是线性进参数 —— 现在却被丢进 30 维非凸 L-BFGS 里盲搜。
2. 经典全局优化候选：CMA-ES / basin-hopping / 代理模型（BO），在 ≤30 维非凸上都很成熟。
3. 稀疏 / 子集选择：MIQP / 分支定界 或 SA，先拿到**经典精确解**作为 oracle 上限。
4. 只有当第 3 步的结论是"经典精确解也吃不下、规模确实卡在 30–100 变量"时，才把结果拿到 D-Wave Leap / CIM 做对照。
5. 这个（很可能是负面的）结果本身可以写进论文 —— "为什么我们不用量子退火"。审稿人会喜欢这种有界限的结论。

### B.6 想先探一下水的最小实验（约 1 天，不需要量子硬件）

1. 取 10-16 的 g/e 基 10 项候选池，构造固定稀疏度 k ∈ {3, 5, 7} 的 QUBO；
2. 用 D-Wave 官方开源包 `dimod` / `neal`（经典 SA + QUBO 工具）求最优，与"多启动 L-BFGS-B + 穷举"对照；
3. 记录三项：达到同等 loss 的时间、解质量、可处理的规模上限；
4. **判据**：若 `neal` 与穷举结果一致 → 说明这个问题不需要量子；只有当 `neal` 在 ≥30 变量上也超时，才值得谈硬件。此时再上 D-Wave，结论才是干净的。

---

*附录：不改动的部分 —— `pipeline_lgf.run_lgf`、`gp.py`、`paramfit.py`、`compiler.py`、`dsl.py`、`graph.py`、`priors.py`、`attribution.py` 全部保持行为不变；LGF-A 是并列引擎 + 新增模块，不是重写。*
