# LGF：融合 LLM-Feynman 与 Graph-based 方程发现的技术方案

> **LGF = LLM-guided Graph-Feynman**
> 当前实现路径：`D:\AI\LLM-Feynman-G`
> 融合对象（历史脉络与算法来源，均**不构成运行期依赖**）：
> ① LLM-Feynman 正式版（`D:\AI\LLM-Feynman\LLM-Feynman-main`，LLM-SR 骨架 + AI-Feynman 简化模块 + 本地 LLM 引擎）
> ② Xu, Chen & Zhang (2026) *Sci. Adv.* 12, eaec0989 提出的 **GraphED**（graph-based equation discovery）
> ③ AI-Feynman 的物理先验体系（量纲分析 / 对称性 / 可分离性 / 组合性）
>
> **文档状态**：本文件已按 2026-09-23 的实际代码快照校准（见第 0 节与第 4 节）。

---

## 0. 摘要与实现状态说明

### 0.1 融合关系的确切口径

**结论：LGF 是一个完全自包含的独立包，与 `D:\AI\LLM-Feynman` 目录无任何运行期依赖。**

已完成的核验（2026-09-23）：

| 核验项 | 命令 | 结果 |
|---|---|---|
| Python 包级 import 依赖 | `Select-String -Path lgf\*.py,main.py,scripts\*.py,tests\*.py -Pattern "^\s*(import\|from)\s+llm_feynman"` | **0 处真实 import**（唯一命中是 `eq_compat.py` 文档字符串中"本模块不 import llm_feynman"的自述） |
| 运行时模块搜索路径注入 | `Select-String -Path ... -Pattern "sys\.path"` | **0 处**指向 `D:\AI\LLM-Feynman`（所有 `sys.path.insert` 均指向本仓库自身 ROOT） |
| 对 `D:\AI\LLM-Feynman` 的引用 | 全仓库 grep `LLM-Feynman` | 19 处，**全部为注释、文档字符串，或带优雅降级的可选数据路径** |
| 上游代码拷贝 | — | 无 `.py` 文件从 `llm_feynman/` 直接拷贝 |

因此，LGF 与 `D:\AI\LLM-Feynman` 的正确关系是：**算法机理层面的继承（设计思想、方言契约、数据格式），而非代码或运行期层面的复用。**

### 0.2 与 LLM-Feynman 正式版的关系（三条继承路径）

| 继承路径 | 来源（`LLM-Feynman-main\llm_feynman\`） | 在 LGF 中的落地 | 性质 |
|---|---|---|---|
| **语义算子层** | `llm_engine.py`（`EQUATION_TEMPLATES`、`SimulatedLLMEngine._generate_context_aware_skeletons`、`RealLLMEngine._build_prompt`） | `lgf/llm_ops.py`（`HeuristicLLM`）与 `lgf/eq_compat.py`（纯重实现的方言适配层） | 思想 + 契约移植，**零 import** |
| **物理先验层** | `feynman_modules.py`（多项式拟合 / 可分离性 / 对称性 / NN 代理） | `lgf/priors.py`（`test_symmetry` / `test_additive_separable` / `test_multiplicative_separable` / `test_compositionality` / `dimensional_analysis`） | 思想移植并**升级为图约束**（`PriorConstraints`） |
| **服务接口层** | `llm_engine.py` 的 Flask `/completions` 协议 + 独立 `D:\AI\LLM-Feynman\llm_engine\engine.py`（Mixtral 多 GPU / 4bit 量化服务） | `lgf/llm_ops.py::HttpFlaskLLM`（协议兼容客户端） | **协议兼容**，服务非必须 |
| **数据格式** | `..\data\`（oscillator1/2、bactgrow、stressstrain 等 csv） | `lgf/benchmarks.py::load_real_csv_simple`，默认 `LGF_REAL_DATA_DIR = D:\AI\LLM-Feynman\LLM-Feynman-main\data` | **可选**，缺失即返回 `None`，主流程不依赖 |

> 关于上游定位的修正：`LLM-Feynman-main` 并非纯 LLM-SR 骨架，其 `README.md` 与 `pipeline.py` 显示它**本身就是 LLM-SR + AI-Feynman 的融合体**（声称 Feynman Lectures 10 式上 Avg Test R² = 0.912，对比 LLM-SR 0.754）。其 `run_experiment.py` 同时跑 6 个方法（LLM-Feynman / LLM-SR / linear / polynomial / NN / symbolic regression）做消融。
>
> 这对 LGF 的意义：从上游可直接继承的**语义层与先验层已有先例**，LGF 真正的新增在于引入 **GraphED 的图表示底座**，把上游"自由 Python 代码 + 启发式简化"升级为"边特征 DAG + 图约束 + 归因驱动变异"。

### 0.3 当前实现完成度（对照代码快照）

| 模块 | 文件 | 状态 |
|---|---|---|
| 图表示 | `lgf/graph.py` | ✅ Node / Param / Graph，边特征含 Constant / GlobalC / CaseC_i |
| DSL 双向转换 | `lgf/dsl.py` | ✅ 图 ↔ 文本往返 |
| 编译器 | `lgf/compiler.py` | ✅ `CompiledGraph`，解析梯度 + sympy 数值求值 |
| 参数拟合 | `lgf/paramfit.py` | ✅ `fit_joint`（L-BFGS-B，单案例 / 多案例联合） |
| 物理先验 | `lgf/priors.py` | ✅ 五类检验 + `tests_to_constraints` → GP 约束 |
| 模板库 | `lgf/templates.py` | ✅ `TEMPLATE_REGISTRY`（10 个物理 / 本构模板） |
| 图遗传算子 | `lgf/gp.py` | ✅ crossover / mutation / extinction / selection |
| 误差归因 | `lgf/attribution.py` | ✅ `analyze`（节点敏感度 → 定位坏子图） |
| LLM 算子 | `lgf/llm_ops.py` | ✅ 三种后端（heuristic / flask / openai）+ 五类指令 |
| 方言适配 | `lgf/eq_compat.py` | ✅ LLM-Feynman 方言 → LGF DSL（纯重实现） |
| 分层双循环 | `lgf/pipeline_lgf.py` | ✅ `run_lgf` 主入口 |
| 配置 | `lgf/config.py` | ✅ `LGFConfig` + XML 序列化（`from_xml` / `to_xml`，37 项 `test_config_xml.py`） |
| 配置入口 | `config.xml`（仓库根目录） | ✅ 三层配置链：默认值 → config.xml → CLI |
| 基准 | `lgf/benchmarks.py` | ✅ 11 个合成基准 + 可选真实 csv 加载 |
| 统一 CLI | `main.py` | ✅ `--framework {lgf,llmsr}`（`llmsr` 已弃用并给出迁移提示），`--config` 接入 config.xml |
| 测试 | `tests/` | ✅ `smoke_graph_dsl.py` + `test_lgf.py`（15 项）+ `test_eq_compat.py`（14 项）+ `test_config_xml.py`（37 项） |
| 实验脚本 | `scripts/run_benchmarks.py` | ✅ 9 题 × 3 配置消融（quick / full） |
| mock 服务 | `scripts/mock_llm_server.py` | ✅ 自带 Flask `/completions` 模拟器 |

---

## 1. 三种方法的机理剖析与互补性

### 1.1 LLM-Feynman 正式版（LLM-SR + AI-Feynman 融合）

上游 `llm_feynman/` 的模块分工：

| 模块 | 实现 | 作用 |
|---|---|---|
| `llm_engine.py` | `SimulatedLLMEngine`（40 条 `EQUATION_TEMPLATES` + 15 类物理上下文关键词分支）/ `RealLLMEngine`（OpenAI 兼容，失败回退模拟器） | 生成 `EQ: <python-expr>` 形式的方程骨架，未知常数写 `params[i]` |
| `feynman_modules.py` | `try_polynomial_fit` / 可分离性 / 对称性 / MLP 代理 | AI-Feynman 式简化，产出"结构线索"注入 LLM prompt |
| `evolutionary_search.py` | 多岛（multi-island）进化，`NUM_ISLANDS` / `BATCH_SIZE` / `MAX_ITERATIONS` | 维护骨架种群，按 fitness 淘汰 |
| `optimizer.py` | L-BFGS-B | 拟合骨架内自由参数 |
| `pipeline.py` | `run_llm_feynman` | 编排：数据 → 简化分析 → 上下构建 → LLM 进化搜索 → 报告 |
| `llm_sr_baseline.py` | 纯 LLM-SR | 消融对照 |
| `data_utils.py` | 数据生成 + `compute_metrics` | `generate_data` / `train_test_split` |
| `..\llm_engine\engine.py` | Flask `/completions`，Mixtral-8x7B 多 GPU + 4bit 量化 | 真实本地 LLM 服务端（独立于 `llm_feynman/` 包） |

**优势**：LLM 携带海量领域先验（知道 Hollomon、Voce、J-C 等模型形式）；物理简化把结构线索喂给 LLM，比纯 LLM-SR 收敛更稳。
**短板**：
1. 搜索空间是**自由 Python 代码**，无结构约束 → 生成大量语法/数值无效解（`SimulatedLLMEngine` 已用 `attempts < 100` 兜底，说明该问题是真实存在的）；LLM 输出解析仅靠 `line.startswith("EQ:")`，鲁棒性有限；
2. 种群进化在**代码字符串层面**（`_mutate_skeleton` 用 3 条正则替换），无法做子结构级交叉（两段代码无法"换器官"）；
3. 每类物理上下文对应**硬编码模板**，换领域需人改代码；
4. LLM 调用代价大，内层高频搜索用不起。

### 1.2 GraphED（sciadv.aec0989, 2026）

| 机制 | 内容 |
|---|---|
| 表示 | 方程 = 根 DAG `G=(V,E)`；节点=算子/变量；算子元数=出度（可结合算子出度 ≥2，原生多元表示）；**边特征 `f_e`** = 常数或**逐案例自由参数 C_i**（如 `x̃_{u←v} = f_e · x_v`） |
| 模板 | 多项式模板 P（仅 add/mul/pow），log/exp/pow/trig 只允许受限参数形态（如 `cos(C₁(x+C₂))`），防 `log(cos(exp(·)))` 类病态嵌套 |
| 进化 | 图交叉（子图交换）+ 三类变异（边特征扰动/节点替换/子图替换）+ top-50% 选择 + **灭绝机制**防早熟 |
| 参数 | 每个候选图、每个数据集独立用 **L-BFGS 多启动（20 次随机初始化）** 拟合边参数；多案例取平均损失 |
| 验证 | DIF 合金钢模型（超 C-S/J-C/H-K/R-J 及 PySR/AI-Feynman/DSO）、锂变形、填充橡胶超弹性 |

**优势**：结构化搜索空间可控（边数/参数数上限）；**自由参数与拓扑耦合在同一表示里**（树方法做不到）；多元图比二元树更紧凑。
**短板**：
1. 初始种群与所有变异均为**随机**——遗传算子不懂物理语义，好结构（如 `e^{1/(C₂ε+C₃)}`）要靠盲搜撞出来；
2. 模板库是**人工静态**设定的，换领域要人重写；
3. 对称性/可分离性等数据中已显式存在的结构信息**完全未利用**，搜索空间未降维；
4. top-k 等价方程（论文 Fig.2B 中 top-3 数学等价）需要人来判断选哪个。

### 1.3 AI-Feynman 的物理先验体系

Udrescu & Tegmark 的分而治之核心：先用神经网络拟合代理函数，再对代理做先验检验，**命中即把问题劈成子问题**：

| 先验 | 检验 | 归约效果 |
|---|---|---|
| 量纲分析 | 变量单位 → Buckingham-Π | 变量数下降（n→n-k） |
| 多项式性 | 代理函数幂次检验 | 直接锁定 pow 结构 |
| 对称性 | f(x₁,x₂)≈f(x₂,x₁) | 变量绑定，图搜索空间平方级缩小 |
| 加性可分 | f(x₁,x₂)≈g(x₁)+h(x₂) | 根图强制为 add(g,h)，两个独立小子空间 |
| 乘性可分 | f≈g·h | 同上，根强制 mul |
| (广义)组合性 | f(x₁,x₂)≈g(h(x₁),x₂) 等 | 变量替换（如 log 变换把 mul 化 add） |

**优势**：指数级降维，是"结构信息 → 搜索空间剪枝"的最强手段。
**短板**：检验流程刚性（面向光滑闭式解）；不做连续参数的精细化辨识；对多案例、含材料参数的本构问题无原生支持。

### 1.4 互补性矩阵

| 能力维度 | LLM-Feynman | GraphED | AI-Feynman | LGF 融合后 |
|---|---|---|---|---|
| 领域语义先验 | ✅ 强 | ❌ 无 | ⚠️ 部分（人为注入） | LLM 注入 |
| 结构化搜索空间 | ❌ 自由代码 | ✅ DAG+模板 | ⚠️ 树/模块 | ✅ 统一图表示 |
| 子结构级交叉/变异 | ❌ 正则变异 | ✅ | ⚠️ | ✅ |
| 自由参数（材料参数） | ⚠️ L-BFGS 内拟合 | ✅ 边特征原生支持 | ❌ | ✅ |
| 多案例 (multicase) | ❌ | ✅ | ❌ | ✅ |
| 搜索空间降维 | ⚠️ 简化线索 | ❌ | ✅ 强 | ✅ Feynman 检验→图约束 |
| 高频内层搜索成本 | ❌ LLM 贵 | ✅ GP 便宜 | ✅ | ✅ GP 做内层 |
| 等价性判定与结构选择 | ⚠️ | ❌ 靠人工 | ❌ | ✅ LLM 初筛 + sympy 终审 |

**结论**：三者几乎两两互补，且 GraphED 的图表示恰好是能把另外两者"装进来"的公共底座——LLM 的输出可以编译成图，Feynman 检验的结论可以表达为图上的结构约束。

---

## 2. LGF 总体架构

### 2.1 设计原则

1. **表示统一**：一切候选方程都是"带边特征的根 DAG"；LLM 不再生成自由代码，而是生成**图的 DSL**（可解析、可校验、可交叉变异）。
2. **分层搜索**：外层 LLM 慢循环（语义级、全局重组，每几十代一次）+ 内层 GP 快循环（组合级、局部微调，每代一次）。
3. **先验三源合流**：领域先验（LLM）+ 物理先验（Feynman 检验）+ 结构先验（GraphED 模板）全部编译成**同一搜索空间上的约束**。
4. **归因驱动变异**：误差归因定位"坏子图"，LLM 变异优先作用于此——从随机变异升级为定向变异。
5. **参数-拓扑耦合优化**：拓扑由图进化负责，参数由 L-BFGS-B 负责（继承 GraphED），支持全局参数与逐案例参数。

### 2.2 架构图

```
┌────────────────────────────────────────────────────────────────────┐
│                        LGF 分层双循环架构                            │
│                                                                    │
│  数据 (合成基准 / multicase csv)    领域知识/文献 (可选, LLM 读入)      │
│        │                              │                            │
│        ▼                              ▼                            │
│  ┌─────────────┐   ┌──────────────────────────────┐                │
│  │ 预处理模块    │   │  外层循环: LLM 语义进化 (慢)     │               │
│  │ · 量纲分析    │──▶│ ① 模板提议 (graph templates)   │               │
│  │ · 变量归约    │   │ ② 初始骨架生成 (领域先验)        │               │
│  └─────────────┘   │ ③ 分解假设 (可分性/组合性→图约束) │               │
│                    │ ④ 定向变异 (基于误差归因)         │               │
│                    │ ⑤ 等价判定与结构选择              │               │
│                    └──────────────┬───────────────┘                │
│                                   │ DSL (图↔文本双向转换)            │
│                                   ▼                                │
│  ┌────────────────────────────────────────────────┐                │
│  │  内层循环: GP 图进化 (快)                         │               │
│  │  种群 = {G₁...G_N} (带边特征 DAG)                 │               │
│  │  · 图交叉 / 随机变异 / 灭绝机制                     │               │
│  │  · L-BFGS-B 联合拟合边参数 (逐案例)                 │               │
│  │  · Feynman 结构约束过滤 (对称性等)                  │               │
│  └──────────────┬─────────────────────────────────┘                │
│                 │ top-K 图 + loss + 归因报告                        │
│                 ▼                                                 │
│  ┌──────────────────────────────┐                                  │
│  │  误差归因模块 (每 K 代)         │                                 │
│  │  · 节点值 z_v(x) 提取           │                                 │
│  │  · 残差 vs z_v 相关度            │                                │
│  │  → 定位"最差子图"              │                                  │
│  └──────────────────────────────┘                                  │
└────────────────────────────────────────────────────────────────────┘
```

### 2.3 一个进化周期内的信息流

1. 内层 GP 跑 K 代（如 10 代），产出 top-K 图；
2. 归因模块对每张 top 图计算节点级误差贡献，输出归因报告；
3. 触发外层：LLM 收到「top-K 图的 DSL + loss 曲线 + 归因报告 + 数据统计摘要」，输出结构化指令（见 3.2 的指令集）；
4. 指令解析器把 LLM 输出翻译成图操作（注入新图 / 定向变异 / 模板更新 / 分解约束），写回种群；
5. 循环，直到 loss 收敛或预算耗尽。

---

## 3. 关键技术细节

### 3.1 统一表示层：equation-as-graph ↔ DSL

**图数据结构**（已实现于 `lgf/graph.py`）：

```python
class Node:
    op: str          # 'add','mul','pow','exp','log','cos','sin','inv','neg',... 或变量名 'x1'
    children: list   # 出边指向的子表达式节点
class Param:
    # 边特征三类: Constant / GlobalC (全局参数) / CaseC_i (逐案例参数)
class Graph:
    root: Node
    n_edges: int     # 复杂度控制 (cfg.max_edges)
    n_params: int    # 自由参数上限控制 (cfg.max_params)
```

构造函数族：`var / const / add / mul / sub / div / scale / power / expn / logn / sinn / cosn / tanhn / sqrtn / invn / negn`，参数构造：`global_param / case_param`。

**DSL（LLM 的读写语言）**，纯文本、与 LLM-Feynman 的 spec 风格一致：

```
G1: mul( pow(P1(x1), C1), exp(add(x2, C2)) )
    P1(x) = C3*x + C4*x^2        # 多项式模板内联
    C1, C3, C4 = free            # 全局自由参数 (GlobalC)
    C2 = case                    # 逐案例参数 (CaseC)
```

- `graph→DSL` 用于喂 LLM（带 loss 注释：`# train_mse=0.00050`）；
- `DSL→graph` 用受限解析器，**解析失败直接丢弃**，保证 LLM 无法注入非法结构——这解决了 LLM-SR 的"无效解浪费"问题；
- `graph→Python/torch` 由 `lgf/compiler.py::CompiledGraph` 自动生成（替代 LLM 写代码，数值评估自包含）。

**方言兼容层**（`lgf/eq_compat.py`，纯重实现）：把上游 `llm_engine.py` 的方言规范化到 LGF DSL。

```
params[i]        -> C{i+1}          (全局自由参数)
np.sqrt(a)       -> sqrt(a)         np.exp(a) -> exp(a)
np.log(a)        -> log(a)          np.sin/cos(a) -> sin/cos(a)
np.pi            -> 3.141592653589793 (数值常量)
a ** 2 / 3 / 4   -> a^2 / a^3 / a^4
a ** 0.5         -> sqrt(a)
a ** params[k]   -> exp(C{k+1} * log(a))   (借助 log/exp 实现非整数幂)
a ** (-1)        -> inv(a)
```

### 3.2 LLM 的三重角色（外层循环指令集）

LLM 通过 `lgf/llm_ops.py` 的客户端调用（`heuristic` / `flask` / `openai` 三种后端），输出限定为以下五类结构化指令：

| 指令 | 触发时机 | 输入 | 输出 | 作用 |
|---|---|---|---|---|
| **T-PROP** 模板提议 | 初始 / 数据分布突变 | 变量名、量纲、数据统计（单调性/饱和性/渐近行为检测摘要） | 新模板（如 `sat(P) = C1·P/(C2+P)`） | 领域先验→结构先验 |
| **G-INIT** 骨架生成 | 初始 | 模板库 + 领域描述 | M 张初始图 DSL | 替代随机初始种群 |
| **H-DECOMP** 分解假设 | 归因显示可分结构 | 归因报告 + top 图 | 结构约束（如"强制根=add(g(x1),h(x2))"） | Feynman 式分而治之 |
| **M-GUIDE** 定向变异 | 每 K 代 | top-K 图 + 归因（哪个子图残差大） | 变异指令（`replace subgraph at node#7 with log(C1·P2)`） | 语义遗传算子 |
| **E-CHECK** 等价判定 | 收敛期 | top-K 图（数学等价候选） | 等价类划分 + 物理合理性排序 | 结构选择自动化 |

**降级策略**（已实现）：所有 op 在 LLM 不可用或 JSON 解析失败时**回退到启发式**（`HeuristicLLM`），保证离线可跑。

**关键工程点**：E-CHECK 中 LLM 只做初筛，等价性最终由符号验证（sympy 化简比对）确认——遵守评审原则，防止 LLM 幻觉误判等价。

### 3.3 Feynman 先验驱动的图分解

```
Feynman 分析 (lgf/priors.py)                 图搜索空间的归约
─────────────────────────                    ──────────────────
量纲分析 (dimensional_analysis)   ──────▶    变量替换/删除 → 图的输入节点缩减
代理场拟合 (SurrogateField, torch MLP) + 检验:
  · 对称性 test_symmetry          ──────▶    生成时强制两子图同构 (mutation 禁止破坏)
  · 加性可分 test_additive_separable  ───▶   根节点锁 add → 拆成两个独立子种群池
  · 乘性可分 test_multiplicative_separable ─▶ 根锁 mul → 同上
  · 组合性 test_compositionality  ──────▶    变量变换 (log 化乘为加等) → 等价图重写
                                              ↓
                                    tests_to_constraints() → PriorConstraints
                                    (编译为 GP 的过滤约束)
```

分解命中的判据（如可分性残差 < 数据噪声水平的 3σ）同时**通知 LLM**（作为 H-DECOMP 输入的置信度），由 LLM 结合物理语义决定是否采纳——物理先验提供候选，语义先验做裁决。

### 3.4 分层双循环与预算分配

| 层 | 执行体 | 频率 | 单次成本 | 职责 |
|---|---|---|---|---|
| 外层 | LLM（heuristic / 本地 Flask / OpenAI 兼容） | 每 K 代 1 次 | 秒-分钟级 | 全局语义重组、模板演化、分解裁决 |
| 内层 | GP + L-BFGS-B | 每代 1 次 | 毫秒-秒级 | 组合搜索、参数精修 |

预算控制：`LGFConfig` 暴露 `population` / `generations` / `time_budget_s` / `max_edges` / `max_params`；CLI 对应 `--population` / `--generations` / `--time_budget_s` / `--max_edges` / `--max_params`。相比纯 LLM-SR（每代都烧 LLM），LLM 调用次数下降 1-2 个数量级；相比纯 GraphED，随机盲搜代数被语义指令压缩。

**灭绝机制的 LLM 化改造**：GraphED 靠随机重启防早熟；LGF 在灭绝事件时触发一次"深度外层调用"，让 LLM 从归因报告推断当前模板库的系统性缺陷（例如"所有 top 图都缺饱和结构但数据呈饱和 → 提议 sat 模板"）。

### 3.5 误差归因：定向变异的依据

`lgf/attribution.py::analyze` 对 top 图提取每个算子节点 v 的中间量 z_v(x)，计算：

- `S_v = |corr(z_v, residual)|`（残差相关度，已实现 `pearsonr` / `spearmanr`）
- 偏依赖曲线形状（线性/饱和/振荡 → 提示替换算子类型）

归因报告示例（喂给 LLM）：

```
Graph G*: ... loss=0.0012
node#5 mul(x1, C1): sensitivity=0.87, partial-dep shows saturating shape
node#9 pow(P2, C3): sensitivity=0.12
→ 建议变异目标: node#5 (残差高度相关且形态不匹配)
```

这把 GP 的"随机选变异点"升级为"归因选点 + LLM 选替换结构"。

### 3.6 参数辨识

`lgf/paramfit.py::fit_joint` 已实现：L-BFGS-B 最小化，支持单案例与多案例（`case_params`）联合拟合；`evaluate` 提供给定参数下的损失评估。

- 全局参数（`global_param`）跨案例共享；逐案例参数（`case_param`）每案例独立；
- multicase 损失 = 各案例独立拟合后平均（GraphED 式），使输出是**参数族**而非固定系数方程；
- torch 可用于代理场与张量化评估（`lgf/priors.py` 已验证 torch 依赖可用）。

### 3.7 复杂度与可行性控制

| 约束 | 默认值 | 说明 |
|---|---|---|
| 最大边数 `max_edges` | 14 | 防爆炸 |
| 最大自由参数 `max_params` | 6 | 可按问题收紧（对齐论文 DIF=2、硬化=3） |
| 模板嵌套深度 | 2 | log/exp 只吃多项式子图 P |
| 时间预算 `time_budget_s` | 180 | 单基准墙钟上限 |
| DSL 解析失败率 | 目标 <5% | 失败即丢弃重采样，日志记录用于 prompt 迭代 |

---

## 4. 与代码库的对接现状

> **重要**：LGF 不是"在 `D:\AI\LLM-Feynman` 上打补丁"，而是**独立仓库** `D:\AI\LLM-Feynman-G`。下面 4.1 中的"新增包"已全部落地为实际文件。

### 4.1 实际模块映射（核对 2026-09-23 代码快照）

```
D:\AI\LLM-Feynman-G\                # 独立仓库, 自包含
├── lgf\                            # 主包 (14 个 .py)
│   ├── __init__.py                 # v0.1.0 + 独立性声明
│   ├── config.py                   # LGFConfig 统一配置
│   ├── graph.py                    # DAG + 边特征 + 复杂度控制
│   ├── dsl.py                      # DSL ↔ Graph 双向转换 + 受限解析器
│   ├── compiler.py                 # graph→numpy/torch 可执行函数 (CompiledGraph)
│   ├── templates.py                # 模板库 (GraphED 默认 + 可动态扩展)
│   ├── priors.py                   # Feynman 检验 (量纲/对称/可分/组合) + 数据画像
│   ├── gp.py                       # 图交叉/变异/选择/灭绝
│   ├── paramfit.py                 # L-BFGS-B 联合拟合 (fit_joint / evaluate)
│   ├── attribution.py              # 节点敏感度 + 偏依赖归因
│   ├── llm_ops.py                  # 五类指令 prompt + JSON 解析 + 3 种后端
│   ├── eq_compat.py                # LLM-Feynman 方言 → LGF DSL (纯重实现)
│   ├── benchmarks.py               # 11 个合成基准 + 可选真实 csv 加载
│   └── pipeline_lgf.py             # 分层双循环主控 (run_lgf)
├── main.py                         # 统一 CLI (--framework lgf)
├── scripts\
│   ├── run_benchmarks.py           # 端到端消融基准 (9 题 × 3 配置)
│   └── mock_llm_server.py          # 自带 Flask /completions 模拟器
├── tests\
│   ├── smoke_graph_dsl.py          # 基础冒烟
│   ├── test_lgf.py                 # 16 项单元测试
│   └── test_eq_compat.py           # 方言兼容层测试
└── docs\                           # 验证报告 / 技术报告
```

### 4.2 与 `D:\AI\LLM-Feynman` 的接触面（全部为可选或注释）

| 接触点 | 位置 | 依赖性质 |
|---|---|---|
| 可选数据路径 | `lgf/benchmarks.py::_DEFAULT_REAL_DATA_DIR`，默认 `D:\AI\LLM-Feynman\LLM-Feynman-main\data`，可用 `LGF_REAL_DATA_DIR` 覆盖 | **可选**，路径不存在即返回 `None` |
| Flask 服务协议 | `lgf/llm_ops.py::HttpFlaskLLM`，POST `{"prompt", "repeat_prompt", "params"}` → `{"content": [...]}` | **协议兼容**，需显式 `--llm_backend flask` |
| 方言契约 | `lgf/eq_compat.py` | **纯重实现**，零 import |
| 文档字符串引用 | `lgf/__init__.py`、`main.py`、`llm_ops.py` 等 19 处 | **仅注释/说明** |

### 4.3 主循环结构（已实现于 `lgf/pipeline_lgf.py`）

```python
def run_lgf(cases, var_names, cfg: LGFConfig, log_dir, llm_client, name):
    # 0) 预处理与先验分析
    prof   = priors.data_profile(cases, var_names)          # 单调/饱和/渐近摘要
    priors_ = priors.run_priors(cases, var_names)           # 对称/可分/组合性
    constraints = priors_.to_constraints()                  # → GP 过滤约束

    # 1) 外层①: LLM 提议模板 + 初始骨架 (受 cfg.use_llm 控制)
    population = llm_ops.G_INIT(...) + random_graphs(...)   # 语义 + 随机多样性
    population = gp_ops.select(population)                  # 复杂度/loss 双目标

    for gen in range(cfg.generations):
        # 2) 内层: GP 快循环
        for G in population:
            theta = paramfit.fit_joint(G, cases)            # 多案例联合 L-BFGS-B
            G.loss = evaluate(G, theta, cases)
        population = gp_ops.crossover_and_mutate(population, constraints)
        population = gp_ops.select(population, cfg)
        population = gp_ops.maybe_extinction(population, cfg)

        # 3) 每 K 代: 归因 + 外层②
        if gen % cfg.llm_every == 0 and cfg.use_attribution:
            rep  = attribution.analyze(topk(population), cases, var_names)
            population = apply_cmds(population, llm_ops.M_GUIDE(rep, ...))

    # 4) 收敛期: 符号验证 + 结构选择
    final = llm_ops.E_CHECK(topk(population), ...)
    return {"best": final[0], "top_final": final, ...}
```

### 4.4 CLI 入口

`main.py` 已收敛为 LGF 单一入口；`--framework llmsr` 保留但会打印弃用迁移提示（指向上游 `run_experiment.py`）。

```bash
# 离线启发式 (默认, 零外部依赖)
python main.py --framework lgf --benchmark esat --generations 20 --population 40

# 一组合成基准
python main.py --framework lgf --suite quick

# 真实数据 (可选, 需放在 LGF_REAL_DATA_DIR)
python main.py --framework lgf --real osc1

# Flask 后端 (对接 mock 或上游 llm_engine.py 服务)
python scripts/mock_llm_server.py --port 5050
python main.py --framework lgf --benchmark esat --llm_backend flask --llm_url http://localhost:5050/completions

# OpenAI 兼容 API
python main.py --framework lgf --benchmark hollomon --llm_backend openai --llm_url https://api.openai.com/v1

# 查看基准列表
python main.py --framework lgf --list_benchmarks
```

### 4.5 配置优先级链（三层覆盖）

LLM 后端字段（`llm_backend / llm_url / llm_api_key / llm_model / llm_temperature / llm_timeout / llm_debug`）按下面三层从低到高叠加，**高层覆盖低层**：

```
LGFConfig dataclass 默认
        ↓
config.xml（--config 指定；缺省 ./config.xml 存在则自动加载）
        ↓
CLI 参数（仅当用户显式给出非默认值时覆盖）
```

实现位置：
- `lgf/config.py::LGFConfig.from_xml()` / `to_xml()`：基于 `xml.etree.ElementTree` 的纯标准库解析（无新增依赖）；字段类型表 `_FIELD_KIND` 统一映射 dataclass 字段到 XML 文本；`optional_str` 空文本→`None`，`bool` 支持 `true/false/yes/no/1/0`，未知字段跳过（向前兼容）。
- `main.py::load_config_layered()`：先 `LGFConfig()` 拿默认值，尝试加载 `args.config`（或 `config.xml`），最后用 CLI 显式值覆盖。
- `tests/test_config_xml.py`：37 项测试覆盖类型转换、缺字段默认、圆环回一致性、未知字段跳过、非法类型 / 根节点 / 缺文件的错误路径。

示例配置（`config.xml` 模板已落在仓库根目录）：

```xml
<lgf>
    <llm_backend>heuristic</llm_backend>          <!-- heuristic | flask | openai -->
    <llm_url>https://api.openai.com/v1</llm_url>  <!-- flask 需 .../completions; openai 仅 .../v1 -->
    <llm_api_key>sk-xxxxxxxx</llm_api_key>         <!-- 留空 → 按 LGF_LLM_API_KEY > OPENAI_API_KEY > DASHSCOPE_API_KEY 解析 -->
    <llm_model>gpt-4o-mini</llm_model>
    <llm_temperature>0.7</llm_temperature>
    <llm_timeout>60.0</llm_timeout>
    <llm_debug>false</llm_debug>
    <!-- ... 其他字段: population / generations / max_edges / use_llm 等 -->
</lgf>
```

典型用法：

```bash
# 默认走 ./config.xml
python main.py --framework lgf --benchmark esat

# 多套配置 (dev / prod)
python main.py --framework lgf --benchmark esat --config config.prod.xml

# CLI 临时覆盖
python main.py --framework lgf --benchmark esat --config config.xml \
    --llm_model gpt-3.5-turbo --llm_temperature 0.3
```

注意：
- `--config` 指向不存在的文件时仅打印警告，不抛错，回退到默认 + CLI 覆盖；
- 不传 `--config` 时若 `./config.xml` 存在会自动加载，避免硬编码 URL；
- 生产环境推荐把 `llm_api_key` 留在环境变量（`LGF_LLM_API_KEY`），文件里只放 url / model 等非敏感字段；
- CLI 调试开关 `--llm_debug` 仍独立工作（设置后端失败信息打到 stderr），与 `LGF_LLM_DEBUG` 环境变量等价。

---

## 5. 实验验证计划

| 阶段 | 基准 | 对比方法 | 验证目标 |
|---|---|---|---|
| S1 | Feynman benchmark（Udrescu 100 式） | AI-Feynman 2.0、PySR、DSO、GraphED 复现 | 先验分解收益、恢复率/调用次数 |
| S2 | 上游 `data/` 的基准题 + LGF 合成基准 | LLM-Feynman 正式版（`run_experiment.py`） | 同预算下 loss 与 LLM 调用次数对比 |
| S3 | GraphED 三案例（DIF 合金钢 / 锂 / 橡胶，multicase） | GraphED 复现 + 经验模型（C-S/J-C/Voce…） | 自由参数族发现能力是否保持并提升 |
| S4 消融 | S2+S3 各取 2 题 | ① 去 LLM（纯 GP+先验）② 去 Feynman 先验 ③ 去归因（LLM 随机变异）④ 单层循环 | 每个组件的边际贡献 |

**已完成的消融**（`scripts/run_benchmarks.py`，9 题 × 3 配置 = 27 runs）：

| 配置 | use_llm | use_priors | use_attribution | mean_test_R² |
|---|---|---|---|---|
| GP-only | ✗ | ✗ | ✗ | 0.8181 |
| +Priors | ✗ | ✓ | ✗ | 0.8183 |
| **Full** | ✓ | ✓ | ✓ | **0.8693** |

完全命中的基准：esat（R² = 0.9947）、voce（0.9996）、hollomon（0.9996）。

**核心度量**：test MSE/RMSE、R²、表达式复杂度（边数）、自由参数数、LLM token 总消耗、墙钟时间。

---

## 6. 创新点（可发表贡献）

1. **统一表示融合三源先验**：将 LLM 语义先验、AI-Feynman 物理先验与图结构先验编译到同一"边特征 DAG"搜索空间上（先验不再是外挂的 prompt 提示，而是可执行的图约束与算子）；
2. **LLM 作为语义级遗传算子**：提出归因驱动的定向变异指令集（T-PROP/G-INIT/H-DECOMP/M-GUIDE/E-CHECK），把 GP 的语义盲搜索升级为语义引导搜索；
3. **分层双循环架构**：LLM 调用次数较 LLM-SR 下降 1-2 个数量级，同时保留语义能力——直接回应本地部署 Mixtral 的算力约束；
4. **符号硬验证兜底**：等价性判定 LLM 初筛 + sympy 终审，规避 LLM 幻觉（工程可靠性卖点）；
5. **继承自由参数 + multicase 能力**：可迁移到水动力场景（如阻力系数、螺旋桨推力系数随进速系数/雷诺数的多工况本构发现——比固体本构更接近 MARIC CFD 主战场）。

---

## 7. 风险与对策

| 风险 | 等级 | 对策 |
|---|---|---|
| LLM 输出 DSL 解析失败率高 | 中 | JSON schema 约束 + 失败重采样 + 失败样本回流 prompt 优化；`HeuristicLLM` 兜底保证不中断 |
| Feynman 检验误报（噪声数据下假可分性） | 中 | 3σ 显著性阈值 + LLM 语义复核 + 分解后子问题独立验证回代 |
| 双循环超参（K、灭绝窗口、种群比）难调 | 中 | 先对齐 GraphED 默认，K 网格搜索；`LGFConfig` 已集中管理 |
| 图编译 torch 路径与 numpy 路径数值不一致 | 低 | 单元测试对拍（同一图两种后端 MSE 差 <1e-10） |
| sympy 等价验证超时 | 低 | 仅对 top-K（≤10 张）+ 化简超时即标记"未知等价" |
| 上游 LLM-Feynman 服务未启动 | 低 | 默认 `--llm_backend heuristic`，完全离线可跑 |

---

## 8. 参考文献

1. Xu H, Chen Y, Zhang D. Beyond empirical models: Discovering constitutive laws in solids with graph-based equation discovery[J]. Science Advances, 2026, 12: eaec0989.
2. Udrescu S-M, Tegmark M. AI Feynman 2.0: Pareto-optimal symbolic regression exploiting graph modularity[R]. arXiv:2006.10782, 2020.
3. Udrescu S-M, Tegmark M. AI Feynman: A physics-inspired method for symbolic regression[J]. Science Advances, 2020, 6(16): eaay2631.
4. LLM-SR: Scientific equation discovery via programming with large language models[C]. ICLR, 2025 (Oral).（上游 `LLM-Feynman-main` 的骨架来源之一）
5. LLM-Feynman 正式版代码：`D:\AI\LLM-Feynman\LLM-Feynman-main`（LLM-SR + AI-Feynman 融合框架，本项目的算法脉络来源）
6. PySR / DSO 等基线：见文献[1]之引用 41, 42。

---

## 9. 实施里程碑（含已完成项）

| 阶段 | 内容 | 状态 |
|---|---|---|
| D1-D3 | `graph.py + dsl.py + compiler.py`，打通"图 ↔ DSL ↔ numpy 可执行"闭环 | ✅ 已完成（`tests/smoke_graph_dsl.py` 通过） |
| D4-D6 | `templates.py + gp.py + paramfit.py`，复现单案例本构结果 | ✅ 已完成（esat R²=0.9947 / voce, hollomon 0.9996） |
| D7-D9 | `priors.py` 五类检验 + 图约束编译 | ✅ 已完成（`tests_to_constraints` 接入 GP） |
| D10-D12 | `llm_ops.py + attribution.py`，接通 Flask 引擎，跑 S2 消融 | ✅ 已完成（27 runs，Full 0.8693 vs GP-only 0.8181） |
| D13-D14 | `pipeline_lgf.py` 双循环整合 + multicase 扩展 + S3 准备 | ✅ 已完成（`esat_mc` 等多案例基准就绪） |
| D15 | `config.py::from_xml/to_xml` + `main.py --config` + `config.xml` 模板 + `test_config_xml.py` | ✅ 已完成（37 项测试 + 端到端 CLI 验证） |
| 后续 | S1（Feynman 100 式）/ S3（GraphED 三案例）/ 水动力场景迁移 | ⏳ 待推进 |
