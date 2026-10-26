# LLM-Feynman-G (LGF)

**LLM-guided Graph-Feynman 方程发现框架** —— 融合三种方程发现范式的分层双循环，落地两个物理建模应用：旋转槽道流 ASM 湍流闭合 与 船舶操纵运动 MMG 建模。

> **完全自包含**：不依赖任何外部框架路径，自带编译器 / 模板库 / GP 算子 / 先验 / 归因 / LLM 客户端。
> 智能体化继任版本见 [LGF-Agent](https://github.com/waynezw0618/LGF-Agent)（五层架构 + 生成/审查分离）。

LGF = LLM-guided Graph-Feynman，融合三种方程发现范式：

| 范式 | 角色 |
|---|---|
| **GraphED** (Sci. Adv. 2026) | 统一表示层 — 方程 = 带边特征的有向无环图 (root DAG) |
| **AI-Feynman** (Sci. Adv. 2020) | 物理先验 — 量纲/对称/可分/组合性检验 |
| **LLM-SR** (ICLR 2025 Oral) | 语义级遗传算子 — 模板生成 + 定向变异 |

架构：外层 LLM 慢循环（每 K 代一次模板提议 / 归因驱动变异）+ 内层 GP 快循环（每代一次 crossover/mutation/selection）。

---

## 核心结果（完整报告见 `outputs/asm/`，摘要如下）

### 应用一：旋转槽道流 ASM 张量基闭合（Yang & Wang 2018, JFM 838）

数据：旋转槽道流 DNS（Re_τ=180，Ro_τ=7.5/15/30/75/150 五工况）。

**① LGF vs PySR 系统对照（20 次独立运行，T₂₃ 通道入池）**

| 指标 | LGF (8 runs) | PySR (12 runs, 3 种子) | 比值 |
|---|---|---|---|
| loss_vw 总体 | **0.087 ± 0.036** | 0.190 ± 0.091 | **2.2× 优** |
| 复跑波动 σ | 小 | — | 逐变体小 1.2–9.6× |
| 最差单工况 NMSE | **0.154** | 0.346 | 2.2× |
| 公式节点数 | **15.1** | 24.2 | 少 38% |

LGF **8/8 全部低于对应变体 PySR 三种子均值**；PySR 最优 0.0418 出现在幸运种子（同配置其他种子差 4–8×，不可复现）；LGF 最优 `lgf_go/m` **0.0514** 进入金标准带（0.04–0.07）。报告：`outputs/asm/t23full_paper/`。

**② 方案 B2：多通道张量基系数反演 + 联合重拟合（当前主力口径）**

$$b_{23}=\sum_j s_j\, a_j(\Theta)\cdot \tilde T^{(j)}_{23}$$

- 系数必须**联合确定**：抵消比 Σ‖C‖/‖b23‖ 高达 99–32201×，逐通道独立拟合后重组误差爆到 3.8–4.5 —— 这是"逐分量 SR 从不用张量基"的机理根源，也是联合 refit 的价值所在（g 0.646→0.152 / e 0.983→0.111）；
- 大形状轮 + 联合 LM 抛光：**e 基 0.0576（55 参）**进入标量 SOTA 0.0514 的 12% 以内，g 基 0.124 ≈ PySR full refit 0.128 且复杂度低 3–4×；
- 知识消融：g 基 physics 0.693 vs generic 1.271（**−45%**）—— LLM 物理先验 = 隐式正则（等复杂度下 PySR maxsize 25→12 全面退化，LGF 不变更优）。

报告：`outputs/asm/t23basisB2_paper/BasisB2_full_matrix_report.md`（29 组两口径全矩阵）+ `formulas_champions.md`（变体冠军公式）。

### 应用二：船舶操纵运动 MMG 建模（shipman 子项目）

四变体 MMG 模型（V1 传统 / V2 分数阶 / V3 记忆核积分 / V4 分数阶+积分），KVLCC2.L7 文献系数合成轨迹反演，23 runs 完整评估 + 中文 arXiv 风格论文源码。见 `outputs/shipman/paper/`（main.tex + 图 + 数字溯源 json）。

---

## 快速开始

```bash
pip install -r requirements.txt

# 单元测试
python tests/smoke_graph_dsl.py        # DSL/图往返冒烟, 数秒
python tests/test_lgf.py               # 16 项全模块测试, ~1 min

# 基准消融 (9 题 × 3 配置)
python scripts/run_benchmarks.py --quick    # ~5 min
python main.py --framework lgf --benchmark esat --generations 20 --population 40
```

真实 LLM 后端（OpenAI 兼容 API，如 GLM）：

```bash
python main.py --framework lgf --benchmark hollomon \
    --llm_backend openai --llm_url https://open.bigmodel.cn/api/paas/v4
```

## ASM 反演跑批入口

```bash
# 数据准备: data/*.txt (DNS 55 列) -> outputs/asm/dataset_*.npz
python scripts/rc_asm_prep.py

# 方案 B2 全矩阵 (LGF 两臂 + PySR 对照, 支持 --stage refit 联合抛光)
python scripts/rc_asm_basisB2_driver.py --variant e --arm lgf_mg

# 冠军公式提取 (LaTeX + numpy)
python scripts/rc_asm_basisB2_formulas.py
```

## 项目结构

```
lgf/            主包: graph/dsl/compiler/paramfit/priors/templates/gp/
                attribution/llm_ops(heuristic|flask|openai)/benchmarks/pipeline_lgf
shipman/        船舶 MMG 子项目 (V1-V4 变体, 零改动接入 lgf 内核)
scripts/        rc_asm_* (ASM 跑批/聚合/出图) + run_benchmarks + mock_llm_server
tests/          冒烟 + 16 项单元测试
data/           旋转槽道流 DNS 原始数据 (5 工况, 55 列)
outputs/asm/    dataset_*.npz 反演数据集 + 全部实验报告 (ASM_*.md, *_paper/)
outputs/shipman/  MMG 实验产物 + 论文材料包
docs/           验证报告 / 融合技术报告 / LGF-Agent 方案
main.py         统一 CLI
```

## 公共 API

```python
from lgf import LGFConfig
from lgf.pipeline_lgf import run_lgf
from lgf.benchmarks import REGISTRY, make_cases
from lgf.llm_ops import make_client

bm = REGISTRY["esat"]()
cases = make_cases(bm, n_per_case=200, noise_rel=0.01, seed=0)
cfg = LGFConfig(population=24, generations=15, use_llm=True,
                use_priors=True, use_attribution=True)
result = run_lgf(cases, bm.var_names, cfg, log_dir="outputs/run",
                 llm_client=make_client("heuristic"), name="esat")
print(result["best"]["dsl"], result["best"]["loss"])
```

## 参考文献

- Yang & Wang (2018). *Structural dynamics and intrusions in a rotating channel flow*. JFM 838
- Pope, S. B. (1975). *A more general effective-viscosity hypothesis*. JFM 72:331
- Weatheritt & Sandberg (2016). JCP 325:22-37
- Yasukawa & Yoshimura (2015). J Mar Sci Technol 20:37-52（KVLCC2 MMG 系数）
- AI-Feynman / GraphED / LLM-SR（见 `references/`，版权原因不入库）

## 引用

```bibtex
@misc{lgf2026,
  title  = {LGF: LLM-guided Graph-Feynman for Scientific Equation Discovery},
  author = {Wei Zhang},
  year   = {2026},
  note   = {MARIC WJPLAB}
}
```

## 许可

MIT。
