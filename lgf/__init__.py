"""LGF — LLM-guided Graph-Feynman 方程发现框架.

融合对象 (历史脉络, 不构成运行期依赖):
- LLM-Feynman (D:\\AI\\LLM-Feynman\\LLM-Feynman-main): LLM-SR 骨架 + 本地 LLM 引擎
- GraphED (Sci. Adv. 2026, eaec0989): 方程=带边特征的有向无环图
- AI-Feynman: 物理先验 (量纲/对称/可分/组合性)

独立性说明:
    本包完全自包含, 不 import 也无需安装 D:\\AI\\LLM-Feynman\\LLM-Feynman-main.
    LGF 自带 GP / 先验检验 / 启发式 LLM / 编译器 / 模板库 / 归因分析 全部组件.

公共 API:
    from lgf import pipeline_lgf, benchmarks, LGFConfig
    pipeline_lgf.main(cases=..., cfg=LGFConfig(...))
"""

__version__ = "0.1.0"
