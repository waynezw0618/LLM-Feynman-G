# -*- coding: utf-8 -*-
"""生成 LGF 算法流程图 (论文/文档用, 矢量风格 box-arrow)."""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(OUT, exist_ok=True)

# 配色
C_DATA   = "#E8F1FB"; C_DATA_E   = "#2E6FB7"
C_OUTER  = "#FDECEC"; C_OUTER_E  = "#C0392B"
C_INNER  = "#EAF6EC"; C_INNER_E  = "#1E8449"
C_PRIOR  = "#FFF6E5"; C_PRIOR_E  = "#B9770E"
C_ATTR   = "#F2EAFB"; C_ATTR_E   = "#6C3483"
C_IO     = "#ECECEC"; C_IO_E     = "#566573"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def box(ax, x, y, w, h, text, fc, ec, fs=10, weight="normal", rounded=0.02):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.004,rounding_size={rounded}",
                       linewidth=1.6, facecolor=fc, edgecolor=ec, zorder=2)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=3, weight=weight, linespacing=1.45)


def arrow(ax, p1, p2, color="#34495E", style="-|>", lw=1.7, rad=0.0, ls="-"):
    a = FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=16,
                        linewidth=lw, color=color, zorder=1,
                        connectionstyle=f"arc3,rad={rad}", linestyle=ls)
    ax.add_patch(a)


def make_main_flow():
    fig, ax = plt.subplots(figsize=(11.5, 8.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ---- 数据输入 ----
    box(ax, 34, 90, 32, 7.5,
        "输入: 多案例数据 {(X_c, y_c)}\n变量名 / 量纲 / 数据画像",
        C_IO, C_IO_E, fs=10.5, weight="bold")

    # ---- AI-Feynman 先验模块 (左) ----
    box(ax, 2, 62, 25, 22,
        "AI-Feynman 物理先验\n"
        "· 代理神经网络拟合\n"
        "· 对称性检验\n"
        "· 加性/乘性可分性\n"
        "· 广义组合性\n"
        "· 量纲分析 (Π 组)\n"
        "→ 编译为图约束",
        C_PRIOR, C_PRIOR_E, fs=9.5, weight="bold")

    # ---- 外层 LLM 慢循环 ----
    box(ax, 32, 58, 40, 26,
        "外层循环: LLM 语义进化 (慢, 每 K 代)\n"
        "T_PROP  模板提议\n"
        "G_INIT  初始骨架生成\n"
        "H_DECOMP 分解假设 → 图约束\n"
        "M_GUIDE  归因驱动定向变异\n"
        "E_CHECK  等价判定与结构选择",
        C_OUTER, C_OUTER_E, fs=10, weight="bold")

    # ---- DSL 转换条 ----
    box(ax, 40, 50, 24, 5.2,
        "DSL <-> Graph 双向转换 (受限解析)",
        "#FFFFFF", C_OUTER_E, fs=9.5)

    # ---- 内层 GP 快循环 ----
    box(ax, 32, 22, 40, 25,
        "内层循环: GP 图进化 (快, 每代)\n"
        "· 图交叉 / 参数扰动 / 算子替换 / 子图替换\n"
        "· top-50% 选择 + 灭绝机制\n"
        "· L-BFGS 多启动拟合边参数 (逐案例)\n"
        "· 先验约束过滤 (对称/可分)",
        C_INNER, C_INNER_E, fs=10, weight="bold")

    # ---- 误差归因模块 (右下) ----
    box(ax, 74, 27, 24, 18,
        "误差归因模块\n"
        "· 提取节点值 z_v(x)\n"
        "· |corr(z_v, 残差)|\n"
        "· 偏依赖曲线形状\n"
        "→ 定位最差子图",
        C_ATTR, C_ATTR_E, fs=9.5, weight="bold")

    # ---- 输出 (底) ----
    box(ax, 30, 7, 44, 9,
        "输出\n最优方程 DSL + 参数族 θ*\n(全局参数 C# / 逐案例参数 K#)  +  等价类排序",
        C_DATA, C_DATA_E, fs=10.5, weight="bold")

    # ---- 箭头 ----
    arrow(ax, (50, 90), (50, 84))                       # 数据 → 外层
    arrow(ax, (27, 73), (32, 73))                       # 先验 → 外层
    arrow(ax, (50, 58), (50, 55.2))                     # 外层 → DSL
    arrow(ax, (50, 50), (50, 47.2))                     # DSL → 内层
    arrow(ax, (32, 34), (27, 34), rad=0.0)              # 内层 ← 先验(虚线)
    arrow(ax, (72, 34), (74, 36), rad=0.0)              # 内层 → 归因
    arrow(ax, (86, 45), (66, 58), color="#6C3483", rad=-0.2)  # 归因 → 外层(反馈)
    ax.text(79, 54, "归因反馈\n(定向变异依据)", fontsize=8.5, color="#6C3483",
            ha="center", va="center", style="italic")
    arrow(ax, (50, 22), (50, 16))                       # 内层 → 输出
    arrow(ax, (86, 27), (74, 14), color="#6C3483", rad=0.15)  # 归因 → 输出

    # 虚线标注: 先验约束进入内层
    arrow(ax, (14.5, 62), (14.5, 34), color=C_PRIOR_E, ls=(0, (5, 4)), lw=1.4)
    arrow(ax, (14.5, 34), (32, 34), color=C_PRIOR_E, ls=(0, (5, 4)), lw=1.4)
    ax.text(9, 47, "结构约束\n注入种群", fontsize=8.5, color=C_PRIOR_E,
            ha="center", va="center", style="italic", rotation=90)

    ax.text(50, 98.5, "LGF 分层双循环架构", fontsize=15, weight="bold",
            ha="center", va="center")
    fig.tight_layout()
    p = os.path.join(OUT, "flowchart_main.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


def make_cycle_flow():
    """一个进化周期的信息流 (横向泳道式)."""
    fig, ax = plt.subplots(figsize=(12.5, 5.6))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    y0, h = 58, 26
    # 1 内层
    box(ax, 2, y0, 20, h,
        "① 内层 GP\n跑 K 代\n产出 top-K 图\n(+loss)", C_INNER, C_INNER_E, fs=9.5, weight="bold")
    # 2 归因
    box(ax, 25, y0, 20, h,
        "② 误差归因\n节点敏感度\n+ 残差相关性\n→ 归因报告", C_ATTR, C_ATTR_E, fs=9.5, weight="bold")
    # 3 外层 LLM
    box(ax, 48, y0, 22, h,
        "③ 外层 LLM\n读入 top-K DSL\n+ loss + 归因\n+ 数据画像", C_OUTER, C_OUTER_E, fs=9.5, weight="bold")
    # 4 指令
    box(ax, 73, y0, 25, h,
        "④ 结构化指令集\nT_PROP / G_INIT\nH_DECOMP / M_GUIDE\nE_CHECK", "#FFFFFF", C_OUTER_E, fs=9.5, weight="bold")
    # 5 落地
    box(ax, 48, 18, 22, 20,
        "⑤ 指令解析\nDSL→Graph\n写入种群\n(非法即丢弃)", C_DATA, C_DATA_E, fs=9.5, weight="bold")
    # 6 循环
    box(ax, 22, 18, 20, 20,
        "⑥ 继续进化\n直到收敛/\n预算耗尽", C_IO, C_IO_E, fs=9.5, weight="bold")

    arrow(ax, (22, y0 + h / 2), (25, y0 + h / 2))
    arrow(ax, (45, y0 + h / 2), (48, y0 + h / 2))
    arrow(ax, (70, y0 + h / 2), (73, y0 + h / 2))
    arrow(ax, (85, y0), (85, 38), rad=0.0)
    arrow(ax, (85, 38), (70, 38), rad=0.0)
    arrow(ax, (48, 28), (42, 28))
    arrow(ax, (22, 28), (22, y0), color="#566573", rad=0.0)
    ax.text(20, 48, "反馈回路", fontsize=8.5, color="#566573", ha="center", style="italic")

    ax.text(50, 93, "一个 LGF 进化周期的信息流", fontsize=14, weight="bold", ha="center")
    fig.tight_layout()
    p = os.path.join(OUT, "flowchart_cycle.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", p)


if __name__ == "__main__":
    make_main_flow()
    make_cycle_flow()
