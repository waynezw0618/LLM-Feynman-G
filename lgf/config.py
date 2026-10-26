"""lgf.config — 统一配置.

集中所有可调旋钮, 方便命令行覆盖与实验扫描.

配置优先级 (低 → 高):
    1. LGFConfig dataclass 内置默认值
    2. 环境变量 (主要用于 LGF_LLM_API_KEY 等敏感字段)
    3. config.xml 文件 (通过 LGFConfig.from_xml 加载)
    4. CLI 参数 (main.py --llm_url / --llm_api_key 等)
    → 同名字段被多层定义时, 高优先级层覆盖低优先级层

XML 接入:
    - LGFConfig.from_xml(path): 解析 XML, 缺字段回退到 dataclass 默认
    - LGFConfig.to_xml(cfg, path): 把 cfg 序列化回 XML
    - XML 根节点 <lgf>, 字段以子元素形式组织 (与 dataclass 字段同名)
"""
from __future__ import annotations
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, Tuple, List, Any


# ============================================================================
# 字段元数据: 维护一份 (name, kind) 列表供 XML 序列化使用
# kind ∈ {str, int, float, bool, optional_str, list_str}
# optional_str 表示 Optional[str]; list_str 表示 List[str]
# ============================================================================
_FIELD_KIND: dict = {
    # 种群与进化
    "population": "int", "generations": "int",
    "offspring_ratio": "float", "crossover_rate": "float", "mutation_rate": "float",
    # 复杂度
    "max_edges": "int", "max_params": "int", "max_depth": "int",
    # 外层 LLM 循环
    "llm_every": "int", "stagnation_window": "int", "extinction_keep": "int",
    "use_llm": "bool", "use_attribution": "bool", "use_priors": "bool",
    # 拟合
    "gp_starts": "int", "gp_maxiter": "int",
    "final_starts": "int", "final_maxiter": "int", "sub_sample": "int",
    "param_init_range": "float", "init_value": "float", "fit_tol": "float",
    "screen_budget_s": "float", "gp_fit_budget_s": "float",
    # 验证
    "equivalence_tol": "float",
    # LLM 后端
    "llm_backend": "str", "llm_url": "optional_str", "llm_model": "optional_str",
    "llm_api_key": "optional_str", "llm_temperature": "float",
    "llm_timeout": "float", "llm_debug": "bool",
    # 其它
    "seed": "int", "time_budget_s": "float", "log_every": "int", "output_dir": "str",
    "case_names": "list_str",
    # P1d: LLM 引导改进旋钮
    "forbidden_ops": "list_str", "llm_ginit_only": "bool",
    "wall_seeds": "bool",
    # v3a: G_INIT 鲁棒性
    "ginit_retry": "int", "ginit_min_kept": "int", "ginit_fewshot": "bool",
    # true-ASM (2026-10-09): 物理 LLM 上下文 + 全局常数硬约束
    "physics_context": "str", "global_params_only": "bool",
}


def _coerce(kind: str, raw: str) -> Any:
    """字符串 → 字段类型. 失败时抛 ValueError, 由 from_xml 决定是跳过还是抛错."""
    if kind == "str":
        return raw
    if kind == "optional_str":
        return raw if raw else None
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        s = raw.strip().lower()
        if s in ("1", "yes", "true", "on"):  return True
        if s in ("0", "no", "false", "off", ""): return False
        raise ValueError(f"无法解析 bool: {raw!r}")
    if kind == "list_str":
        return [x.strip() for x in raw.split(",") if x.strip()]
    raise ValueError(f"未知字段类型: {kind}")


def _format(kind: str, value: Any) -> str:
    """字段值 → XML 文本表示."""
    if kind in ("str", "optional_str"):
        return "" if value is None else str(value)
    if kind == "bool":
        return "true" if value else "false"
    if kind == "list_str":
        return ", ".join(value or [])
    return str(value)


@dataclass
class LGFConfig:
    # ===== 种群与进化 =====
    population: int = 50                # 种群规模
    generations: int = 40               # 最大代数 (主终止条件)
    offspring_ratio: float = 1.0        # 每次产生新个体数 = population * ratio
    crossover_rate: float = 0.5
    mutation_rate: float = 0.7

    # ===== 复杂度控制 (继承 GraphED 限制) =====
    max_edges: int = 14
    max_params: int = 6                 # 全局 + 案例 参数总数上限
    max_depth: int = 6

    # ===== 外层 LLM 循环 =====
    llm_every: int = 5                  # 每 K 代触发一次归因 + LLM 干预
    stagnation_window: int = 12         # 多少代无改进触发灭绝
    extinction_keep: int = 4            # 灭绝时保留 top-N
    use_llm: bool = True
    use_attribution: bool = True
    use_priors: bool = True

    # ===== 拟合 =====
    gp_starts: int = 2                  # GP 阶段每图 L-BFGS 多启动次数
    gp_maxiter: int = 30
    final_starts: int = 20              # 最终精修
    final_maxiter: int = 200
    sub_sample: int = 200               # GP 阶段子采样点数
    param_init_range: float = 1.5       # 自由参数随机初始化区间
    init_value: float = 0.5             # 自由参数默认值 (用于 first start)
    fit_tol: float = 1e-9
    # V4 (2026-10-12): GP 阶段单图 paramfit 预算 (硬限时, 见 paramfit.fit_joint);
    # 初筛 (gen0 种群) 每图预算. 旧版全种群每代重拟合 → budget 被 paramfit 吃光,
    # GP 实际代数远低于 generations.
    screen_budget_s: float = 3.0        # gen0 初筛: 单图 1-start 快速拟合
    gp_fit_budget_s: float = 10.0       # 进化期: 新个体单图拟合预算

    # ===== 验证 =====
    equivalence_tol: float = 1e-3       # 与真值函数数值等价相对误差

    # ===== LLM 后端 =====
    llm_backend: str = "heuristic"      # heuristic | flask | openai
    # flask 模式需填到 .../completions; openai 模式只填到版本段 (如 .../v1)
    llm_url: Optional[str] = None
    llm_model: Optional[str] = None     # 空则 flask 忽略 / openai 回退 gpt-3.5-turbo
    # 空则按 LGF_LLM_API_KEY > OPENAI_API_KEY > DASHSCOPE_API_KEY 环境变量解析
    llm_api_key: Optional[str] = None
    llm_temperature: float = 0.7
    llm_timeout: float = 60.0
    llm_debug: bool = False             # True 时后端失败打印到 stderr

    # ===== 其它 =====
    seed: int = 0
    time_budget_s: float = 600.0        # 单个 benchmark 软上限
    log_every: int = 1
    output_dir: str = "outputs"

    # ===== P1d: LLM 引导改进旋钮 =====
    # B3: 目标级禁用算子 (如 T5g 纯幂律先验: ["sub", "div"]) —
    #     apply_mutations 节点级硬拦截 + M_GUIDE prompt 硬规则
    forbidden_ops: List[str] = field(default_factory=list)
    # C2: LLM 仅出 G_INIT 初始种群, GP 进化全程离线 (量化 M_GUIDE 中途干预净贡献)
    llm_ginit_only: bool = False
    # B2: 壁面特征程序化种子注入 (共享机制; 裸配置消融时置 False)
    wall_seeds: bool = True

    # ===== v3a: G_INIT 鲁棒性改进 (A1 解析重试 + A3 few-shot) =====
    ginit_retry: int = 2            # G_INIT 解析率不达标时最大重试次数 (0=关闭, 单次)
    ginit_min_kept: int = 3          # 重试终止阈值: n_parsed >= 此值即停
    ginit_fewshot: bool = True       # G_INIT prompt 注入 few-shot 合法 DSL 示例

    # ===== true-ASM (2026-10-09): 代数应力模型专用旋钮 =====
    # 物理 LLM 上下文 — 注入 G_INIT/M_GUIDE prompt, 让 LLM 带物理先验出候选
    # (张量基含义 / 不变量 / Ro_loc 工况分离 / 涡黏参考结构 等)
    physics_context: str = ""
    # 真 ASM 硬约束: 只允许全局常数 C#; 含逐工况常数 K# 的图一律拒绝
    # (G_INIT 解析后 / M_GUIDE apply_mutations 后 / 最终精修 三处拦截)
    global_params_only: bool = False

    # 内部用
    case_names: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    # ------------------------------------------------------------------
    # XML 序列化 / 反序列化
    # ------------------------------------------------------------------
    @classmethod
    def from_xml(cls, path: str) -> "LGFConfig":
        """从 config.xml 加载 LGFConfig.

        - 缺字段 → 沿用 dataclass 默认值
        - 类型无法转换 → 抛 ValueError 并指明字段名
        - 找不到文件 → 抛 FileNotFoundError (由调用方决定 fallback)
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"config.xml 不存在: {path}")
        tree = ET.parse(path)
        root = tree.getroot()
        if root.tag != "lgf":
            raise ValueError(f"config.xml 根节点必须是 <lgf>, 当前是 <{root.tag}>")
        cfg = cls()                                           # 先取全部默认值
        for elem in root:
            name = elem.tag
            if name not in _FIELD_KIND:
                # 未知字段: 跳过但打印提示 (向前兼容未来字段)
                print(f"[LGF-Config] 未知字段 <{name}>, 跳过")
                continue
            kind = _FIELD_KIND[name]
            text = (elem.text or "").strip()
            if text == "" and kind in ("str", "optional_str", "list_str"):
                # 空字符串对 optional_str 表示 None; 对 str 表示 ""; 对 list_str 表示 []
                if kind == "optional_str":
                    setattr(cfg, name, None)
                elif kind == "list_str":
                    setattr(cfg, name, [])
                else:
                    setattr(cfg, name, "")
                continue
            try:
                setattr(cfg, name, _coerce(kind, text))
            except Exception as e:
                raise ValueError(f"config.xml <{name}> 解析失败: {e}") from e
        return cfg

    def to_xml(self, path: str) -> None:
        """把当前 cfg 写成 XML 文件. 父目录不存在会自动创建."""
        root = ET.Element("lgf")
        for f in fields(self):
            if f.name not in _FIELD_KIND:
                continue
            kind = _FIELD_KIND[f.name]
            child = ET.SubElement(root, f.name)
            child.text = _format(kind, getattr(self, f.name))
        # 写文件
        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
