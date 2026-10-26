"""lgf.eq_compat — 与 LLM-Feynman 方言的兼容适配层 (纯重实现, 零运行期依赖).

背景:
    参考实现 D:\\AI\\LLM-Feynman\\LLM-Feynman-main\\llm_feynman\\llm_engine.py 定义了
    一套轻量"方程骨架"契约:
      (1) LLM 用 `EQ: <python-expr>` 逐行输出候选;
      (2) 未知常数写成 `params[i]`;
      (3) 允许 numpy 函数 `np.sqrt / np.exp / np.log / np.sin / np.cos / np.pi`;
      (4) 内置一个"物理上下文感知"模板库 (振子/能量/引力/动量/单摆/LC 电路/
          Stefan-Boltzmann/理想气体/透镜/库仑/洛伦兹/相对论/德布罗意/向心/
          加性可分/乘性可分/标度对称).

本模块把上述两项能力**移植**到 LGF 自有 DSL 上, 供 LGF 的启发式 LLM 与外部
`--llm_backend flask` 路径复用. 本模块不 import llm_feynman, 也不要求其存在.

映射约定 (LLM-Feynman 方言 → LGF DSL):
    params[i]        -> C{i+1}          (全局自由参数)
    np.sqrt(a)       -> sqrt(a)
    np.exp(a)        -> exp(a)
    np.log(a)        -> log(a)
    np.sin(a)        -> sin(a)   np.cos(a) -> cos(a)
    np.pi            -> 3.141592653589793   (数值常量)
    a ** 2 / 3 / 4   -> a^2 / a^3 / a^4
    a ** 0.5         -> sqrt(a)
    a ** params[k]   -> exp(C{k+1} * log(a))   (借助 log/exp 实现非整数幂)
    a ** (-1)        -> inv(a)
"""
from __future__ import annotations
import json
import math
import re
from typing import Dict, List, Optional, Sequence

import sympy as sp


# ============================================================================
# 1. 物理上下文感知模板库 (移植自 llm_engine.py::_generate_context_aware_skeletons)
#    以 LGF DSL 表达; {v0}/{v1}/... 为变量占位, C# 为自由参数.
# ============================================================================
PHYSICS_CONTEXT_TEMPLATES: Dict[str, List[str]] = {
    # 关键词: ["oscillator", "harmonic", "spring", "frequency"]
    "oscillator": [
        "C1 * sqrt($1 / $2)",
        "sqrt($1 / $2 - ($3 / (2 * $1))^2)",
    ],
    # ["energy", "kinetic"]
    "energy": [
        "C1 * $1 * $2^2",
    ],
    # ["gravit", "force", "inverse square", "coulomb", "charge"]
    "inverse_square": [
        "C1 * $1 * $2 * $3 / $4^2",
    ],
    # ["momentum"]
    "momentum": [
        "C1 * $1 * $2",
    ],
    # ["pendulum", "period"]
    "pendulum": [
        "C1 * 3.141592653589793 * sqrt($1 / $2)",
    ],
    # ["circuit", "resonant", "lc", "frequency"]
    "lc_circuit": [
        "C1 / sqrt($1 * $2)",
    ],
    # ["stefan", "boltzmann", "radiation"]
    "stefan_boltzmann": [
        "C1 * $1 * $2^4",
    ],
    # ["gas", "ideal", "pressure"]
    "ideal_gas": [
        "C1 * $1 * $2 * $3 / $4",
    ],
    # ["lens", "focal", "optics"]
    "lens": [
        "C1 / (inv($1) - (C2 - 1) * (inv($2) - inv($3)))",
    ],
    # ["lorentz", "magnetic force"]
    "lorentz": [
        "C1 * $1 * $2 * $3",
    ],
    # ["relativist", "e=mc", "rest energy"]
    "relativistic": [
        "C1 * $1 * $2^2",
    ],
    # ["de broglie", "wavelength"]
    "de_broglie": [
        "C1 * $1 / ($2 * $3)",
    ],
    # ["centripetal", "acceleration", "circular"]
    "centripetal": [
        "C1 * $1^2 / $2",
    ],
}


# 关键词 → 模板域名 (顺序敏感: 先匹配先返回)
_CONTEXT_KEYWORDS: List[tuple] = [
    ("oscillator", ("oscillator", "harmonic", "spring", "frequency")),
    ("pendulum", ("pendulum", "period")),
    ("inverse_square", ("gravit", "force", "inverse square", "coulomb", "charge")),
    ("energy", ("energy", "kinetic")),
    ("momentum", ("momentum",)),
    ("lc_circuit", ("circuit", "resonant", "lc", "frequency")),
    ("stefan_boltzmann", ("stefan", "boltzmann", "radiation")),
    ("ideal_gas", ("gas", "ideal", "pressure")),
    ("lens", ("lens", "focal", "optics")),
    ("lorentz", ("lorentz", "magnetic force")),
    ("relativistic", ("relativist", "e=mc", "rest energy")),
    ("de_broglie", ("de broglie", "wavelength")),
    ("centripetal", ("centripetal", "acceleration", "circular")),
]

# 可分性关键词 (按变量数自适应, 单独处理)
_ADDITIVE_KWS = ("additive separab", "f ≈ g")
_MULTIPLICATIVE_KWS = ("multiplicative separab",)


def _separable_skeleton(kind: str, var_names: Sequence[str]) -> Optional[str]:
    """按变量数把变量对半分, 生成加性/乘性可分骨架.

    n=2: additive -> C1*($1) + C2*($2);  multiplicative -> C1*($1) * ($2)
    n=4: additive -> C1*(a*b) + C2*(c*d)
    """
    n = len(var_names)
    if n < 2:
        return None
    half = max(1, n // 2)
    ga = [f"({v})" for v in var_names[:half]]
    gb = [f"({v})" for v in var_names[half:]]
    if not gb:
        gb = [ga.pop()]
    if kind == "additive":
        return f"C1 * {' * '.join(ga)} + C2 * {' * '.join(gb)}"
    return f"C1 * {' * '.join(ga)} * {' * '.join(gb)}"


def instantiate_context_template(tpl: str, var_names: Sequence[str]) -> Optional[str]:
    """把模板里的 $1/$2/... 槽位按 var_names 顺序填入 (索引从 1 起).

    返回 None 表示变量数不足.
    """
    idxs = sorted({int(m) for m in re.findall(r"\$(\d+)", tpl)})
    if idxs and max(idxs) > len(var_names):
        return None
    out = tpl
    for i in idxs:
        out = out.replace(f"${i}", f"({var_names[i - 1]})")
    return out


def context_aware_skeletons(problem_context: str, var_names: Sequence[str],
                            num_skeletons: int = 4) -> List[str]:
    """依据问题上下文关键词, 返回若干 LGF DSL 骨架 (未做去重/配额截断前的全量)."""
    ctx = (problem_context or "").lower()
    out: List[str] = []
    for domain, kws in _CONTEXT_KEYWORDS:
        if any(w in ctx for w in kws):
            for tpl in PHYSICS_CONTEXT_TEMPLATES.get(domain, []):
                inst = instantiate_context_template(tpl, var_names)
                if inst is not None:
                    out.append(inst)
    # 加性可分: 按变量数自适应生成 (取代原写死 4 变量模板)
    if any(w in ctx for w in _ADDITIVE_KWS):
        s = _separable_skeleton("additive", var_names)
        if s is not None:
            out.append(s)
    if any(w in ctx for w in _MULTIPLICATIVE_KWS):
        s = _separable_skeleton("multiplicative", var_names)
        if s is not None:
            out.append(s)
    # 标度对称提示: 形如 "y ∝ v^2"
    for m in re.finditer(r"([a-z_]\w*)\s*∝\s*([a-z_]\w*)\^(\d+)", ctx):
        var_sym = m.group(2)
        power = m.group(3)
        if var_sym in [v.lower() for v in var_names]:
            others = [v for v in var_names if v.lower() != var_sym]
            if others:
                out.append(f"C1 * {var_sym}^{power} * {others[0]}")
    return out


# ============================================================================
# 2. EQ: 方言 → LGF DSL 适配器
# ============================================================================
_EQ_PREFIX = re.compile(r"^\s*EQ\s*:\s*", re.IGNORECASE)
_PARAM_RE = re.compile(r"params\[(\d+)\]")
_NP_FUNC_RE = re.compile(r"(?<![\w.])np\s*\.\s*(sqrt|exp|log|sin|cos|tan|tanh|abs|pi)\b")


def extract_eq_lines(text: str) -> List[str]:
    """从 LLM-Feynman 风格的响应中抽取 `EQ:` 行 (无前缀时按行返回)."""
    lines: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        if _EQ_PREFIX.match(s):
            body = _EQ_PREFIX.sub("", s).strip()
            if body:
                lines.append(body)
        elif lines == [] and "=" in s:
            # 兼容 "y = <expr>" 单行形式
            body = s.split("=", 1)[1].strip()
            if body:
                lines.append(body)
    return lines


def normalize_eq_expr(expr: str, var_names: Sequence[str]) -> str:
    """把 LLM-Feynman 方言规范化成 sympy 可解析的表达式字符串.

    - params[i] -> C{i+1}
    - np.f(x)   -> f(x);  np.pi -> 数值
    - {v0} 风格占位 -> 对应变量名
    - Python 幂 `**` 原样保留 (sympy 支持)
    """
    s = expr.strip()
    # 花括号占位 {v0} / {0}
    def _brace(m):
        idx = int(m.group(1))
        return var_names[idx] if idx < len(var_names) else m.group(0)
    s = re.sub(r"\{v?(\d+)\}", _brace, s)
    # params[i]
    s = _PARAM_RE.sub(lambda m: f"C{int(m.group(1)) + 1}", s)
    # np.pi -> 数值; np.func -> func
    s = re.sub(r"(?<![\w.])np\s*\.\s*pi\b", repr(math.pi), s)
    s = _NP_FUNC_RE.sub(lambda m: m.group(1), s)
    return s


# sympy 支持的函数白名单
_ALLOWED_FUNCS = {
    "sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "tanh": sp.tanh,
    "Abs": sp.Abs,
}


def _sympy_to_lgf(node, var_names: Sequence[str]) -> str:
    """sympy 表达式 → LGF DSL 文本."""
    if node.is_Number:
        return repr(float(node)) if not node.is_Integer else str(int(node))
    if isinstance(node, sp.Symbol):
        return str(node)
    if isinstance(node, sp.Add):
        return " + ".join(_sympy_to_lgf(a, var_names) for a in node.args)
    if isinstance(node, sp.Mul):
        neg = False
        parts = []
        for a in node.args:
            if a == -1:
                neg = True
            else:
                parts.append(_sympy_to_lgf(a, var_names))
        body = " * ".join(parts) if parts else "1"
        return f"-({body})" if neg else body
    if isinstance(node, sp.Pow):
        base, ex = node.args
        bs = _sympy_to_lgf(base, var_names)
        if ex.is_Integer:
            k = int(ex)
            if k in (2, 3, 4):
                return f"{bs}^{k}"
            if k == -1:
                return f"inv({bs})"
            if k > 0:
                # 用 exp(k*log(base)) 表达高次幂 (LGF powi 仅支持 2..4)
                return f"exp({k} * log({bs}))"
            # k < -1: 取负数后落在 {2,3,4} 时用 inv(bs^k), 否则 inv(exp(-k*log(bs)))
            ak = -k
            if ak in (2, 3, 4):
                return f"inv({bs}^{ak})"
            return f"inv(exp({ak} * log({bs})))"
        if ex.is_Rational and ex.q == 2 and ex.p == 1:
            return f"sqrt({bs})"
        if isinstance(ex, sp.Symbol):
            # a ** C# -> exp(C# * log(a))
            return f"exp({_sympy_to_lgf(ex, var_names)} * log({bs}))"
        return f"exp(({_sympy_to_lgf(ex, var_names)}) * log({bs}))"
    if isinstance(node, sp.Function):
        fname = node.func.__name__
        arg = node.args[0] if node.args else None
        mapped = {"sqrt": "sqrt", "exp": "exp", "log": "log",
                  "sin": "sin", "cos": "cos", "tan": "tan", "tanh": "tanh"}.get(fname)
        if mapped and arg is not None:
            return f"{mapped}({_sympy_to_lgf(arg, var_names)})"
        # Abs -> sqrt(a*a)
        if fname == "Abs" and arg is not None:
            a = _sympy_to_lgf(arg, var_names)
            return f"sqrt(({a}) * ({a}))"
        raise ValueError(f"不支持函数: {fname}")
    raise ValueError(f"不支持节点: {node} ({type(node).__name__})")


def eq_line_to_lgf_dsl(eq_line: str, var_names: Sequence[str]) -> Optional[str]:
    """把单条 `EQ:` 表达式(或裸表达式)转成 LGF DSL. 失败返回 None."""
    norm = normalize_eq_expr(_EQ_PREFIX.sub("", eq_line), var_names)
    try:
        local_ns = dict(_ALLOWED_FUNCS)
        for v in var_names:
            local_ns[v] = sp.Symbol(v)
        # 未知标识符 (如 x1) 由 sympy 自动当 Symbol; 限制 eval 范围
        expr = sp.sympify(norm, locals=local_ns, evaluate=True)
    except Exception:
        return None
    # 检查: 自由符号只能是已知变量 或 C#/K# 参数
    syms = {str(s) for s in expr.free_symbols}
    params = {s for s in syms if re.fullmatch(r"[CK]\d+", s)}
    unknown = syms - set(var_names) - params
    if unknown:
        return None
    try:
        return _sympy_to_lgf(expr, var_names)
    except Exception:
        return None


def parse_eq_response(text: str, var_names: Sequence[str]) -> List[str]:
    """把整段 LLM-Feynman 风格响应转成 LGF DSL 列表 (逐行, 失败项丢弃)."""
    out: List[str] = []
    for ln in extract_eq_lines(text):
        d = eq_line_to_lgf_dsl(ln, var_names)
        if d:
            out.append(d)
    return out
