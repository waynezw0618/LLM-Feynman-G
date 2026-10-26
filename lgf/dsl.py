"""lgf.dsl — 图 ↔ DSL 文本双向转换, 供 LLM 读写.

DSL 设计原则:
- 简洁, 与 LLM-Feynman 的 spec 文本风格一致 (纯文本, 缩进可选);
- 受限语法: 仅允许模板内列出的算子, 解析失败即丢弃 (杜绝 LLM 注入非法结构);
- 支持数值系数与自由参数 (边特征) 两种缩放;
- 支持多项式定义 `P#(x) = C#*x + C#*x^2 + C#` 与调用 `P#(x2)`;
- 支持模板占位 `$1`, `$2` (模板实例化时替换).

运算符表 (op name 与 Graph.op 严格一致):
  add mul sub div exp log sin cos tanh sqrt inv neg
  powi  (通过 `x^k`, k ∈ {2,3,4})

示例:
  P1(x) = C1*x + C2*x^2 + C3
  sigma = C4 * exp( 1 / ( C5*P1(eps) + K1 ) )
"""
from __future__ import annotations
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .graph import (
    Graph, Node, Param,
    var as mk_var, const as mk_const, slot as mk_slot,
    add_with, mul_with, sub as mk_sub, div as mk_div,
    scale as mk_scale, power as mk_powi,
    expn, logn, sinn, cosn, tanhn, sqrtn, invn, negn,
    walk,
)


CALLABLE_OPS: Set[str] = {
    "add", "mul", "sub", "div",
    "exp", "log", "sin", "cos", "tanh", "sqrt", "inv", "neg",
}
UNARY_FUNCS: Set[str] = {"exp", "log", "sin", "cos", "tanh", "sqrt", "inv", "neg"}


# ============================================================================
# Tokenizer
# ============================================================================
# V4 (2026-10-12) 解析率修复: LLM 后端 (MiniMax/GLM) 高频输出三类合法数学写法
# 但旧 tokenizer 拒收, 直接拉低 G_INIT 解析率 (yw 轮 m 臂 6/30):
#   ① Python 风格幂 `x**2` → tokenize 入口归一化为 `x^2`
#   ② 科学计数 `1e-3` / `2.5E+2` → NUMBER 支持指数段
#   ③ 分号 `;` 结尾 (LLM 习惯) → 跳过
TOKEN_RE = re.compile(
    r"(?P<NUMBER>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"|(?P<IDENT>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<OP>[+\-*/^()=,])"
    r"|(?P<DOLLAR>\$)"
    r"|(?P<SEMI>;)"
    r"|(?P<WS>\s+)"
)


def tokenize(text: str) -> List[Tuple[str, str]]:
    # `**` → `^`: 连续两个乘号在合法 DSL 中不成立, 安全归一化
    text = text.replace("**", "^")
    out: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = TOKEN_RE.match(text, pos)
        if not m:
            raise SyntaxError(f"DSL 字符无法识别 at pos {pos}: {text[pos:pos+20]!r}")
        kind = m.lastgroup
        val = m.group()
        pos = m.end()
        if kind in ("WS", "SEMI"):
            continue
        out.append((kind, val))
    return out


# ============================================================================
# AST 节点 (tuple):
#   ('num', value)
#   ('var', name)
#   ('param', name)              # name 'C#' -> global, 'K#' -> case
#   ('slot', index)
#   ('powi', arg, k)             # k ∈ {2,3,4}
#   ('op', name, args, edge_specs_per_arg)   # edge_spec: None | ('num', v) | ('param', name)
# ============================================================================


# ============================================================================
# Parser
# ============================================================================
class Parser:
    def __init__(self, tokens, poly_defs=None):
        self.toks = tokens
        self.i = 0
        self.poly_defs = poly_defs or {}

    def peek(self, k=0) -> Tuple[str, str]:
        return self.toks[self.i + k] if self.i + k < len(self.toks) else ("EOF", "")

    def eat(self, kind=None, val=None):
        tok = self.peek()
        if kind is not None and tok[0] != kind:
            raise SyntaxError(f"期望 {kind}, 得到 {tok}")
        if val is not None and tok[1] != val:
            raise SyntaxError(f"期望 {val!r}, 得到 {tok[1]!r}")
        self.i += 1
        return tok

    # expr := term (('+' | '-') term)*
    def parse_expr(self):
        left = self.parse_term()
        while True:
            t = self.peek()
            if t[0] == "OP" and t[1] == "+":
                self.i += 1
                right = self.parse_term()
                left = ("op", "add", [left, right], [None, None])
            elif t[0] == "OP" and t[1] == "-":
                self.i += 1
                right = self.parse_term()
                left = ("op", "sub", [left, right], [None, None])
            else:
                break
        return left

    # term := coeffed (('*' | '/') coeffed)*
    def parse_term(self):
        left = self.parse_coeffed()
        while True:
            t = self.peek()
            if t[0] == "OP" and t[1] == "*":
                self.i += 1
                right = self.parse_coeffed()
                left = ("op", "mul", [left, right], [None, None])
            elif t[0] == "OP" and t[1] == "/":
                self.i += 1
                right = self.parse_coeffed()
                left = ("op", "div", [left, right], [None, None])
            else:
                break
        return left

    # coeffed := [ NUMBER '*' | 'C#'*' | 'K#'*' ]? unary
    # 数值前缀 → AST 中作为 ('num', v) edge_spec (后续在 _build_mul 中转为 const 节点);
    # 参数前缀 → ('param', name) edge_spec.
    def parse_coeffed(self):
        prefix_spec = None
        t = self.peek()
        if t[0] == "NUMBER":
            self.i += 1
            t2 = self.peek()
            if t2[0] == "OP" and t2[1] == "*":
                self.i += 1
                prefix_spec = ("num", float(t[1]))
            else:
                self.i -= 1
        elif t[0] == "IDENT" and re.match(r"^[CK]\d+$", t[1]):
            self.i += 1
            t2 = self.peek()
            if t2[0] == "OP" and t2[1] == "*":
                self.i += 1
                prefix_spec = ("param", t[1])
            else:
                self.i -= 1
        u = self.parse_unary()
        if prefix_spec is not None:
            return ("op", "mul", [u], [prefix_spec])
        return u

    # unary := ['-' | '+'] power
    def parse_unary(self):
        t = self.peek()
        sign = 1
        if t[0] == "OP" and t[1] == "-":
            sign = -1
            self.i += 1
        elif t[0] == "OP" and t[1] == "+":
            self.i += 1
        p = self.parse_power()
        return ("op", "neg", [p], [None]) if sign == -1 else p

    # power := atom ['^' (('+'|'-')? NUMBER)]    (V4: 支持负指数 / 0.5 / 1)
    # 注意: 本层是 AST (tuple), 不能返回图工厂 (mk_powi/sqrtn/invn) 的 Node!
    def parse_power(self):
        a = self.parse_atom()
        t = self.peek()
        if t[0] == "OP" and t[1] == "^":
            self.i += 1
            neg = False
            t2 = self.peek()
            if t2[0] == "OP" and t2[1] in ("-", "+"):
                neg = (t2[1] == "-")
                self.i += 1
            exp_tok = self.eat("NUMBER")
            kf = float(exp_tok[1])
            k = int(kf)
            if abs(kf - k) > 1e-12:
                # 非整数指数: 仅放行 0.5 (= sqrt)
                if not neg and abs(kf - 0.5) < 1e-12:
                    return ("op", "sqrt", [a], [None])
                raise SyntaxError(f"powi 指数仅支持整数或 ^0.5, 得到 ^{exp_tok[1]}")
            if k == 1 and not neg:
                return a
            if not neg and k in (2, 3, 4):
                return ("powi", a, k)
            if neg and 1 <= k <= 4:
                # x^-k → inv(x^k); k=1 时直接 inv(x)
                inner = a if k == 1 else ("powi", a, k)
                return ("op", "inv", [inner], [None])
            raise SyntaxError(
                f"powi 指数仅支持 [-4,-1]、[2,4] 及 0.5, 得到 ^{exp_tok[1]}")
        return a

    # atom := NUMBER | IDENT | '$' INT | '(' expr ')'
    def parse_atom(self):
        t = self.peek()
        if t[0] == "NUMBER":
            self.i += 1
            return ("num", float(t[1]))
        if t[0] == "DOLLAR":
            self.i += 1
            idx_tok = self.eat("NUMBER")
            return ("slot", int(idx_tok[1]))
        if t[0] == "IDENT":
            name = t[1]
            # 函数调用?
            if (self.i + 1 < len(self.toks)
                    and self.toks[self.i + 1][0] == "OP"
                    and self.toks[self.i + 1][1] == "("):
                if name not in CALLABLE_OPS and not re.match(r"^P\d+$", name):
                    raise SyntaxError(f"未知函数名: {name}")
                self.i += 1  # consume IDENT
                self.eat("OP", "(")
                args = []
                if not (self.peek()[0] == "OP" and self.peek()[1] == ")"):
                    args.append(self.parse_expr())
                    while self.peek()[0] == "OP" and self.peek()[1] == ",":
                        self.i += 1
                        args.append(self.parse_expr())
                self.eat("OP", ")")
                if name in CALLABLE_OPS:
                    return ("op", name, args, [None] * len(args))
                # 多项式引用
                pidx = int(name[1:])
                pvar, _proot = self.poly_defs.get(pidx, (None, None))
                if pvar is None:
                    raise SyntaxError(f"未定义多项式 {name}")
                if len(args) != 1:
                    raise SyntaxError(f"P# 引用须恰好 1 个参数, 得到 {len(args)}")
                return ("polyref", pidx, args[0])
            self.i += 1
            if re.match(r"^[CK]\d+$", name):
                return ("param", name)
            return ("var", name)
        if t[0] == "OP" and t[1] == "(":
            self.i += 1
            e = self.parse_expr()
            self.eat("OP", ")")
            return e
        raise SyntaxError(f"无法解析: {t}")


# ============================================================================
# Poly 头解析:  P#(var) = ...   (支持加法, 每项: Ci*var^k 或 number*var^k 或 var^k)
# ============================================================================
POLY_DEF_RE = re.compile(r"^\s*P(\d+)\s*\(\s*([A-Za-z_]\w*)\s*\)\s*=\s*(.+?)\s*$")


def _parse_poly_def_line(line: str, poly_defs: Dict[int, Tuple[str, Node]]):
    m = POLY_DEF_RE.match(line)
    if not m:
        return False
    idx = int(m.group(1))
    var_name = m.group(2)
    body = m.group(3)
    parser = Parser(tokenize(body))
    ast = parser.parse_expr()
    root = _ast_to_graph(ast, [var_name])
    poly_defs[idx] = (var_name, root)
    return True


def _ast_to_graph(ast: Tuple, var_names: List[str]) -> Node:
    kind = ast[0]
    if kind == "num":
        return mk_const(ast[1])
    if kind == "var":
        try:
            idx = var_names.index(ast[1])
        except ValueError:
            raise SyntaxError(f"未知变量 {ast[1]!r}; 已知: {var_names}")
        return mk_var(idx)
    if kind == "slot":
        return mk_slot(ast[1])
    if kind == "param":
        name = ast[1]
        kind_ = "case" if name.startswith("K") else "global"
        p = Param(name=name, kind=kind_, init=1.0)
        return mk_scale(mk_const(1.0), p)
    if kind == "powi":
        return mk_powi(_ast_to_graph(ast[1], var_names), ast[2])
    if kind == "polyref":
        raise SyntaxError("polyref 必须在 _ast_to_graph 外展开")
    if kind == "op":
        op = ast[1]
        args = [_ast_to_graph(a, var_names) for a in ast[2]]
        edge_specs = ast[3]
        if op == "mul":
            return _build_mul(args, edge_specs)
        if op == "add":
            return _build_add(args, edge_specs)
        if op == "sub":
            return mk_sub(args[0], args[1])
        if op == "div":
            return mk_div(args[0], args[1])
        if op == "exp": return expn(args[0])
        if op == "log": return logn(args[0])
        if op == "sin": return sinn(args[0])
        if op == "cos": return cosn(args[0])
        if op == "tanh": return tanhn(args[0])
        if op == "sqrt": return sqrtn(args[0])
        if op == "inv": return invn(args[0])
        if op == "neg": return negn(args[0])
        if op == "powi":
            return mk_powi(args[0], ast[4])
        raise SyntaxError(f"不支持 op: {op}")
    raise SyntaxError(f"未知 AST 节点: {ast}")


def _espec_to_param(es):
    if es is None:
        return None
    if es[0] == "num":
        return Param(name=f"_c_{abs(es[1])}", kind="global", init=es[1])
    if es[0] == "param":
        kind_ = "case" if es[1].startswith("K") else "global"
        return Param(name=es[1], kind=kind_, init=1.0)
    raise SyntaxError(f"bad edge spec {es}")


def _build_add(args, edge_specs) -> Node:
    if len(args) == 1 and (not edge_specs or edge_specs[0] is None):
        return args[0]
    pairs = [(a, _espec_to_param(es)) for a, es in zip(args, edge_specs or [None] * len(args))]
    return add_with(*pairs)


def _build_mul(args, edge_specs) -> Node:
    """构造 mul 节点, 处理:
       - 'num' edge_spec: 转 const 节点参与 (不出现在一个点 param 上);
       - 'param' edge_spec: 视为子项的 edge param;

    V4 (2026-10-12) theta-对齐修复: 不再把 bare-param 子节点 (scale(const(1), p))
    抽出合并到首子项 — 该合并会改变树结构 → walk 序变 → _normalize_params
    重命名错位 → 记录的 theta 在消费端 (parse_dsl 重编译) 与参数语义脱钩
    (实测: lgf_go/m 记录 loss 0.36, 消费端 4043, 两个系数互换).
    现在原样保留 bare-param 子节点; `C# * (1)` 因子往返稳定
    (mul child scale(const(1),p) 打印 `p * (1)`, 重解析同形).
    代价: `C1 * C2 * x` 保留 2 个 (乘积简并) 参数而非合并为 1 —
    对拟合无害 (L-BFGS 可解简并方向), 换取往返稳定性.
    """
    # 1) 处理 edge_spec: 'num' → const 子节点; 'param' → 子项 edge param
    children: List[Tuple[Node, Optional[Param]]] = []
    for a, es in zip(args, edge_specs or [None] * len(args)):
        if es is not None and es[0] == "num":
            children.append((mk_const(es[1]), None))
        elif es is not None and es[0] == "param":
            children.append((a, _espec_to_param(es)))
        else:
            children.append((a, None))
    return mul_with(*children)


# ============================================================================
# 公开入口: DSL → Graph
# ============================================================================
def parse_dsl(text: str, var_names: Sequence[str]) -> Graph:
    """解析 DSL 文本为 Graph. 支持 P#(x)=... 多项式定义头, 以及可选的 `IDENT = expr` 前缀."""
    poly_defs: Dict[int, Tuple[str, Node]] = {}
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    expr_lines = []
    for ln in lines:
        if _parse_poly_def_line(ln, poly_defs):
            continue
        expr_lines.append(ln)
    if not expr_lines:
        raise SyntaxError("DSL 中没有可解析的表达式")
    body = " ".join(expr_lines)
    # 去掉可选的 `IDENT = expr` 前缀 (允许 LLM/人类书写签名)
    m_lead = re.match(r"^\s*[A-Za-z_]\w*\s*=\s*(.+)$", body)
    if m_lead:
        body = m_lead.group(1)
    body = _expand_poly_refs(body, poly_defs)
    toks = tokenize(body)
    parser = Parser(toks, poly_defs=poly_defs)
    ast = parser.parse_expr()
    if parser.i < len(parser.toks):
        raise SyntaxError(f"未消耗完 token: {parser.toks[parser.i:]}")
    root = _ast_to_graph(ast, list(var_names))
    return Graph(root, list(var_names))


def _expand_poly_refs(body: str, poly_defs: Dict[int, Tuple[str, Node]]) -> str:
    """`Pk(var)` -> 对应多项式 (基于该 var) 的 inline 表达式. 仅用于无嵌套多变量的情形."""
    if not poly_defs:
        return body
    pat = re.compile(r"P(\d+)\s*\(\s*([A-Za-z_]\w*)\s*\)")
    def repl(m):
        pidx = int(m.group(1))
        vname = m.group(2)
        if pidx not in poly_defs:
            return m.group(0)
        orig_var, proot = poly_defs[pidx]
        if orig_var != vname:
            raise SyntaxError(f"polyref 变量 {vname!r} 与定义 {orig_var!r} 不一致")
        return "(" + Graph(proot, [vname]).to_dsl(with_poly_defs=False) + ")"
    return pat.sub(repl, body)


# ============================================================================
# Graph → DSL
# ============================================================================
def graph_to_dsl(g: Graph, with_poly_defs: bool = True) -> str:
    poly_defs: Dict[int, Tuple[str, List[Tuple[int, Node]]]] = {}
    counter = [0]

    def _print(node: Node) -> str:
        # 多项式检测 (顶层 add 节点)
        if with_poly_defs and node.op == "add" and len(node.children) >= 2:
            terms = []
            var_id = None
            ok = True
            for ch, edge_p in node.children:
                m = _match_monomial(ch)
                if not m:
                    ok = False; break
                vi, k, coef_node = m
                if var_id is None:
                    var_id = vi
                elif var_id != vi:
                    ok = False; break
                terms.append((k, coef_node, vi))
            if ok and var_id is not None:
                counter[0] += 1
                idx = counter[0]
                poly_defs[idx] = (g.var_names[var_id], terms)
                return f"P{idx}({g.var_names[var_id]})"
        return _print_node(node)

    def _print_node(node: Node) -> str:
        op = node.op
        if op == "var":
            return g.var_names[node.attrs["index"]]
        if op == "const":
            v = node.attrs["const"]
            return str(int(v)) if v == int(v) else repr(v)
        if op == "slot":
            return f"${node.attrs['index']}"
        if op == "powi":
            return f"{_print(node.children[0][0])}^{node.attrs['power']}"
        if op == "neg":
            return f"-({_print(node.children[0][0])})"
        if op == "add":
            parts = []
            for ch, ep in node.children:
                cs = _print(ch)
                if ep is None:
                    parts.append(cs)
                else:
                    parts.append(f"{ep.name} * ({cs})")
            return " + ".join(parts)
        if op == "mul":
            parts = []
            for ch, ep in node.children:
                cs = _print(ch)
                if ep is None:
                    parts.append(f"({cs})")
                else:
                    parts.append(f"{ep.name} * ({cs})")
            return " * ".join(parts)
        if op == "sub":
            return f"({_print(node.children[0][0])}) - ({_print(node.children[1][0])})"
        if op == "div":
            return f"({_print(node.children[0][0])}) / ({_print(node.children[1][0])})"
        if op in UNARY_FUNCS:
            return f"{op}({_print(node.children[0][0])})"
        raise ValueError(f"unknown op: {op}")

    body = _print(g.root)
    if with_poly_defs and poly_defs:
        defs = []
        for idx, (vname, terms) in poly_defs.items():
            rt = []
            # 只按幂次 k 排序: k 相等时元组比较会落到 Node 对象上 (不可比, TypeError)
            for k, coef_node, _vi in sorted(terms, key=lambda t: t[0]):
                pname = _coef_print(coef_node)
                if k == 0:
                    rt.append(pname)
                elif k == 1:
                    rt.append(f"{pname} * {vname}")
                else:
                    rt.append(f"{pname} * {vname}^{k}")
            defs.append(f"P{idx}({vname}) = " + " + ".join(rt))
        return "\n".join(defs) + "\n" + body
    return body


def _coef_print(node: Node) -> str:
    """打印单项系数: 若为 scale(const(1), p) → 'C#'; 若为 const → '数值'; 否则原文."""
    if node.op == "const":
        v = node.attrs["const"]
        return str(int(v)) if v == int(v) else repr(v)
    if (node.op == "mul" and len(node.children) == 1
            and node.children[0][1] is not None
            and node.children[0][0].op == "const"
            and node.children[0][0].attrs["const"] == 1.0):
        return node.children[0][1].name
    # 退化: 打印子树
    from .graph import Graph as _G
    return _G(node, []).to_dsl(with_poly_defs=False)


def _match_monomial(node: Node):
    """判断 node 是否是 `coeff * var^k` 形式 (单变量, 系数为 scale(const(1), p) 或 const)."""
    # 直接是 var (k=1)
    if node.op == "var":
        return (node.attrs["index"], 1, mk_const(1.0))
    # 直接是 powi(var, k)
    if node.op == "powi" and node.children[0][0].op == "var":
        return (node.children[0][0].attrs["index"], int(node.attrs["power"]), mk_const(1.0))
    # mul 情形
    if node.op == "mul":
        var_seen = []   # (k)
        other = []      # other children (coefficient pieces)
        for ch, ep in node.children:
            if ep is None and ch.op == "var":
                var_seen.append(1)
            elif ep is None and ch.op == "powi" and ch.children[0][0].op == "var":
                var_seen.append(int(ch.attrs["power"]))
            else:
                other.append((ch, ep))
        if len(var_seen) == 0 or len(var_seen) > 1:
            return None
        k = var_seen[0]
        # var index
        if node.children[0][0].op == "var" and node.children[0][1] is None:
            var_i = node.children[0][0].attrs["index"]
        else:
            # 找 var 子节点
            for ch, ep in node.children:
                if ep is None and ch.op == "var":
                    var_i = ch.attrs["index"]; break
                if ep is None and ch.op == "powi" and ch.children[0][0].op == "var":
                    var_i = ch.children[0][0].attrs["index"]; break
        # 系数
        if not other:
            coef_node = mk_const(1.0)
        elif len(other) == 1:
            ch, ep = other[0]
            if ep is None and ch.op == "const":
                coef_node = ch
            elif ep is not None and ch.op == "const" and ch.attrs["const"] == 1.0:
                coef_node = node  # 保留 scale(const(1), p)
            else:
                return None
        else:
            return None
        return (var_i, k, coef_node)
    # 1 孩子 mul with edge param: scale(var^k, p)
    if node.op == "mul" and len(node.children) == 1 and node.children[0][1] is not None:
        sub, p = node.children[0]
        if sub.op == "var":
            return (sub.attrs["index"], 1, node)
        if sub.op == "powi" and sub.children[0][0].op == "var":
            return (sub.children[0][0].attrs["index"], int(sub.attrs["power"]), node)
    return None