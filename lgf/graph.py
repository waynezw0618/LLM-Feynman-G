"""lgf.graph — 统一表示层: 方程 = 带边特征的有向无环图 (root DAG).

设计要点:
- 节点 op ∈ {var, const, slot} ∪ nary(add,mul) ∪ binary(sub,div)
            ∪ unary(exp,log,sin,cos,tanh,sqrt,inv,neg) ∪ powi.
- 每条边可携带一个 Param (边特征 f_e): 自由参数或常数, 作用于子节点取值.
- Param.kind ∈ {'global', 'case'}: global 在多案例间共享, case 每案例独立拟合.
- 表示对齐 GraphED (边特征即乘法系数), 可与 DSL 文本双向转换供 LLM 读写.

约定:
- 1 孩子且无 edge_param 的 mul 节点在 simplify() 中被压缩为子节点;
  带 edge_param 的 1 孩子 mul 即"缩放节点" (= f_e * x).
- add 至少 2 孩子 (1 孩子无意义).
- 'inv' = 1/x, 'neg' = -x, 'sqrt' = sqrt(x), 'powi' = x**int.
- 数值非有限 (log 负/exp 溢出) 由损失函数加罚值屏蔽, 不在图层做硬约束.
"""
from __future__ import annotations
import copy
import math
import random
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import sympy as sp

import numpy as np


# ===== 算子表 =====
NARY_OPS = ("add", "mul")
BINARY_OPS = ("sub", "div")
UNARY_OPS = ("exp", "log", "sin", "cos", "tanh", "sqrt", "inv", "neg")
LEAF_OPS = ("var", "const", "slot")
POWI_POWERS = (2, 3, 4)
ALL_OPS = set(NARY_OPS + BINARY_OPS + UNARY_OPS + ("powi",) + LEAF_OPS)


# ===== 边特征 (Param) =====
@dataclass
class Param:
    """边特征 f_e. 作用于 child 取值: f_e * child_value."""
    name: str                # 唯一名, 归一化为 C1..Cm / K1..Kp
    kind: str = "global"     # 'global' | 'case'
    init: float = 1.0

    def clone(self, new_name: Optional[str] = None) -> "Param":
        return Param(name=new_name or self.name, kind=self.kind, init=self.init)


# ===== 节点 =====
class Node:
    __slots__ = ("op", "children", "attrs")

    def __init__(self, op: str,
                 children: Optional[List[Tuple["Node", Optional[Param]]]] = None,
                 attrs: Optional[Dict[str, Any]] = None):
        assert op in ALL_OPS, f"unknown op: {op}"
        self.op = op
        # 非叶节点: (child, edge_param_or_None) 列表
        # 叶节点: children 为空列表
        self.children: List[Tuple[Node, Optional[Param]]] = children or []
        self.attrs: Dict[str, Any] = attrs or {}

    def clone(self) -> "Node":
        return Node(
            op=self.op,
            children=[(c.clone(), None if p is None else p.clone()) for c, p in self.children],
            attrs=dict(self.attrs),
        )

    def depth(self) -> int:
        if not self.children:
            return 1
        return 1 + max(c.depth() for c, _ in self.children)

    def size(self) -> int:
        return 1 + sum(c.size() for c, _ in self.children)

    def n_edges(self) -> int:
        return len(self.children) + sum(c.n_edges() for c, _ in self.children)


# ===== 工厂函数 (便于在模板/GP 中构造图) =====
def var(i: int) -> Node:
    return Node("var", attrs={"index": int(i)})

def const(v: float) -> Node:
    return Node("const", attrs={"const": float(v)})

def slot(i: int) -> Node:
    return Node("slot", attrs={"index": int(i)})

def add(*nodes) -> Node:
    assert len(nodes) >= 2, "add 至少 2 个节点"
    return Node("add", children=[(n, None) for n in nodes])

def mul(*nodes) -> Node:
    assert len(nodes) >= 1, "mul 至少 1 个节点"
    return Node("mul", children=[(n, None) for n in nodes])

def add_with(*pairs) -> Node:
    """接受 (Node, Optional[Param]) 元组序列, 保留边特征."""
    return Node("add", children=list(pairs))

def mul_with(*pairs) -> Node:
    return Node("mul", children=list(pairs))

def sub(a: Node, b: Node) -> Node:
    return Node("sub", children=[(a, None), (b, None)])

def div(a: Node, b: Node) -> Node:
    return Node("div", children=[(a, None), (b, None)])

def scale(child: Node, p: Param) -> Node:
    """缩放节点 = mul(children=[(child, p)])."""
    return Node("mul", children=[(child, p)])

def power(child: Node, k: int) -> Node:
    assert k in POWI_POWERS, f"powi power {k} not supported; use mul(x,x) or sqrt for 2"
    return Node("powi", children=[(child, None)], attrs={"power": int(k)})

def _unary(name: str, x: Node) -> Node:
    return Node(name, children=[(x, None)])

def expn(x): return _unary("exp", x)
def logn(x): return _unary("log", x)
def sinn(x): return _unary("sin", x)
def cosn(x): return _unary("cos", x)
def tanhn(x): return _unary("tanh", x)
def sqrtn(x): return _unary("sqrt", x)
def invn(x): return _unary("inv", x)
def negn(x): return _unary("neg", x)


# ===== 工具: 遍历 / 路径操作 =====
def walk(root: Node):
    """前序遍历: yield (path: tuple[int], node).

    V4 (2026-10-12) 根因修复: 旧实现正序 push 子节点到 LIFO 栈 → pop 序
    为"根 → 末子节点 → …"的逆序 DFS, 与先序遍历不符. 后果: _collect_params
    先遇到末子树的参数 → _normalize_params 重编号与 DSL 源文本语义错位
    (实测 `C1 * x1 + C2 * x2^2` 解析后 expr = C1*x2**2 + C2*x1,
    test_paramfit_linear_recovery 语义断言失败的真因). 修为逆序 push,
    pop 恢复子节点升序 — 标准 preorder.
    """
    stack = [((), root)]
    while stack:
        path, node = stack.pop()
        yield path, node
        for i in range(len(node.children) - 1, -1, -1):
            ch, _ = node.children[i]
            stack.append((path + (i,), ch))


def node_at(root: Node, path: Tuple[int, ...]) -> Node:
    n = root
    for i in path:
        n = n.children[i][0]
    return n


def set_node_at(root: Node, path: Tuple[int, ...], new_node: Node) -> Node:
    """in-place 替换并返回新 root (若替换 root 自身)."""
    if not path:
        return new_node
    parent = node_at(root, path[:-1])
    idx = path[-1]
    parent.children[idx] = (new_node, parent.children[idx][1])
    return root


# ===== 主类: Graph =====
class Graph:
    def __init__(self, root: Node, var_names: Sequence[str], name: str = "G"):
        self.root = root
        self.var_names: List[str] = list(var_names)
        self.name = name
        self.loss: float = float("inf")
        self.theta = None   # 完整 theta 向量
        self.metrics: Dict[str, float] = {}
        self._simplify()
        self._normalize_params()
        self._validate()

    # ----- 构造与克隆 -----
    def clone(self, new_name: Optional[str] = None) -> "Graph":
        g = Graph(self.root.clone(), self.var_names, name=new_name or self.name)
        g.loss = self.loss
        g.metrics = dict(self.metrics)
        # V3-0 根因修复 (2026-10-08): theta 不拷贝 → best_g.theta=None →
        # attribution.analyze 内 theta[:P_g] 抛 TypeError 被 pipeline except:pass
        # 吞掉 → M_GUIDE 在 P1c/P1d 全部 run 从未执行 (6/6 n_pop 恒 40 的真因)
        if self.theta is not None:
            g.theta = np.asarray(self.theta, dtype=float).copy()
        return g

    # ----- 复杂度统计 -----
    def count_nodes(self) -> int:
        return self.root.size()

    def count_edges(self) -> int:
        return self.root.n_edges()

    def count_params(self) -> int:
        n = 0
        for _, node in walk(self.root):
            for _, p in node.children:
                if p is not None:
                    n += 1
        return n

    def depth(self) -> int:
        return self.root.depth()

    def complexity(self) -> Dict[str, int]:
        return dict(nodes=self.count_nodes(),
                    edges=self.count_edges(),
                    params=self.count_params(),
                    depth=self.depth())

    # ----- 参数归一化 (C1..Cm, K1..Kp) -----
    def _collect_params(self) -> List[Param]:
        seen: Dict[Tuple[str, int], Param] = {}
        order: List[Param] = []
        for _, node in walk(self.root):
            for _, p in node.children:
                if p is None:
                    continue
                key = (p.name, id(p))
                if key not in seen:
                    seen[key] = p
                    order.append(p)
        return order

    def _normalize_params(self):
        """V4 (2026-10-12) 保名修复: 不再按遍历序强制重编号 C1..Cn.

        旧实现把参数一律按 walk 序重命名为 C1, C2, ... — 当 DSL 源文本的
        参数出现顺序与编号不一致时 (典型: eq_compat 走 sympy, Add.args 为
        规范序, 产出 `C2*sqrt(..) + C1*v1*v2^2`), 重编号会把先遍历到的 C2
        改名为 C1, 与源文本语义错位 (test_numeric_equivalence_original_vs_lgf
        max|err|=6.63 的真因; 亦即 memory 挂账的 "parse 保名" 待办).

        保名规则:
        - 现有名字已是本 kind 的规范名 (global→C#, case→K#) 且未冲突 → 保留
        - 其余 (GP 新建 _u*/_k*、重名、跨 kind 错标) → 分配空闲编号

        theta 对齐不受影响且更稳: params() 统一按名字编号排序 (见其
        docstring), 拟合/评估/记录/消费端共享同一顺序; 保名 + 排序使
        to_dsl 往返 (graph→DSL→graph) 的 theta 对齐对任意树形稳定.
        """
        order = self._collect_params()
        used: Dict[str, set] = {"global": set(), "case": set()}
        pending: List[Param] = []
        for p in order:
            tag = "K" if p.kind == "case" else "C"
            if p.name and re.fullmatch(rf"{tag}\d+", p.name) \
                    and p.name not in used[p.kind]:
                used[p.kind].add(p.name)
            else:
                pending.append(p)
        for p in pending:
            tag = "K" if p.kind == "case" else "C"
            i = 1
            while f"{tag}{i}" in used[p.kind]:
                i += 1
            p.name = f"{tag}{i}"
            used[p.kind].add(p.name)

    # ----- 简化 (canonicalization) -----
    def _simplify(self):
        """去除恒等节点 / 单孩子 mul 无 param."""
        # 递归: 处理子节点, 再处理自身
        def fix(n: Node) -> Node:
            for i, (ch, _) in enumerate(n.children):
                n.children[i] = (fix(ch), n.children[i][1])
            # mul 1 child & no param => child
            if n.op == "mul" and len(n.children) == 1 and n.children[0][1] is None:
                return n.children[0][0]
            # add 1 child & no param => child
            if n.op == "add" and len(n.children) == 1 and n.children[0][1] is None:
                return n.children[0][0]
            return n
        self.root = fix(self.root)

    def _validate(self):
        for path, n in walk(self.root):
            if n.op in NARY_OPS:
                if n.op == "add" and len(n.children) < 2:
                    raise ValueError(f"add must have >=2 children at {path}")
                if n.op == "mul" and len(n.children) < 1:
                    raise ValueError(f"mul must have >=1 child at {path}")
            elif n.op in BINARY_OPS and len(n.children) != 2:
                raise ValueError(f"{n.op} must have 2 children at {path}")
            elif n.op in UNARY_OPS and len(n.children) != 1:
                raise ValueError(f"{n.op} must have 1 child at {path}")
            elif n.op == "powi" and (len(n.children) != 1 or n.attrs.get("power") not in POWI_POWERS):
                raise ValueError(f"powi malformed at {path}: {n.attrs}")
            elif n.op == "var" and (not n.children) is False or (n.op == "var" and "index" not in n.attrs):
                pass
            elif n.op == "const" and "const" not in n.attrs:
                raise ValueError(f"const node missing value at {path}")
            elif n.op == "slot" and "index" not in n.attrs:
                raise ValueError(f"slot node missing index at {path}")

    # ----- 参数布局 (供 paramfit 使用) -----
    def params(self) -> List[Param]:
        """按 (kind, 编号) 排序返回参数: globals(C#) 在前, cases(K#) 在后, 各自编号升序.

        V4 (2026-10-12) 与 _normalize_params 保名修复配套: theta↔参数的对应
        统一为"按名字编号排序", 与消费端 `sorted(C# names) ↔ theta` 的配对
        约定一致 (scripts/rc_asm_*_figs.py textual_eval). 不再依赖遍历序 —
        DSL 往返/重解析后 theta 对齐对任意树形稳定 (同一名字集 ⇒ 同一顺序).
        """
        globals_, cases_ = [], []
        seen = set()
        for _, node in walk(self.root):
            for _, p in node.children:
                if p is None or id(p) in seen:
                    continue
                seen.add(id(p))
                (cases_ if p.kind == "case" else globals_).append(p)

        def _numkey(p) -> int:
            m = re.fullmatch(r"[CK](\d+)", p.name or "")
            return int(m.group(1)) if m else 10 ** 9

        globals_.sort(key=_numkey)
        cases_.sort(key=_numkey)
        return globals_ + cases_

    def global_names(self) -> List[str]:
        return [p.name for p in self.params() if p.kind == "global"]

    def case_names(self) -> List[str]:
        return [p.name for p in self.params() if p.kind == "case"]

    def theta_dim(self, n_cases: int) -> int:
        n_g = len(self.global_names()); n_c = len(self.case_names())
        return n_g + n_c * n_cases

    def set_param_inits(self, values: Dict[str, float]):
        for p in self.params():
            if p.name in values:
                p.init = float(values[p.name])

    # ----- 替换 (用于 poly 与模板实例化) -----
    def substitute_var(self, var_index: int, replacement: Node) -> "Graph":
        """将 var(下标=var_index) 替换为 replacement 的克隆."""
        def fix(n: Node) -> Node:
            chs = []
            for ch, p in n.children:
                ch2 = fix(ch)
                if ch.op == "var" and ch.attrs.get("index") == var_index and ch is n.children[len(chs)][0]:
                    # need fresh clone of replacement each occurrence
                    from copy import deepcopy
                    chs.append((deepcopy(replacement), p))
                else:
                    chs.append((ch2, p))
            return Node(n.op, children=chs, attrs=dict(n.attrs))
        # 上面写法会在循环内访问未完成 chs; 改写为递归
        def fix2(n: Node) -> Node:
            new_children = []
            for ch, p in n.children:
                if ch.op == "var" and ch.attrs.get("index") == var_index:
                    new_children.append((copy.deepcopy(replacement), p))
                else:
                    new_children.append((fix2(ch), p))
            return Node(n.op, children=new_children, attrs=dict(n.attrs))
        new_root = fix2(self.root)
        return Graph(new_root, self.var_names, name=self.name + "_subst")

    def substitute_slot(self, slot_index: int, replacement: Node) -> "Graph":
        def fix2(n: Node) -> Node:
            new_children = []
            for ch, p in n.children:
                if ch.op == "slot" and ch.attrs.get("index") == slot_index:
                    new_children.append((copy.deepcopy(replacement), p))
                else:
                    new_children.append((fix2(ch), p))
            return Node(n.op, children=new_children, attrs=dict(n.attrs))
        new_root = fix2(self.root)
        return Graph(new_root, self.var_names, name=self.name + "_inst")

    # ----- sympy 表达式 (供编译/打印/等价检验) -----
    def to_sympy(self, var_syms: Optional[Dict[str, sp.Symbol]] = None,
                 param_syms: Optional[Dict[str, sp.Symbol]] = None):
        if var_syms is None:
            var_syms = {v: sp.Symbol(v, real=True) for v in self.var_names}
        if param_syms is None:
            param_syms = {p.name: sp.Symbol(p.name, real=True) for p in self.params()}

        def E(n: Node):
            if n.op == "var":
                return var_syms[self.var_names[n.attrs["index"]]]
            if n.op == "const":
                return sp.Float(n.attrs["const"])
            if n.op == "slot":
                return sp.Symbol(f"SLOT{n.attrs['index']}")
            args = []
            for ch, pp in n.children:
                ce = E(ch)
                if pp is not None:
                    ce = ce * param_syms[pp.name]
                args.append(ce)
            op = n.op
            if op == "add": return sp.Add(*args)
            if op == "mul": return sp.Mul(*args)
            if op == "sub": return args[0] - args[1]
            if op == "div": return args[0] / args[1]
            if op == "powi": return args[0] ** int(n.attrs["power"])
            if op == "inv": return sp.Integer(1) / args[0]
            if op == "neg": return -args[0]
            if op == "sqrt": return sp.sqrt(args[0])
            if op == "exp": return sp.exp(args[0])
            if op == "log": return sp.log(args[0])
            if op == "sin": return sp.sin(args[0])
            if op == "cos": return sp.cos(args[0])
            if op == "tanh": return sp.tanh(args[0])
            raise ValueError(f"unknown op: {op}")
        return E(self.root)

    def pretty(self) -> str:
        expr = self.to_sympy()
        return sp.sstr(expr)

    # ----- DSL (通过 dsl 模块实现) -----
    def to_dsl(self, with_poly_defs: bool = True) -> str:
        from . import dsl
        return dsl.graph_to_dsl(self, with_poly_defs=with_poly_defs)

    # ----- 用于缓存的结构签名 -----
    def signature(self) -> str:
        # 用 sp.srepr 对方程做参量化脱敏 (以符号代替具体值)
        expr = self.to_sympy()
        # 参量化签名: sympy 表达式
        return sp.srepr(expr) + "||" + ",".join(self.var_names)


# ===== 自由创建 Param 的辅助 =====
def global_param(name: Optional[str] = None, init: float = 1.0) -> Param:
    if name is None:
        name = f"_u{random.randint(0, 1_000_000)}"
    return Param(name=name, kind="global", init=init)

def case_param(name: Optional[str] = None, init: float = 1.0) -> Param:
    if name is None:
        name = f"_k{random.randint(0, 1_000_000)}"
    return Param(name=name, kind="case", init=init)


# ===== 复杂度检查 =====
def is_within_complexity(g: Graph, max_edges: int, max_params: int, max_depth: int) -> bool:
    c = g.complexity()
    return c["edges"] <= max_edges and c["params"] <= max_params and c["depth"] <= max_depth
