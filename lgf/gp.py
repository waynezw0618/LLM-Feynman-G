"""lgf.gp — 图遗传算子 (crossover / mutation / selection / extinction).

算子以 Graph 为单位操作, 在 (max_edges, max_params, max_depth) 约束内随机/启发式采样.
"""
from __future__ import annotations
import copy
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .graph import (
    Graph, Node, Param, var, const, add, mul, sub, div, scale, power,
    expn, logn, sinn, cosn, tanhn, sqrtn, invn, negn,
    global_param as gpar, case_param as kpar,
    walk, node_at, set_node_at,
)
from .templates import TEMPLATE_REGISTRY, list_templates
from .priors import PriorConstraints


# ===== 辅助: 所有节点路径列表 =====
def all_paths(g: Graph) -> List[Tuple[int, ...]]:
    return [p for p, _ in walk(g.root)]


def _indices_ok(g: Graph, var_names: List[str]) -> bool:
    """所有 var 节点索引必须落在 var_names 范围内 (模板图自带短 var_names, 须校验)."""
    n = len(var_names)
    for _, node in walk(g.root):
        if node.op == "var":
            if not (0 <= int(node.attrs["index"]) < n):
                return False
    return True


def _rewrap(g: Graph, var_names: List[str], name: str = "f") -> Graph:
    """把模板图 (自带短 var_names) 重包装为数据集的完整 var_names."""
    return Graph(g.root.clone(), list(var_names), name=name)


def subtree_at(g: Graph, path: Tuple[int, ...]) -> Node:
    return node_at(g.root, path)


def replace_subtree(g: Graph, path: Tuple[int, ...], new_sub: Node) -> Graph:
    """返回新 Graph (深克隆)."""
    new_root = set_node_at(g.root.clone(), path, new_sub)
    return Graph(new_root, g.var_names, name=g.name)


def random_param(rng, kind: str = "global", init: float = 1.0) -> Param:
    p = Param(name=f"_{kind[0]}{rng.integers(0, 1_000_000)}", kind=kind,
             init=float(init))
    return p


def random_terminal(rng, var_names: List[str]) -> Node:
    """随机叶节点: 变量或常量."""
    if rng.random() < 0.5 and var_names:
        return var(rng.integers(0, len(var_names)))
    return const(float(rng.uniform(-2.0, 2.0)))


def random_unary(rng, var_names: List[str], depth: int, max_depth: int) -> Node:
    if depth >= max_depth:
        return random_terminal(rng, var_names)
    op = rng.choice(["exp", "log", "sin", "cos", "tanh", "sqrt", "inv", "neg",
                     "scale_var", "term"])
    if op == "term":
        return random_terminal(rng, var_names)
    if op == "scale_var":
        child = random_unary(rng, var_names, depth + 1, max_depth)
        return scale(child, random_param(rng, "global", rng.uniform(0.1, 2.0)))
    child = random_unary(rng, var_names, depth + 1, max_depth)
    if op == "exp": return expn(child)
    if op == "log": return logn(child)
    if op == "sin": return sinn(child)
    if op == "cos": return cosn(child)
    if op == "tanh": return tanhn(child)
    if op == "sqrt": return sqrtn(child)
    if op == "inv": return invn(child)
    if op == "neg": return negn(child)
    return child


def random_binary(rng, var_names: List[str], depth: int, max_depth: int) -> Node:
    if depth >= max_depth:
        a = random_unary(rng, var_names, depth + 1, max_depth)
        b = random_unary(rng, var_names, depth + 1, max_depth)
    else:
        a = random_unary(rng, var_names, depth + 1, max_depth)
        b = random_unary(rng, var_names, depth + 1, max_depth)
    op = rng.choice(["add", "sub", "mul", "div"])
    if op == "add": return add(a, b)
    if op == "sub": return sub(a, b)
    if op == "mul": return mul(a, b)
    if op == "div": return div(a, b)
    return a


def random_graph(rng, var_names: List[str],
                 max_edges: int = 14, max_params: int = 6, max_depth: int = 5,
                 constraints: Optional[PriorConstraints] = None) -> Graph:
    """采样一个随机图 (优先从模板库; 否则随机构建)."""
    var_names = list(var_names)
    templates = list_templates()
    name = "f"
    if rng.random() < 0.6:
        tname = rng.choice(templates)
        try:
            if tname == "mul2" and len(var_names) >= 2:
                g, _ = TEMPLATE_REGISTRY[tname](rng.integers(0, len(var_names)),
                                                  rng.integers(0, len(var_names)),
                                                  name=name)
            else:
                g, _ = TEMPLATE_REGISTRY[tname](rng.integers(0, len(var_names)), name=name)
            # 重包装为数据集 var_names 并校验
            g = _rewrap(g, var_names, name)
            c = g.complexity()
            if (c["edges"] <= max_edges and c["params"] <= max_params
                    and c["depth"] <= max_depth and _indices_ok(g, var_names)):
                # 注入小扰动
                for p in g.params():
                    p.init = float(rng.uniform(0.1, 2.0) * (1.0 if rng.random() > 0.3 else -1.0))
                return g
        except Exception:
            pass
    # 退回随机构建
    for _attempt in range(8):
        op = rng.choice(["add", "mul", "exp", "log", "sin"])
        if op == "mul":
            g = Graph(mul(random_unary(rng, var_names, 1, max_depth),
                          random_unary(rng, var_names, 1, max_depth)),
                      var_names, name=name)
        elif op == "add":
            g = Graph(add(random_unary(rng, var_names, 1, max_depth),
                          random_unary(rng, var_names, 1, max_depth)),
                      var_names, name=name)
        else:
            g = Graph(random_unary(rng, var_names, 1, max_depth), var_names, name=name)
        c = g.complexity()
        if (c["edges"] <= max_edges and c["params"] <= max_params
                and c["depth"] <= max_depth and _indices_ok(g, var_names)):
            for p in g.params():
                p.init = float(rng.uniform(0.1, 2.0))
            return g
    # 兜底: 最简单的单参数线性图
    return Graph(add(scale(var(0), gpar("C1")), scale(var(0), gpar("C2"))),
                 var_names, name=name)


# ===== 交叉 =====
def crossover(g1: Graph, g2: Graph, rng,
              max_edges: int = 14, max_params: int = 6, max_depth: int = 6) -> Optional[Graph]:
    paths1 = all_paths(g1)
    paths2 = all_paths(g2)
    if not paths1 or not paths2:
        return None
    p1 = paths1[rng.integers(0, len(paths1))]
    p2 = paths2[rng.integers(0, len(paths2))]
    try:
        child = replace_subtree(g1.clone(), p1, subtree_at(g2, p2).clone())
    except Exception:
        return None
    if not _indices_ok(child, list(g1.var_names)):
        return None
    c = child.complexity()
    if c["edges"] > max_edges or c["params"] > max_params or c["depth"] > max_depth:
        return None
    return child


# ===== 变异 =====
def mutate_edge_param(g: Graph, rng, magnitude: float = 0.3) -> Graph:
    g2 = g.clone()
    par_list = g2.params()
    if not par_list:
        return g2
    p = par_list[rng.integers(0, len(par_list))]
    p.init = float(p.init + rng.normal(0, magnitude))
    if abs(p.init) < 1e-6:
        p.init = 1e-3
    return g2


def mutate_op_replace(g: Graph, rng) -> Graph:
    """算子替换变异. 不破坏图结构约束; 若替换会导致非法, 跳过."""
    g2 = g.clone()
    paths = all_paths(g2)
    if not paths:
        return g2
    target = paths[rng.integers(0, len(paths))]
    node = subtree_at(g2, target)
    UNARY = ["exp", "log", "sin", "cos", "tanh", "sqrt", "inv", "neg"]
    BINARY = ["sub", "div"]
    if node.op in UNARY:
        node.op = rng.choice([o for o in UNARY if o != node.op])
        return g2
    if node.op in BINARY:
        node.op = rng.choice([o for o in BINARY if o != node.op])
        return g2
    if node.op in ("add", "mul"):
        # 仅当孩子数 >= 2 时才互换; 单孩子 mul(=scale) 不可变 add
        if len(node.children) >= 2 and rng.random() < 0.4:
            new_op = "mul" if node.op == "add" else "add"
            backup = node.op
            node.op = new_op
            # 防御: validate 失败立即回滚
            try:
                g2._validate()
            except Exception:
                node.op = backup
        return g2
    return g2


def mutate_subtree_replace(g: Graph, rng, var_names: List[str],
                           max_edges: int = 14, max_params: int = 6,
                           max_depth: int = 4) -> Graph:
    g2 = g.clone()
    paths = all_paths(g2)
    if not paths:
        return g2
    # 选非叶节点
    non_leaf = [p for p in paths if subtree_at(g2, p).op not in ("var", "const")]
    if not non_leaf:
        return g2
    target = non_leaf[rng.integers(0, len(non_leaf))]
    new_sub = random_graph(rng, var_names, max_edges=max_edges // 2,
                            max_params=max_params, max_depth=max_depth - 1).root
    new_g = replace_subtree(g2, target, new_sub)
    c = new_g.complexity()
    if c["edges"] > max_edges or c["params"] > max_params or c["depth"] > max_depth:
        return g2
    return new_g


def mutate(g: Graph, rng, var_names: List[str], *,
           max_edges: int = 14, max_params: int = 6, max_depth: int = 6,
           attribution_targets: Optional[List[Tuple[int, ...]]] = None,
           intensity: float = 0.7) -> Graph:
    """随机/定向变异. attribution_targets 给出优先生成的子树路径 (从归因得到)."""
    r = rng.random()
    if attribution_targets and r < 0.5:
        target = attribution_targets[rng.integers(0, len(attribution_targets))]
        new_sub = random_graph(rng, var_names, max_edges=max_edges // 2,
                                max_params=max_params - 1, max_depth=max_depth - 1).root
        new_g = replace_subtree(g, target, new_sub)
        c = new_g.complexity()
        if c["edges"] <= max_edges and c["params"] <= max_params and c["depth"] <= max_depth:
            return new_g
    if r < 0.55:
        return mutate_edge_param(g, rng, magnitude=0.3)
    if r < 0.8:
        return mutate_op_replace(g, rng)
    return mutate_subtree_replace(g, rng, var_names, max_edges, max_params, max_depth)


# ===== 选择 / 灭绝 =====
def select(pop: List[Graph], keep_ratio: float = 0.5, rng=None) -> List[Graph]:
    """按 loss 排序, 保留 top keep_ratio."""
    pop_sorted = sorted(pop, key=lambda g: g.loss)
    n = max(2, int(len(pop_sorted) * keep_ratio))
    return pop_sorted[:n]


def extinction(pop: List[Graph], keep: int, rng, var_names: List[str],
               max_edges: int = 14, max_params: int = 6, max_depth: int = 6,
               templates: List[str] = None) -> List[Graph]:
    """保留 top-keep, 其余替换为随机新图 (含 1-2 个模板图注入)."""
    pop_sorted = sorted(pop, key=lambda g: g.loss)
    top = pop_sorted[:keep]
    out = list(top)
    while len(out) < len(pop):
        # 30% 概率注入模板
        if rng.random() < 0.4 and templates:
            tname = rng.choice(templates)
            try:
                if tname == "mul2" and len(var_names) >= 2:
                    g, _ = TEMPLATE_REGISTRY[tname](rng.integers(0, len(var_names)),
                                                      rng.integers(0, len(var_names)))
                else:
                    g, _ = TEMPLATE_REGISTRY[tname](rng.integers(0, len(var_names)))
                g = _rewrap(g, list(var_names))
                for p in g.params():
                    p.init = float(rng.uniform(0.1, 2.0))
                c = g.complexity()
                if (c["edges"] <= max_edges and c["params"] <= max_params
                        and c["depth"] <= max_depth and _indices_ok(g, list(var_names))):
                    out.append(g); continue
            except Exception:
                pass
        out.append(random_graph(rng, var_names, max_edges, max_params, max_depth))
    return out[:len(pop)]


def diversity_refill(pop: List[Graph], n_new: int, rng, var_names: List[str],
                     max_edges: int = 14, max_params: int = 6, max_depth: int = 6) -> List[Graph]:
    return [random_graph(rng, var_names, max_edges, max_params, max_depth)
            for _ in range(n_new)]