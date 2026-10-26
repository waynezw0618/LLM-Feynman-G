import sys, os
sys.path.insert(0, r"d:\AI\LLM-Feynman-G")
import sympy as sp
from lgf import eq_compat as EC

expr = "params[0] * np.sqrt(v1 / v2)"
norm = EC.normalize_eq_expr(expr, ["v1", "v2"])
print("norm =", repr(norm))

local_ns = {"sqrt": sp.sqrt, "exp": sp.exp, "log": sp.log,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "tanh": sp.tanh, "Abs": sp.Abs, "v1": sp.Symbol("v1"), "v2": sp.Symbol("v2")}
try:
    e = sp.sympify(norm, locals=local_ns, evaluate=True)
    print("sympify OK ->", e, type(e))
    print("free syms =", e.free_symbols)
except Exception as ex:
    import traceback; traceback.print_exc()

print("--- direct converter ---")
print(EC.eq_line_to_lgf_dsl(expr, ["v1", "v2"]))

print("--- separable ---")
print(EC.context_aware_skeletons("multiplicative separability f = g*h", ["a", "b"], 4))
print(EC.context_aware_skeletons("additive separab f approx g+h", ["a", "b"], 4))
