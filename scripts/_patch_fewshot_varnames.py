# -*- coding: utf-8 -*-
"""修正 few-shot 示例: 用真实特征名 (eta2,s3,zeta2v,so2v,s2o2v,zeta2e,so2e,s2o2e,yw),
不泄漏 T5g 精确解, 只示范合法算子用法."""
import ast, sys

p = r'lgf\llm_ops.py'
m = open(p, encoding='utf-8').read()

old = ('    fewshot = ("\\n[FEW-SHOT 合法 DSL 范例, 直接复用其结构/算子组合]\\n"\n'
       '               \'  {"dsl": "(zeta2 + C3 * (1)) / (zeta2 + C2 * (1)) + C1 * (1)", "name": "rational_shift"}\\n\'\n'
       '               \'  {"dsl": "C1 * (tanh(inv(C3 * (o23) + C2 * (1))))", "name": "saturation_tanh"}\\n\'\n'
       '               \'  {"dsl": "(C4 * (s12)) / (s12 + C1 * (C3 * (eta2) + C2 * (1)))", "name": "wall_damp"}\\n\'\n'
       '               "注意: 每条都是一条完整的 dsl 字符串, 必须严格匹配上述算子名."\n'
       '               ) if use_fewshot else ""\n')

new = ('    fewshot = ("\\n[FEW-SHOT 合法 DSL 范例, 直接复用其算子/结构]\\n"\n'
       '               \'  {"dsl": "(C4 * (1)) / (K1 * (1) + C3 * (eta2)) + C2 * (zeta2e)", "name": "rational_plus_linear"}\\n\'\n'
       '               \'  {"dsl": "C1 * (tanh(inv(C3 * (s2o2e) + C2 * (1))))", "name": "saturation_tanh"}\\n\'\n'
       '               \'  {"dsl": "C1 * (exp(inv(C2 * (yw) + K1 * (1))))", "name": "wall_damp"}\\n\'\n'
       '               "注意: 变量名只能用上述已知特征名; 算子名只能用白名单; 系数写 C1/C2/K1 等."\n'
       '               ) if use_fewshot else ""\n')

if old not in m:
    i = m.find('fewshot = (')
    print('NOT FOUND. Actual:')
    print(repr(m[i:i+700]))
    sys.exit(1)
m2 = m.replace(old, new, 1)
open(p, 'w', encoding='utf-8', newline='').write(m2)
ast.parse(m2)
print('OK patched fewshot varnames')
