"""shipman.context — physics_context 文案 + few-shot DSL + 后端配置."""
from __future__ import annotations

# 供 G_INIT/M_GUIDE 的物理上下文 (经 cfg.physics_context → extra_context 注入 prompt)
PHYSICS_CTX_MMG = """[船舶操纵 MMG 建模任务]
目标: 从仿真/试验数据辨识 3-DOF 操纵方程的船体力项 (灰箱: 推进/舵力已扣除).
方程: (m'+my')v̇m + ... = Y_H(vm, r);  (IzG'+xG'^2m'+Jz')ṙ + ... = N_H(vm, r)
所有量 prime 无量纲 (力除 ρ/2·Lpp·d·U², 力矩除 ρ/2·Lpp²·d·U²).

特征列含义 (预计算列, DSL 中按名引用):
  vm=船中侧速, r=艏摇角速度, u=纵速, delta=舵角
  vmvm=vm², vmr=vm·r, rr=r², vm3=vm³, vm2r=vm²r, vmr2=vm·r², r3=r³, vm4=vm⁴
  uabsu=u|u|, vmabsvm=vm|vm|, rabsr=r|r| (阻尼型)
  dv_a05 = 0.5 阶 Grünwald-Letnikov 分数阶导数列 (a03/a05/a07/a09 = α 0.3/0.5/0.7/0.9)
  conv_r_K1/K2 = 记忆核卷积列 (指数核 τ=5s/50s, 作用于 r), conv_r_PW = 幂律核
  注: du/dv/dr 前缀 = 对应变量的 D^α 列; 如 dv_a05 是 vm 的 0.5 阶导数.

物理先验 (硬规则):
  1. Y_H 关于 vm 为奇函数: 只允许 vm, r, vm³, vm²r, vm·r², r³, vm|vm| 型项
  2. 符号: Y_v<0, N_v<0, N_r<0 (静航向稳定性导数); KVLCC2 属航向不稳定 (N_v+N_r 组合仍偏不稳)
  3. 只允许全局常数 C# (跨工况共享), 禁止逐工况参数 K#
  4. 禁用 tanh; 有理式仅在分母恒正时可用
  5. 分数阶列 d*_a* 只在数据确实含记忆效应时才应入选 (V1 数据下系数应→0)

few-shot DSL 骨架示例:
  {"dsl": "C1*vm + C2*r + C3*vm3 + C4*vm2r + C5*vmr2 + C6*r3", "name": "mmg_sway_poly3"}
  {"dsl": "C1*vm + C2*r + C3*dv_a05", "name": "mmg_sway_frac"}
  {"dsl": "C1*vm + C2*r + C3*conv_r_K1", "name": "mmg_yaw_memory"}
"""

# 后端配置 (沿用 scripts/rc_asm_run.py BACKENDS 约定; glm53 为 asm 轮验证配置)
BACKENDS = {
    "heuristic": dict(backend="heuristic"),
    "glm53": dict(backend="openai", llm_model="glm-5.3",
                  extra_body={"thinking": {"type": "enabled"},
                              "reasoning_effort": "low"},
                  max_tokens=16384),
}
