# 真 ASM 最优模型方程汇总（常数代入）

生成: rc_asm_true_figs.py · 单位: 无量纲 (b_ij, S/W 已乘 k/eps)

## 1. 建模输入特征（对问题 2 的回答）

全部 17 个特征均作为候选输入提供给 PySR 与 LGF，由算法自选：

- **Pope (1975) 十个对称无迹张量基**: T1..T10（在当前张量分量 (ij) 处取值，S_ij 与 W_ij 已乘 k/eps 归一化）
- **5 个标量不变量**: eta2=tr(S²), zeta2=tr(W²), s3=tr(S³), so2=tr(S²W), s2o2=tr(S²W²)
- **湍流雷诺数**: ret=k²/(ν·eps)
- **局部旋转数**: roloc=2Ωk/eps（带符号）

结构上即 W&S (JCP 2016) eq.19: b_ij = Σ_m g_m(I, Re_t, Ro_loc)·T^(m)_ij，g_m 的函数形式由符号回归发现，常数一律为跨 5 工况全局常数。

注: s3、so2 在本二维平均流场恒为零；v 变体下 zeta2≡eta2。ret/roloc 提供了但**没有任何一条最优公式选用**（详见下方表格与报告 §6）。

## 2. 主对比表

| 变体 | 方法 | loss_asm ↓ | rho_a ↑ | 最差工况 nmse | 复杂度 | 选用特征 |
|---|---|---|---|---|---|---|
| v | PySR | 0.4189 | +0.538 | 0.4619 | 30 nodes | T8, eta2, s2o2 |
| v | LGF-mg (minimax) | 0.5571 | +0.567 | 0.5814 | 11 edges | T2, T7 |
| v | LGF-go (minimax) | 0.5121 | +0.746 | 0.5354 | 15 edges | T1, T2, T3, T4 |
| g | PySR | 0.3736 | +0.606 | 0.4857 | 29 nodes | T4, T8, eta2, s3, s2o2 |
| g | LGF-mg (minimax) | 0.6082 | +0.478 | 0.7820 | 10 edges | T2, eta2 |
| g | LGF-go (minimax) | 0.3797 | +0.725 | 0.5025 | 11 edges | T1, T2, eta2, zeta2 |
| e | PySR | 0.4666 | +0.481 | 0.6326 | 28 nodes | T8, T9, s2o2 |
| e | LGF-mg (minimax) | 0.5584 | +0.160 | 0.7323 | 12 edges | T7, T8, s2o2 |
| e | LGF-go (minimax) | 0.5366 | +0.380 | 0.7805 | 7 edges | T2, zeta2 |

## 3. 全部最优方程

### 变体 v — $v$: $\mathbf{W}=\overline{W}$ (vorticity)

**PySR** (complexity=30, loss_asm=0.4189, rho_a=+0.538):

```
b_ij = T8*0.2637855*s2o2*(exp(-4.846445/eta2) + exp(-4.846445*eta2)*exp(eta2/eta2)*s2o2*0.2637855/s2o2)/(s2o2*s2o2)
```

**LGF-mg** (minimax/MiniMax-M2.7, edges=11, loss_asm=0.5571, rho_a=+0.567):

```
b_ij = ((4.7573e-05) * (1)) * (T7) + (-1.00695) * (T2) + (0.00507645) * (1) + T2
```

<sub>原始 DSL: `(C3 * (1)) * (T7) + C2 * (T2) + C1 * (1) + T2` · theta=`[0.005076, -1.006947, 4.8e-05]`</sub>

**LGF-go** (minimax/MiniMax-M2.7, edges=15, loss_asm=0.5121, rho_a=+0.746):

```
b_ij = (0.0058514) * (T4) + (-0.00685299) * (T2) + (-0.0208006) * (T1) + (0.00686077) * (T3) + ((0.000140394) * (T2)) * (T3)
```

<sub>原始 DSL: `C5 * (T4) + C4 * (T2) + C3 * (T1) + C2 * (T3) + (C1 * (T2)) * (T3)` · theta=`[0.00014, 0.006861, -0.020801, -0.006853, 0.005851]`</sub>

### 变体 g — $g$: GS, $\mathbf{W}=\overline{W}+\Omega$

**PySR** (complexity=29, loss_asm=0.3736, rho_a=+0.606):

```
b_ij = (eta2*0.7407941*T4/(s2o2*(2.0976007 - (s3 + 3.3073611))*(exp(s2o2) + exp(s3) - 1*3.3073611)) - T8/s2o2 + s3)*(-0.33395973)
```

**LGF-mg** (minimax/MiniMax-M2.7, edges=10, loss_asm=0.6082, rho_a=+0.478):

```
b_ij = ((-0.00190211) * (tanh((0.0314002) * (eta2)))) * (T2) + (-0.061674) * (tanh((1.33846e+06) * (T2)))
```

<sub>原始 DSL: `(C3 * (tanh(C4 * (eta2)))) * (T2) + C1 * (tanh(C2 * (T2)))` · theta=`[-0.061674, 1338464.676605, -0.001902, 0.0314]`</sub>

**LGF-go** (minimax/MiniMax-M2.7, edges=11, loss_asm=0.3797, rho_a=+0.725):

```
b_ij = (-0.090244) * (tanh(T1)) + (-0.234501) * (((2.91675) * (T2)) * (inv(eta2 + zeta2)))
```

<sub>原始 DSL: `C3 * (tanh(T1)) + C1 * ((C2 * (T2)) * (inv(eta2 + zeta2)))` · theta=`[-0.234501, 2.916746, -0.090244]`</sub>

### 变体 e — $e$: $\mathbf{W}=\overline{W}+2\Omega$

**PySR** (complexity=28, loss_asm=0.4666, rho_a=+0.481):

```
b_ij = (0.38392627*T8 - 0.574540977498582*T9*sqrt(121.45341 - s2o2)/(71.3206093926068 - 0.12034881*s2o2))/s2o2
```

**LGF-mg** (minimax/MiniMax-M2.7, edges=12, loss_asm=0.5584, rho_a=+0.160):

```
b_ij = ((-0.516839) * (T8)) / (((0.272698) * (T7)) / (T7 + (-1.41063) * (1)) + (-1.41592) * (s2o2))
```

<sub>原始 DSL: `(C4 * (T8)) / ((C3 * (T7)) / (T7 + C2 * (1)) + C1 * (s2o2))` · theta=`[-1.415922, -1.410626, 0.272698, -0.516839]`</sub>

**LGF-go** (minimax/MiniMax-M2.7, edges=7, loss_asm=0.5366, rho_a=+0.380):

```
b_ij = ((-0.400176) * (T2)) * (inv((0.0118046) * (T2) + zeta2))
```

<sub>原始 DSL: `(C2 * (T2)) * (inv(C1 * (T2) + zeta2))` · theta=`[0.011805, -0.400176]`</sub>

## 4. 迭代图口径说明（诚实声明）

- **LGF**: 逐代日志只保存 best_graph（无 θ）。为保证与最终模型同一评估口径，对每个去重后的逐代最优图用与 final refit 相同设置（n_starts=8, maxiter=150）在全量行（5 工况 × 6 分量）上重新拟合后评估 loss_asm 与 rho_a,case；末点 `final` 为 result.json 的最终最优模型（final refit 产物，可能与末代图不同）。
- **LGF 曲线语义**: 粗线 = best-so-far 包络（loss 取累积最小、rho_a 取累积最大），细淡线 = 逐代原始值。包络用于抑制逐代重拟合落入劣质局部极小造成的非单调上跳；星号 = result.json 的 final refit 模型，位于末代 +1.5 处。
- **PySR**: 未保存逐迭代历史，仅 Pareto 前沿（复杂度阶梯）→ 以 complexity 为横轴画 loss_asm / rho_a（等效的复杂度-精度权衡曲线）。
- 所有 rho_a,case 均为 W&S (2016) eq.22 的离散实现：逐点 3×3 张量余弦对有效壁面点平均。
