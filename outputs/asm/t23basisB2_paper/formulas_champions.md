# 变体冠军公式套（正式口径：公式形状由 LGF 符号回归发现，常数来自联合重拟合）

> 生成脚本：`scripts/rc_asm_basisB2_formulas.py`；数据源：`outputs/asm/runs_basisB2/*/result.json`。
> 所有公式严格保持张量基形式 $b_{23}=\sum_j s_j\,a_j(\Theta)\,\tilde T^{(j)}_{23}$，
> 其中 $\tilde T^{(j)}_{23}$ 为逐工况 RMS 归一的通道分量，$s_j$ 为该通道 scale（各公式下方给出），
> $\Theta=\{\eta_2,\zeta_2,S^2/\Omega^2,Re_\tau,r_{o,\mathrm{loc}},y_w\}$（yw∈[0,1] 无量纲壁距）。

### 变体 e —— 冠军组 `e_lgf_mg_big` (LGF-mg big (joint refit))

- 最终口径 NMSE(b23) = **0.0576**（worst 0.068），SR 中间口径 0.792；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..4 对应通道 T1s, T5s, T6s, T10s。

**a1(Θ)** — 通道 T1s，scale = 0.0402578：

$$a_{1}(\Theta) = \frac{- 0.000307839 Re_\tau + 3.01635 \left(\eta_{2} + 11.4144\right) \log{\left(1.54276 y_w + 0.922545 \right)}}{\eta_{2} + 11.4144}$$

**a2(Θ)** — 通道 T5s，scale = 0.606326：

$$a_{2}(\Theta) = 1.89998 y_w - \frac{20.8929 y_w}{\sqrt{r_{o,\mathrm{loc}}} + 9.66228} - \frac{1.24817}{\eta_{2} + 1.16246}$$

**a3(Θ)** — 通道 T6s，scale = 0.7168：

$$a_{3}(\Theta) = 0.0533727 y_w^{2} - \frac{9.08352 y_w}{S^2/\Omega^2 - 52.5758} + \frac{0.0182819}{3.04326 r_{o,\mathrm{loc}} - 0.656019}$$

**a4(Θ)** — 通道 T10s，scale = 0.436623：

$$a_{4}(\Theta) = \frac{0.319931 r_{o,\mathrm{loc}} - 0.206477 \left(r_{o,\mathrm{loc}} + 4.6822\right) \tanh{\left(0.00388523 r_{o,\mathrm{loc}} + 7.55284 y_w \right)}}{r_{o,\mathrm{loc}} + 4.6822}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.0402578
a1 = -0.000307839*ret/(eta2 + 11.4144) + 3.01635*np.log(1.54276*yw + 0.922545)
# a2, channel T5s, scale=0.606326
a2 = 1.89998*yw - 20.8929*yw/(np.sqrt(roloc) + 9.66228) - 1.24817/(eta2 + 1.16246)
# a3, channel T6s, scale=0.7168
a3 = 0.0533727*yw**2 - 9.08352*yw/(s2o2 - 52.5758) + 0.0182819/(3.04326*roloc - 0.656019)
# a4, channel T10s, scale=0.436623
a4 = 0.319931*roloc/(roloc + 4.6822) - 0.206477*np.tanh(0.00388523*roloc + 7.55284*yw)
```

逐工况 NMSE：7.5: 0.048 | 15: 0.050 | 30: 0.066 | 75: 0.068 | 150: 0.055

数值核对：✅ max rel diff = 1.47e-14

### 变体 g —— 冠军组 `g_lgf_mg_big` (LGF-mg big (joint refit))

- 最终口径 NMSE(b23) = **0.1238**（worst 0.237），SR 中间口径 0.376；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..4 对应通道 T1s, T5s, T6s, T10s。

**a1(Θ)** — 通道 T1s，scale = 0.0202466：

$$a_{1}(\Theta) = 0.000582218 \eta_{2} - 2.59543 y_w^{2} + 2.33371 y_w - 0.359922$$

**a2(Θ)** — 通道 T5s，scale = 0.397212：

$$a_{2}(\Theta) = - 0.00940079 \sqrt{r_{o,\mathrm{loc}}} - 1.37266 y_w + 0.300209 \left(\log{\left(r_{o,\mathrm{loc}} \right)} + 1.04348\right) \tanh{\left(y_w \right)} + 0.0323185$$

**a3(Θ)** — 通道 T6s，scale = 0.139926：

$$a_{3}(\Theta) = \frac{\left(3.5454 y_w^{2} - 1.73012 y_w + 0.143592\right) \log{\left(y_w + 203.34 \right)}}{0.884546 r_{o,\mathrm{loc}} + 9.19053 y_w + 1.38539}$$

**a4(Θ)** — 通道 T10s，scale = 0.152257：

$$a_{4}(\Theta) = - \frac{59.2735 \eta_{2} \left(1.22487 y_w \zeta_{2} - 0.0501875 \left(\zeta_{2} - 9.16154\right) \tanh{\left(r_{o,\mathrm{loc}} \right)}\right) - 0.269994 \left(\eta_{2} + 899.52\right) \left(\zeta_{2} - 9.16154\right)}{\left(\eta_{2} + 899.52\right) \left(\zeta_{2} - 9.16154\right)}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.0202466
a1 = 0.000582218*eta2 - 2.59543*yw**2 + 2.33371*yw - 0.359922
# a2, channel T5s, scale=0.397212
a2 = -0.00940079*np.sqrt(roloc) - 1.37266*yw + 0.300209*np.log(2.83908*roloc)*np.tanh(yw) + 0.0323185
# a3, channel T6s, scale=0.139926
a3 = (3.5454*yw**2 - 1.73012*yw + 0.143592)*np.log(yw + 203.34)/(0.884546*roloc + 9.19053*yw + 1.38539)
# a4, channel T10s, scale=0.152257
a4 = 59.2735*eta2*(-1.22487*yw*zeta2/(zeta2 - 9.16154) + 0.0501875*np.tanh(roloc))/(eta2 + 899.52) + 0.269994
```

逐工况 NMSE：7.5: 0.125 | 15: 0.070 | 30: 0.106 | 75: 0.237 | 150: 0.081

数值核对：✅ max rel diff = 2.59e-12

### 变体 m —— 冠军组 `m_lgf_go_big` (LGF-go big (joint refit))

- 最终口径 NMSE(b23) = **0.0722**（worst 0.155），SR 中间口径 34.029；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..10 对应通道 T1s, T5s_v, T5s_g, T5s_e, T6s_v, T6s_g, T6s_e, T10s_v, T10s_g, T10s_e。

**a1(Θ)** — 通道 T1s，scale = 0.0192964：

$$a_{1}(\Theta) = - \frac{0.00464753 \sqrt{Re_\tau} y_w}{0.000428044 S^2/\Omega^2 + 1.71477} + \frac{2.50071 r_{o,\mathrm{loc}}}{0.830462 r_{o,\mathrm{loc}} + 7.58463} - 2.33993 y_w$$

**a2(Θ)** — 通道 T5s_v，scale = 0.69033：

$$a_{2}(\Theta) = 2.49911 y_w^{2} - 0.0867483 y_w \left(\log{\left(r_{o,\mathrm{loc}} \right)} - 5.97991\right) - 2.28592 \tanh{\left(r_{o,\mathrm{loc}} \right)} + 2.2864$$

**a3(Θ)** — 通道 T5s_g，scale = 1.54364：

$$a_{3}(\Theta) = - \frac{4.52963 \cdot 10^{-7} r_{o,\mathrm{loc}}^{\frac{5}{2}} - \left(0.0032569 r_{o,\mathrm{loc}} + 1\right) \left(5.43221 \tanh{\left(1.1742 r_{o,\mathrm{loc}} \right)} - 5.44091 \tanh{\left(1.23157 r_{o,\mathrm{loc}} \right)}\right)}{0.0032569 r_{o,\mathrm{loc}} + 1}$$

**a4(Θ)** — 通道 T5s_e，scale = 2.775：

$$a_{4}(\Theta) = - 4.03543 \cdot 10^{-5} r_{o,\mathrm{loc}} y_w^{2} - 0.013683 y_w + 0.00136314 + e^{- 11.4082 r_{o,\mathrm{loc}}}$$

**a5(Θ)** — 通道 T6s_v，scale = 0.145571：

$$a_{5}(\Theta) = - \frac{2.67749 y_w \zeta_{2}}{\zeta_{2} + 30.2419} + 0.107535 - 29.4225 e^{- 3.60952 r_{o,\mathrm{loc}}}$$

**a6(Θ)** — 通道 T6s_g，scale = 4.93981：

$$a_{6}(\Theta) = \frac{2.24864 \zeta_{2} - \left(0.000167474 r_{o,\mathrm{loc}} + 2.25357 \tanh{\left(1.7304 r_{o,\mathrm{loc}} \right)}\right) \left(\zeta_{2} - 1.88983 \cdot 10^{-5}\right)}{\zeta_{2} - 1.88983 \cdot 10^{-5}}$$

**a7(Θ)** — 通道 T6s_e，scale = 4.72961：

$$a_{7}(\Theta) = \frac{0.0063886 \sqrt{r_{o,\mathrm{loc}}} y_w + \left(1.45957 y_w + \tanh{\left(y_w \right)}\right) \left(2.19698 \tanh{\left(1.52017 r_{o,\mathrm{loc}} \right)} - 2.19851\right)}{1.45957 y_w + \tanh{\left(y_w \right)}}$$

**a8(Θ)** — 通道 T10s_v，scale = 0.523326：

$$a_{8}(\Theta) = \frac{0.0978421 y_w + \left(0.000377304 r_{o,\mathrm{loc}} - 0.349575 y_w\right) \left(r_{o,\mathrm{loc}} - 0.709357\right) - 0.00489058}{r_{o,\mathrm{loc}} - 0.709357}$$

**a9(Θ)** — 通道 T10s_g，scale = 0.131477：

$$a_{9}(\Theta) = \sqrt{r_{o,\mathrm{loc}}} \left(- 0.0011614 r_{o,\mathrm{loc}} + 0.0685236 \log{\left(5.73961 y_w + 0.230465 \right)}\right)$$

**a10(Θ)** — 通道 T10s_e，scale = 0.28521：

$$a_{10}(\Theta) = \frac{24.833 y_w \left(0.0185322 r_{o,\mathrm{loc}} - 0.00880291 \left(r_{o,\mathrm{loc}} + 1\right) \log{\left(r_{o,\mathrm{loc}} \right)}\right) + 0.00468331 \left(\sqrt{r_{o,\mathrm{loc}}} + r_{o,\mathrm{loc}}\right) \left(r_{o,\mathrm{loc}} + 1\right)}{r_{o,\mathrm{loc}} + 1}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.0192964
a1 = -0.00464753*np.sqrt(ret)*yw/(0.000428044*s2o2 + 1.71477) + 2.50071*roloc/(0.830462*roloc + 7.58463) - 2.33993*yw
# a2, channel T5s_v, scale=0.69033
a2 = 2.49911*yw**2 - 0.0867483*yw*np.log(0.00252906*roloc) - 2.28592*np.tanh(roloc) + 2.2864
# a3, channel T5s_g, scale=1.54364
a3 = -4.52963e-7*roloc**(3/2)/(0.0032569 + 1/roloc) + 5.43221*np.tanh(1.1742*roloc) - 5.44091*np.tanh(1.23157*roloc)
# a4, channel T5s_e, scale=2.775
a4 = -4.03543e-5*roloc*yw**2 - 0.013683*yw + 0.00136314 + np.exp(-11.4082*roloc)
# a5, channel T6s_v, scale=0.145571
a5 = -2.67749*yw*zeta2/(zeta2 + 30.2419) + 0.107535 - 29.4225*np.exp(-3.60952*roloc)
# a6, channel T6s_g, scale=4.93981
a6 = -0.000167474*roloc + 2.24864*zeta2/(zeta2 - 1.88983e-5) - 2.25357*np.tanh(1.7304*roloc)
# a7, channel T6s_e, scale=4.72961
a7 = 0.0063886*np.sqrt(roloc)*yw/(1.45957*yw + np.tanh(yw)) + 2.19698*np.tanh(1.52017*roloc) - 2.19851
# a8, channel T10s_v, scale=0.523326
a8 = 0.000377304*roloc - 0.349575*yw - 0.0225776*(0.216612 - 4.33359*yw)/(roloc - 0.709357)
# a9, channel T10s_g, scale=0.131477
a9 = 0.183739*np.sqrt(roloc)*(-0.00632093*roloc + 0.37294*np.log(5.73961*yw + 0.230465))
# a10, channel T10s_e, scale=0.28521
a10 = 0.00468331*np.sqrt(roloc) + 0.00468331*roloc + 24.833*yw*(0.0185322*roloc/(roloc + 1) - 0.00880291*np.log(roloc))
```

逐工况 NMSE：7.5: 0.033 | 15: 0.054 | 30: 0.041 | 75: 0.078 | 150: 0.155

数值核对：✅ max rel diff = 1.56e-13

### 变体 v —— 冠军组 `v_lgf_go` (LGF-go legacy (joint refit))

- 最终口径 NMSE(b23) = **2.2305**（worst 3.058），SR 中间口径 7.668；公式形状固定，常数来自联合重拟合。
- 重构式：`b23 = Σ_j scale_j · a_j(Θ) · T̃_j23`，j = 1..4 对应通道 T1s, T5s, T6s, T10s。

**a1(Θ)** — 通道 T1s，scale = 0.100647：

$$a_{1}(\Theta) = 0.179751 \left(\log{\left(r_{o,\mathrm{loc}} \right)} + 1.33836\right) \log{\left(2.44248 - 2.12821 y_w \right)}$$

**a2(Θ)** — 通道 T5s，scale = 9.65403：

$$a_{2}(\Theta) = - \left(0.0157721 \tanh{\left(3.638 r_{o,\mathrm{loc}} \right)} + 0.013384 \tanh{\left(0.00630331 S^2/\Omega^2 \right)}\right) \log{\left(y_w + 0.266476 \right)}$$

**a3(Θ)** — 通道 T6s，scale = 0.278312：

$$a_{3}(\Theta) = \frac{- 7.12859 \cdot 10^{-5} \eta_{2} \left(S^2/\Omega^2 - 0.0095088\right) + 0.0373976 S^2/\Omega^2 + 0.00930046}{S^2/\Omega^2 - 0.0095088}$$

**a4(Θ)** — 通道 T10s，scale = 9.56966：

$$a_{4}(\Theta) = \frac{0.190595 r_{o,\mathrm{loc}} \left(r_{o,\mathrm{loc}} + 75.1609 y_w\right)}{466.199 r_{o,\mathrm{loc}}^{2} - 46.7772}$$

Python 可调用表达式（`import numpy as np` 后直接用；输入为标量或同形数组，已内联联合重拟合常数）：

```python
# a1, channel T1s, scale=0.100647
a1 = 0.179751*np.log(3.81279*roloc)*np.log(2.44248 - 2.12821*yw)
# a2, channel T5s, scale=9.65403
a2 = 0.410519*(-0.0384198*np.tanh(3.638*roloc) - 0.0326027*np.tanh(0.00630331*s2o2))*np.log(yw + 0.266476)
# a3, channel T6s, scale=0.278312
a3 = -7.12859e-5*eta2 - 15.7459*(-0.00237507*s2o2 - 0.000590659)/(s2o2 - 0.0095088)
# a4, channel T10s, scale=9.56966
a4 = -0.190595*roloc*(roloc + 75.1609*yw)/(46.7772 - 466.199*roloc**2)
```

逐工况 NMSE：7.5: 1.602 | 15: 1.771 | 30: 2.152 | 75: 2.569 | 150: 3.058

数值核对：✅ max rel diff = 7.85e-13
