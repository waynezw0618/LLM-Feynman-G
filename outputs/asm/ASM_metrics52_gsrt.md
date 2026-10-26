# Weatheritt & Sandberg (2016) §5.2 评估 — GS 组合实验

口径: 式(21) rho_a(y) 逐点 b_ij 余弦对齐; 式(22) rho_a,case=∫rho_a dy (y/h∈[-1,1]); 式(23) tau_r=(b22+1/3)/(b11+1/3) 剖面平均绝对误差; 式(24) tau_e=mean_y(||dtau||_F/(2k)).

⚠ 与论文的差异 (诚实声明): 论文的 rho_a 是 RANS a posteriori 解对 LES/DNS 的对齐; 我们无 RANS 回代, 在 DNS 平均场上评估, 属 **on-design a priori 对齐**. 偏相关样本是 4 arm × 5 case = 20 点 (论文为 50 个独立模型), arm 间非独立抽样, r 的显著性解释需谨慎.

## 1. 式(21)(22): 各向异性对齐 rho_a,case (=∫rho_a dy, 满分 2.0)

| arm | Ro7.5 | Ro15 | Ro30 | Ro75 | Ro150 | mean |
|---|---|---|---|---|---|---|
| heuristic_pope_wy | 1.442 | 1.264 | 1.184 | 1.060 | 1.023 | 1.195 |
| heuristic_gsonly | 1.398 | 1.258 | 1.212 | 1.115 | 1.039 | 1.204 |
| heuristic_pope_wy_rt | 1.422 | 1.204 | 1.175 | 1.035 | 0.984 | 1.164 |
| heuristic_pope_gsrt | 1.502 | 1.263 | 1.185 | 1.096 | 1.017 | 1.213 |

参考: 完美模型 rho_a,case = 2.0 (积分区间长). 线性涡黏 (b=0) 的 rho_a 无定义 (分母为 0), 论文以线性 SST 为基线; 我们的基线是 heuristic_pope_wy (P1a).

## 2. 式(23): tau_r = tau22/tau11 剖面平均绝对误差 (越小越好)

| arm | Ro7.5 | Ro15 | Ro30 | Ro75 | Ro150 | mean |
|---|---|---|---|---|---|---|
| heuristic_pope_wy | 0.106 | 0.143 | 0.123 | 0.149 | 0.183 | 0.141 |
| heuristic_gsonly | 0.119 | 0.152 | 0.129 | 0.156 | 0.183 | 0.148 |
| heuristic_pope_wy_rt | 0.174 | 0.192 | 0.150 | 0.188 | 0.254 | 0.191 |
| heuristic_pope_gsrt | 0.231 | 0.287 | 0.194 | 0.176 | 0.108 | 0.200 |

注: tau_r 在 b11+1/3 接近 0 处误差放大 (论文同口径), 分母截断 0.05 防发散.

## 3. 式(24): tau_e 应力幅值相对误差 (越小越好)

| arm | Ro7.5 | Ro15 | Ro30 | Ro75 | Ro150 | mean |
|---|---|---|---|---|---|---|
| heuristic_pope_wy | 0.2454 | 0.1855 | 0.1397 | 0.1161 | 0.1120 | 0.1597 |
| heuristic_gsonly | 0.2457 | 0.2093 | 0.1625 | 0.1179 | 0.1173 | 0.1705 |
| heuristic_pope_wy_rt | 0.2609 | 0.2213 | 0.1657 | 0.1385 | 0.1477 | 0.1868 |
| heuristic_pope_gsrt | 0.3593 | 0.3699 | 0.2904 | 0.2143 | 0.1531 | 0.2774 |

tau_e 是论文中的混淆变量: 单目标优化可能牺牲幅值 (a_model ≈ C·a_dns, C≠1 而对齐 rho_a 不变), 需在偏相关中控制.

## 4. 式(25)-(28): 控 tau_e 后 Fit ↔ rho_a,case 偏相关

- 样本: 20 个 (arm, case) 点
- 简单相关 corr(Fit, rho_a) = -0.033
- **偏相关 (控 tau_e) r = -0.372, p = 0.106**
- 论文 region I 参照: r = 0.93, p = 1e-11 (50 模型)

逐点明细 (Fit 低=训练好; 论文 Fit 越高越好, 此处 loss 反向):

| arm | Ro | Fit (loss) | rho_a,case | tau_e |
|---|---|---|---|---|
| heuristic_pope_wy | 7.5 | 0.4318 | 1.442 | 0.2454 |
| heuristic_pope_wy | 15 | 0.4318 | 1.264 | 0.1855 |
| heuristic_pope_wy | 30 | 0.4318 | 1.184 | 0.1397 |
| heuristic_pope_wy | 75 | 0.4318 | 1.060 | 0.1161 |
| heuristic_pope_wy | 150 | 0.4318 | 1.023 | 0.1120 |
| heuristic_gsonly | 7.5 | 0.4817 | 1.398 | 0.2457 |
| heuristic_gsonly | 15 | 0.4817 | 1.258 | 0.2093 |
| heuristic_gsonly | 30 | 0.4817 | 1.212 | 0.1625 |
| heuristic_gsonly | 75 | 0.4817 | 1.115 | 0.1179 |
| heuristic_gsonly | 150 | 0.4817 | 1.039 | 0.1173 |
| heuristic_pope_wy_rt | 7.5 | 0.5029 | 1.422 | 0.2609 |
| heuristic_pope_wy_rt | 15 | 0.5029 | 1.204 | 0.2213 |
| heuristic_pope_wy_rt | 30 | 0.5029 | 1.175 | 0.1657 |
| heuristic_pope_wy_rt | 75 | 0.5029 | 1.035 | 0.1385 |
| heuristic_pope_wy_rt | 150 | 0.5029 | 0.984 | 0.1477 |
| heuristic_pope_gsrt | 7.5 | 0.4893 | 1.502 | 0.3593 |
| heuristic_pope_gsrt | 15 | 0.4893 | 1.263 | 0.3699 |
| heuristic_pope_gsrt | 30 | 0.4893 | 1.185 | 0.2904 |
| heuristic_pope_gsrt | 75 | 0.4893 | 1.096 | 0.2143 |
| heuristic_pope_gsrt | 150 | 0.4893 | 1.017 | 0.1531 |

图: figures/metrics52_rho_a_profiles.png (5 工况 rho_a(y) 剖面, 4 arm 对比)
