# 检查: 壁面附近点的 valid 性 / cond / k / eps; ypd 的真实含义验证
import sys, os
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))

from rc_asm_prep import load_case

fc = load_case(75.0, variant="vort")
yh, yp, k, eps = fc["yh"], fc["yp"], fc["k"], fc["eps"]
valid = (k > 1e-4) & (eps > 1e-12) & np.isfinite(fc["b6"]).all(1)
print(f"点数: total={len(yh)}  valid={valid.sum()}")
print(f"y/h 范围: [{yh.min():.4f}, {yh.max():.4f}]")
print(f"yp (从中心计) 范围: [{yp.min():.1f}, {yp.max():.1f}]")
print(f"valid 点的 |ypd| 范围: [{np.abs(yp[valid]).min():.1f}, {np.abs(yp[valid]).max():.1f}]")
print()
print("壁面附近 8 个点 (按 |y/h| 降序):")
idx = np.argsort(-np.abs(yh))[:8]
print(f"{'y/h':>9} {'y+':>8} {'k':>10} {'eps':>10} {'valid':>6} {'k/eps':>10}")
for i in sorted(idx, key=lambda j: yh[j]):
    print(f"{yh[i]:9.4f} {yp[i]:8.1f} {k[i]:10.3g} {eps[i]:10.3g} {str(valid[i]):>6} {k[i]/max(eps[i],1e-30):10.3g}")
print()
# 真实壁面距离 (wall units)
ypd_wall = 180.0 - np.abs(yp)
print("若 ypd 是真壁面距离, 应为 (1-|y/h|)*180; 当前 ypd=|y/h|*180 是离中心距离")
print("近壁点示例: y/h=0.9997 -> 当前 ypd =", f"{abs(yp[np.argmin(np.abs(yh-0.9997))]):.1f}", "; 真壁距 =", f"{180-abs(yp[np.argmin(np.abs(yh-0.9997))]):.1f}")
