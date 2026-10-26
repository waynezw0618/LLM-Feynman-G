# -*- coding: utf-8 -*-
"""Temporary numeric probe: verify budget-column sign conventions of Yang & Wang DNS data."""
import numpy as np

d = np.loadtxt(r'd:\AI\LLM-Feynman-G\data\Retau180Ro30.txt', comments='%%')
y, U, W = d[:, 0], d[:, 1], d[:, 2]
uu, vv, ww, uv, uw, vw = d[:, 3], d[:, 4], d[:, 5], d[:, 6], d[:, 7], d[:, 8]
dU = np.gradient(U, y)      # d(U+)/d(y/h)
dW = np.gradient(W, y)
hp = 180.0                  # h+ = Re_tau
dUp = dU / hp               # wall-unit gradient
dWp = dW / hp

# production: P_ij = -<u_i u_k> dU_j/dx_k - <u_j u_k> dU_i/dx_k
Puu = -2 * uv * dUp
Puv = -vv * dUp
Puw = -uv * dWp - vw * dUp
Pvw = -ww * dWp
for nm, calc, col in [('uu', Puu, 10), ('uv', Puv, 42), ('uw', Puw, 50), ('vw', Pvw, 57)]:
    data = d[:, col]
    print(nm, 'produc corr =', round(float(np.corrcoef(calc, data)[0, 1]), 4),
          ' ratio med =', round(float(np.median(calc / (data + 1e-30))), 4))

# rotation term columns (block offsets: produc=0,rot=1,...): uu@11, vv@19, ww@27, uv@35, uw@43, vw@51
for nm, col in [('uu', 11), ('vv', 19), ('ww', 27), ('uv', 35), ('uw', 43), ('vw', 51)]:
    print('rot_%s range' % nm, round(float(d[:, col].min()), 4), round(float(d[:, col].max()), 4))

# Coriolis term candidates for streamwise rotation Omega along x1:
# G_ij = -2*Omega_1*( <u_i u_m> eps_{j1m} + <u_j u_m> eps_{i1m} )
Omega_p = 30.0 / 360.0     # Omega+ = Ro_tau/(2 h+)
def eps(a, b, c):
    return int((a - b) * (b - c) * (c - a) / 2)
R = np.zeros((y.size, 3, 3))
R[:, 0, 0] = uu; R[:, 1, 1] = vv; R[:, 2, 2] = ww
R[:, 0, 1] = R[:, 1, 0] = uv; R[:, 0, 2] = R[:, 2, 0] = uw; R[:, 1, 2] = R[:, 2, 1] = vw
G = np.zeros_like(R)
for i in range(3):
    for j in range(3):
        s = np.zeros(y.size)
        for m in range(3):
            e_j1m = [eps(j, 0, m) if j != 0 and m != 0 else 0][0]
            e_i1m = [eps(i, 0, m) if i != 0 and m != 0 else 0][0]
            s += R[:, m, j] * e_j1m + R[:, m, i] * e_i1m
        G[:, i, j] = -2 * Omega_p * s
pairs = {'uu': (0, 0, 11), 'vv': (1, 1, 19), 'ww': (2, 2, 27),
         'uv': (0, 1, 35), 'uw': (0, 2, 43), 'vw': (1, 2, 51)}
for nm, (i, j, col) in pairs.items():
    data = d[:, col]
    print('rot_%s: G(+) corr=' % nm, round(float(np.corrcoef(G[:, i, j], data)[0, 1]), 4),
          ' G(-Omega) corr=', round(float(np.corrcoef(-G[:, i, j], data)[0, 1]), 4),
          ' data max|.|=', round(float(np.max(np.abs(data))), 4))

# dissipation: eps = ? normalization of dissip columns
ed = d[:, 13] + d[:, 21] + d[:, 29]
k = d[:, 9]
print('dissip sum/2: min', round(float(0.5 * ed.min()), 5),
      ' eps/k center', round(float(0.5 * ed[64] / k[64]), 4),
      ' idx20', round(float(0.5 * ed[20] / k[20]), 4),
      ' idx5', round(float(0.5 * ed[5] / k[5]), 4))
print('k+ center', k[64], 'uu center', uu[64])
