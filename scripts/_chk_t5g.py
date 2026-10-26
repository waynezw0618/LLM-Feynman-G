import numpy as np
d3 = np.load(r'd:\AI\LLM-Feynman-G\outputs\asm\dataset_pope3_wy.npz')
dw = np.load(r'd:\AI\LLM-Feynman-G\outputs\asm\dataset_pope_wy.npz')
print('pope3 keys feat:', [str(x) for x in d3['featB_keys']])
print('pope3 sel_basis:', [str(x) for x in d3['sel_basis']])
print('pope3 has B_T5g:', 'c0_B_T5g' in d3.files)
print('pope_wy keys feat:', [str(x) for x in dw['featB_keys']])
print('pope_wy has B_T5g:', 'c0_B_T5g' in dw.files)
print('pope_wy has B_T5v:', 'c0_B_T5v' in dw.files)
# T5g 与 T5v 差异多大 (Ro30)
if 'c0_B_T5g' in d3.files:
    dv = d3['c2_B_T5v']; dg = d3['c2_B_T5g']
    print('T5g vs T5v corr (Ro30):', np.corrcoef(dv.ravel(), dg.ravel())[0, 1].round(4))
