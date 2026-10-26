import json, glob, os
os.chdir(r"d:\AI\LLM-Feynman-G")
arms = {
  "mm_bare_go_T5g":       r"outputs\asm\runs_v3_mm_bare_go\B_a_T5g_final.json",
  "mm_bare_mg_T2v":       r"outputs\asm\runs_v3_mm_bare_mg\B_a_T2v_final.json",
  "heur_bare_mg_T2v":     r"outputs\asm\runs_v3_heur_bare_mg\B_a_T2v_final.json",
  "heur_full_mg_T2v":     r"outputs\asm\runs_v3_heur_full_mg\B_a_T2v_final.json",
  "mm_full_mg_T2v":       r"outputs\asm\runs_v3_mm_full_mg\B_a_T2v_final.json",
  "heur_full_go_T5g":     r"outputs\asm\runs_v3_heur_full_go\B_a_T5g_final.json",
  "mm_bare_mg_T3v":       r"outputs\asm\runs_v3_mm_bare_mg\B_a_T3v_final.json",
  "heur_bare_mg_T3v":     r"outputs\asm\runs_v3_heur_bare_mg\B_a_T3v_final.json",
  "heur_bare_mg_T4v":     r"outputs\asm\runs_v3_heur_bare_mg\B_a_T4v_final.json",
  "heur_bare_mg_T1":      r"outputs\asm\runs_v3_heur_bare_mg\B_a_T1_final.json",
  "mm_bare_mg_T1":        r"outputs\asm\runs_v3_mm_bare_mg\B_a_T1_final.json",
}
for k, p in arms.items():
    if not os.path.exists(p):
        print(f"=== {k}: MISSING")
        continue
    d = json.load(open(p, encoding="utf-8"))
    b = d["best"]
    cfg = d.get("cfg", {})
    print(f"=== {k}  loss={b['loss']:.4f}  fit={cfg.get('fitness','?')}  forbid={cfg.get('forbid_map','')}  ginit_only={cfg.get('ginit_only',False)}")
    print(f"  dsl : {b['dsl']}")
    if len(b['a']) > 0:
        nz = [(i, round(x,4)) for i, x in enumerate(b["a"]) if abs(x) > 1e-6]
        print(f"  a   : {nz}")
    coefs = b.get("coefs", {})
    for tt, vv in coefs.items():
        print(f"  Ro{int(tt.replace('Ro',''))}: NMSE={vv.get('NMSE','?'):.4f}  rmse={vv.get('rmse','?'):.4f}")
    print()
