# -*- coding: utf-8 -*-
"""v3 8 臂全景: loss 矩阵 + M_GUIDE 统计 + G_INIT 解析率."""
import glob, json, os

TGT = ["T1", "T2v", "T3v", "T4v", "T5g"]
ARMS = ["v3_heur_bare_mg", "v3_mm_bare_mg", "v3_heur_bare_go", "v3_mm_bare_go",
        "v3_heur_full_mg", "v3_mm_full_mg", "v3_heur_full_go", "v3_mm_full_go"]
ROOT = r"outputs\asm"


def load(p):
    d = json.load(open(p, encoding="utf-8"))
    return d


def find(arm, tgt):
    for p in glob.glob(os.path.join(ROOT, f"runs_{arm}", "*_final.json")):
        if os.path.basename(p).endswith(f"_{tgt}_final.json"):
            return p
    return None


print("=" * 118)
print("V1  loss 矩阵 (best.loss, 系数空间 NMSE 均值) + M_GUIDE 统计")
print("=" * 118)
hdr = f"{'arm':22s}" + "".join(f"{t:>10s}" for t in TGT) + f"{'mean':>10s} | {'mg.call':>8s}{'mg.appl':>8s}{'mg.alive':>9s} | {'ginit kept/cand':>16s}"
print(hdr)
print("-" * 118)
mat = {}
for arm in ARMS:
    losses, mgc, mga, mgl, gk, gc = [], 0, 0, 0, 0, 0
    for t in TGT:
        p = find(arm, t)
        if not p:
            losses.append(None); continue
        d = load(p)
        losses.append(d["best"]["loss"])
        st = d.get("llm_stats", {})
        mg = st.get("mguide", {}); gi = st.get("ginit", {})
        mgc += mg.get("n_calls", 0); mga += mg.get("n_applied", 0); mgl += mg.get("n_alive", 0)
        gk += gi.get("n_parsed", 0); gc += gi.get("n_candidates", 0)
    mat[arm] = losses
    vals = [x for x in losses if x is not None]
    mean = sum(vals) / len(vals) if vals else float("nan")
    row = f"{arm:22s}" + "".join((f"{x:>10.4f}" if x is not None else f"{'-':>10s}") for x in losses)
    row += f"{mean:>10.4f} | {mgc:>8d}{mga:>8d}{mgl:>9d} | {gk:>7d}/{gc:<8d}"
    print(row)

print()
print("=" * 118)
print("V2  主判据 (配对对比)")
print("=" * 118)


def mean(v):
    vv = [x for x in v if x is not None]
    return sum(vv) / len(vv) if vv else float("nan")


def cmp_pair(a, b, label):
    ma, mb = mean(mat[a]), mean(mat[b])
    print(f"  {label}")
    print(f"    {a:22s} = {ma:.4f}")
    print(f"    {b:22s} = {mb:.4f}")
    print(f"    delta(b-a) = {mb-ma:+.4f} ({(mb-ma)/ma*100:+.1f}%)" if ma == ma and mb == mb else "  n/a")
    print()
    for t in TGT:
        ia = TGT.index(t)
        xa, xb = mat[a][ia], mat[b][ia]
        if xa is not None and xb is not None and xa > 1e-12:
            print(f"    {t:4s}: {xa:.4f} -> {xb:.4f}  ({(xb-xa)/xa*100:+.1f}%)")
    print()


cmp_pair("v3_heur_bare_mg", "v3_mm_bare_mg", "裸 + M_GUIDE: heur vs M2.7")
cmp_pair("v3_heur_bare_go", "v3_mm_bare_go", "裸 + G_INIT-only: heur vs M2.7")
cmp_pair("v3_heur_full_mg", "v3_mm_full_mg", "full + M_GUIDE: heur vs M2.7")
cmp_pair("v3_heur_full_go", "v3_mm_full_go", "full + G_INIT-only: heur vs M2.7")

print("=" * 118)
print("V3  M_GUIDE 净效应 (同后端 mg vs go)")
print("=" * 118)
for be in ["heur", "mm"]:
    for cfg in ["bare", "full"]:
        a, b = f"v3_{be}_{cfg}_mg", f"v3_{be}_{cfg}_go"
        print(f"  {be:5s} {cfg:5s}: mg={mean(mat[a]):.4f}  go={mean(mat[b]):.4f}  "
              f"delta(mg-go)={mean(mat[a])-mean(mat[b]):+.4f}")
