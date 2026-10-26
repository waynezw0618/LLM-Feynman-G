"""临时 runner: 逐个跑测试文件并打印 ASCII 摘要 (用完即删)."""
import os, subprocess, sys

FILES = [
    "tests/smoke_graph_dsl.py",
    "tests/test_lgf.py",
    "tests/test_eq_compat.py",
    "tests/test_llm_config.py",
]
KEYS = ("PASS=", "FAIL=", "TOTAL=", "ALL TESTS", "FAILED:", "OK)", "FAILED)", "Traceback")

env = dict(os.environ, PYTHONIOENCODING="utf-8")
all_ok = True
for f in FILES:
    p = subprocess.run([sys.executable, "-W", "ignore", f],
                       capture_output=True, text=True, encoding="utf-8",
                       errors="replace", env=env)
    lines = [l.strip() for l in p.stdout.splitlines()]
    hits = [l for l in lines if any(k in l for k in KEYS)]
    ok = p.returncode == 0 and not any("Traceback" in l for l in hits)
    all_ok &= ok
    print(f"=== {f}  exit={p.returncode}  {'OK' if ok else 'FAILED'}")
    for l in hits[-3:]:
        print("    ", l)
    if p.stderr.strip():
        print("     [stderr]", p.stderr.strip().splitlines()[-1][:160])

print("\nSUMMARY:", "ALL GREEN" if all_ok else "SOME FAILED")
sys.exit(0 if all_ok else 1)
