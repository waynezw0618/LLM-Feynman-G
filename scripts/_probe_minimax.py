"""查 minimax 跑批状态 (避免 PS $ 转义问题)."""
import os, time, ctypes, ctypes.wintypes, json
from subprocess import run

PID = 76432
LOG = r"D:\AI\LLM-Feynman-G\outputs\asm\runs_minimax_pope_gsonly_stdout.log"
ERR = r"D:\AI\LLM-Feynman-G\outputs\asm\runs_minimax_pope_gsonly_stderr.log"
DIR = r"D:\AI\LLM-Feynman-G\outputs\asm\runs_minimax_pope_gsonly"

# 1) PID 还活着吗 (kernel32 OpenProcess)
PROCESS_QUERY_LIMITED = 0x1000
h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED, False, PID)
alive = bool(h)
if h: ctypes.windll.kernel32.CloseHandle(h)

print(f"[t={time.strftime('%H:%M:%S')}] pid={PID} alive={alive}")

# 2) 输出目录文件
if os.path.isdir(DIR):
    items = []
    for n in sorted(os.listdir(DIR)):
        p = os.path.join(DIR, n)
        if os.path.isfile(p):
            sz = os.path.getsize(p)
            mt = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(p)))
            items.append((n, sz, mt))
        elif os.path.isdir(p):
            mt = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(p)))
            items.append((n + "/", "-", mt))
    print(f"DIR {DIR}:")
    for n, sz, mt in items:
        print(f"  {n:35s} {str(sz):>10s}  mtime={mt}")
else:
    print(f"DIR {DIR} NOT EXISTS")

# 3) stdout / stderr 末尾
for tag, path in [("STDOUT", LOG), ("STDERR", ERR)]:
    if not os.path.exists(path):
        print(f"{tag}: (no log)")
        continue
    sz = os.path.getsize(path)
    with open(path, "rb") as f:
        data = f.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    print(f"{tag} sz={sz} lines={len(lines)}")
    for ln in lines[-15:]:
        print(f"  | {ln}")
