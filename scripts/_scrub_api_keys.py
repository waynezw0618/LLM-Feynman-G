# -*- coding: utf-8 -*-
"""批量脱敏: 把 outputs/asm/runs_*/ 下所有 *_final.json 中明文的 llm_api_key
替换为 ***masked*** (只动 'llm_api_key' 这一个键的值, 其余内容原样保留).

用法: python -X utf8 scripts/_scrub_api_keys.py [--dry]
"""
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATTERN = re.compile(r'("llm_api_key"\s*:\s*")([^"\\]+)(")')

dry = "--dry" in sys.argv
files = sorted(glob.glob(os.path.join(REPO, "outputs", "asm", "runs_*", "*_final.json")))
n_hit = 0
for fp in files:
    with open(fp, encoding="utf-8") as f:
        txt = f.read()
    new, n = PATTERN.subn(lambda m: m.group(1) + "***masked***" + m.group(3)
                          if m.group(2) not in ("", "EMPTY", "None", "***masked***")
                          else m.group(0), txt)
    if n and new != txt:
        n_hit += 1
        if dry:
            print(f"[dry] {fp}: {n} 处明文 key")
        else:
            with open(fp, "w", encoding="utf-8") as f:
                f.write(new)
            print(f"[scrub] {os.path.relpath(fp, REPO)}: {n} 处 -> ***masked***")
print(f"共扫描 {len(files)} 个 final.json, 处理 {n_hit} 个 ({'dry-run' if dry else '已写盘'})")
