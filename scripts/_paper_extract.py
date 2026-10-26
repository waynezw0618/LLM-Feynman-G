# -*- coding: utf-8 -*-
"""Extract text of the JCP paper, locate Section 5.2 evaluation methodology."""
import sys
from pathlib import Path
from pypdf import PdfReader

PDF = Path(r"d:\AI\LLM-Feynman-G\1-s2.0-S0021999116303643-main.pdf")
OUT = Path(r"d:\AI\LLM-Feynman-G\outputs\asm\_paper_text.txt")

r = PdfReader(str(PDF))
print("pages:", len(r.pages))
text = "\n\n".join((p.extract_text() or "") for p in r.pages)
OUT.write_text(text, encoding="utf-8")
print("written:", OUT, len(text), "chars")

# print TOC-ish lines and locate section 5.2
import re
lines = text.splitlines()
for i, ln in enumerate(lines):
    s = ln.strip()
    if re.match(r"^(\d+(\.\d+)*)\.?\s+[A-Z]", s) and len(s) < 80:
        print(f"[{i}] {s}")
