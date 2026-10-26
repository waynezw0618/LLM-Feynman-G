# -*- coding: utf-8 -*-
"""Word -> PDF (调用本机 Word COM)."""
import os, sys
import win32com.client

src = r"d:\AI\LLM-Feynman-G\docs\LGF_融合技术报告.docx"
dst = r"d:\AI\LLM-Feynman-G\docs\LGF_融合技术报告.pdf"

if not os.path.exists(src):
    print("missing:", src); sys.exit(1)

word = win32com.client.DispatchEx("Word.Application")
word.Visible = False
word.DisplayAlerts = 0
try:
    doc = word.Documents.Open(src, ReadOnly=False)
    # 更新目录/域(若有)与分页
    try:
        doc.Fields.Update()
    except Exception:
        pass
    # 17 = wdExportFormatPDF
    doc.SaveAs(dst, FileFormat=17)
    doc.Close(False)
    print("saved", dst, os.path.getsize(dst), "bytes")
finally:
    word.Quit()
