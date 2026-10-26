shipman 论文材料包 (paper_bundle_20261015)
==========================================

内容:
  main.tex      — 中文 arXiv 风格论文源码 (ctexart, xelatex 编译)
  figs/*.png    — 论文全部插图 (300 dpi)
  main.pdf      — 编译好的 PDF
  eval_summary.json — 实验数字溯源 (每个 run 的系数恢复/重积分/硬规则)

编译方法:
  xelatex main.tex   (运行两遍以稳定交叉引用)
  需要 TeX 发行版含 ctex 宏包 (MiKTeX/TeX Live 均自带).

复现实验:
  cd d:/AI/LLM-Feynman-G
  python -X utf8 shipman/rc_sm_prep.py          # 数据
  powershell -ExecutionPolicy Bypass -File runner_sm.ps1   # 23 runs
  python -X utf8 shipman/rc_sm_eval.py          # 评估
  python -X utf8 shipman/rc_sm_figs.py          # 出图
  python -X utf8 shipman/rc_sm_paperfill.py     # 论文填数

数据基准: KVLCC2.L7 (Yasukawa & Yoshimura 2015, J Mar Sci Technol 20:37-52,
doi:10.1007/s00773-014-0293-y), 全套系数取自原文 Table 1/3.
