// gen_asm_word_report.js — 生成 ASM 变体与混合训练对比 Word 报告
// 输出: lgf/sample/ASM_变体与混合训练对比报告.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageNumber, PageBreak,
} = require("docx");

const REPO = "d:\\AI\\LLM-Feynman-G";
const FIGS = path.join(REPO, "outputs", "asm", "figs_trueasm");
const OUTDIR = path.join(REPO, "lgf", "sample");
const OUTFILE = process.env.ASM_DOCX_OUT
  || path.join(OUTDIR, "ASM_变体与混合训练对比报告.docx");

const ZH = "Microsoft YaHei";
const CODE = "Consolas";

// ---------- helpers ----------
const t = (text, opts = {}) => new TextRun({ text, font: ZH, size: 21, ...opts });
const bold = (text) => t(text, { bold: true });
const h1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: ZH, bold: true })] });
const h2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: ZH, bold: true })] });
const p = (runs, opts = {}) => new Paragraph({ spacing: { after: 120, line: 300 }, children: Array.isArray(runs) ? runs : [t(runs)], ...opts });
const bullet = (runs) => new Paragraph({ numbering: { reference: "bullets", level: 0 }, spacing: { after: 80, line: 300 }, children: Array.isArray(runs) ? runs : [t(runs)] });

function codeBlock(lines) {
  return lines.map((ln) => new Paragraph({
    spacing: { after: 0, line: 260 },
    shading: { fill: "F2F2F2", type: ShadingType.CLEAR },
    children: [new TextRun({ text: ln, font: CODE, size: 18 })],
  }));
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 60, after: 200 },
    children: [new TextRun({ text, font: ZH, size: 18, italics: true, color: "555555" })],
  });
}

function img(file, w, h, alt) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({
      type: "png",
      data: fs.readFileSync(path.join(FIGS, file)),
      transformation: { width: w, height: h },
      altText: { title: alt, description: alt, name: alt },
    })],
  });
}

const border = { style: BorderStyle.SINGLE, size: 1, color: "BBBBBB" };
const borders = { top: border, bottom: border, left: border, right: border };
const CW = 9360; // US Letter content width

function cell(text, w, opts = {}) {
  return new TableCell({
    borders, width: { size: w, type: WidthType.DXA },
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    shading: opts.head ? { fill: "D9E2F3", type: ShadingType.CLEAR } : undefined,
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.CENTER,
      children: [new TextRun({ text, font: opts.code ? CODE : ZH, size: opts.head ? 19 : 19, bold: !!opts.head, boldComplexScript: undefined })],
    })],
  });
}

function makeTable(headerCells, rows, widths) {
  return new Table({
    width: { size: CW, type: WidthType.DXA },
    columnWidths: widths,
    rows: [
      new TableRow({ tableHeader: true, children: headerCells.map((h, i) => cell(h, widths[i], { head: true })) }),
      ...rows.map((r) => new TableRow({
        children: r.map((c, i) => cell(String(c), widths[i], { align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })),
      })),
    ],
  });
}

// ---------- data ----------
const runsTable = [
  ["PySR v", "0.4189", "+0.538", "0.697", "0.589", "0.526", "0.455", "0.422"],
  ["LGF-mg v", "0.5571", "+0.567", "0.675", "0.629", "0.612", "0.498", "0.423"],
  ["LGF-go v", "0.5121", "+0.746", "0.747", "0.694", "0.740", "0.773", "0.777"],
  ["PySR g", "0.3736", "+0.606", "0.719", "0.658", "0.633", "0.539", "0.480"],
  ["LGF-mg g", "0.6082", "+0.478", "0.661", "0.593", "0.484", "0.399", "0.254"],
  ["LGF-go g", "0.3797", "+0.725", "0.781", "0.701", "0.683", "0.724", "0.735"],
  ["PySR e", "0.4666", "+0.481", "0.675", "0.634", "0.532", "0.329", "0.232"],
  ["LGF-mg e", "0.5584", "+0.160", "0.358", "0.259", "0.157", "0.031", "-0.004"],
  ["LGF-go e", "0.5366", "+0.380", "0.598", "0.495", "0.408", "0.285", "0.113"],
  ["MIX PySR", "0.3610", "+0.610", "0.798", "0.695", "0.617", "0.491", "0.449"],
  ["MIX LGF-go", "0.3378", "+0.819", "0.823", "0.827", "0.846", "0.819", "0.782"],
];

const equations = [
  {
    title: "变体 v（纯涡量 W = W̄）",
    items: [
      { name: "PySR", meta: "complexity=30 · loss_asm=0.4189 · ρ_a=+0.538 · 特征: T8, eta2, s2o2",
        code: ["b_ij = T8 * 0.2637855 * s2o2 * ( exp(-4.846445/eta2)",
               "         + exp(-4.846445*eta2) * s2o2 * 0.2637855 / s2o2 )",
               "        / (s2o2 * s2o2)"] },
      { name: "LGF-mg (MiniMax-M2.7)", meta: "edges=11 · loss_asm=0.5571 · ρ_a=+0.567 · 特征: T2, T7",
        code: ["b_ij = 4.7573e-05 * T7 - 1.00695 * T2 + 0.00507645"] },
      { name: "LGF-go (MiniMax-M2.7)", meta: "edges=15 · loss_asm=0.5121 · ρ_a=+0.746 · 特征: T1, T2, T3, T4",
        code: ["b_ij = 0.0058514*T4 - 0.00685299*T2 - 0.0208006*T1",
               "        + 0.00686077*T3 + 0.000140394*T2*T3"] },
    ],
  },
  {
    title: "变体 g（Gatski–Speziale: W = W̄ + Ω）",
    items: [
      { name: "PySR", meta: "complexity=29 · loss_asm=0.3736 · ρ_a=+0.606 · 特征: T4, T8, eta2, s3, s2o2",
        code: ["b_ij = ( eta2 * 0.7407941 * T4 / ( s2o2 * (2.0976007 - (s3 + 3.3073611))",
               "             * ( exp(s2o2) + exp(s3) - 3.3073611 ) )",
               "         - T8/s2o2 + s3 ) * (-0.33395973)"] },
      { name: "LGF-mg (MiniMax-M2.7)", meta: "edges=10 · loss_asm=0.6082 · ρ_a=+0.478 · 特征: T2, eta2",
        code: ["b_ij = -0.00190211 * tanh(0.0314002 * eta2) * T2",
               "        - 0.061674 * tanh(1.33846e+06 * T2)"] },
      { name: "LGF-go (MiniMax-M2.7)", meta: "edges=11 · loss_asm=0.3797 · ρ_a=+0.725 · 特征: T1, T2, eta2, zeta2",
        code: ["b_ij = -0.090244 * tanh(T1) - 0.234501 * 2.91675*T2 / (eta2 + zeta2)"] },
    ],
  },
  {
    title: "变体 e（扩展内禀: W = W̄ + 2Ω = ω_a）",
    items: [
      { name: "PySR", meta: "complexity=28 · loss_asm=0.4666 · ρ_a=+0.481 · 特征: T8, T9, s2o2",
        code: ["b_ij = ( 0.38392627*T8 - 0.57454098*T9 * sqrt(121.45341 - s2o2)",
               "             / (71.32061 - 0.12034881*s2o2) ) / s2o2"] },
      { name: "LGF-mg (MiniMax-M2.7)", meta: "edges=12 · loss_asm=0.5584 · ρ_a=+0.160 · 特征: T7, T8, s2o2",
        code: ["b_ij = -0.516839 * T8 / ( 0.272698*T7/(T7 - 1.41063) - 1.41592*s2o2 )"] },
      { name: "LGF-go (MiniMax-M2.7)", meta: "edges=7 · loss_asm=0.5366 · ρ_a=+0.380 · 特征: T2, zeta2",
        code: ["b_ij = -0.400176 * T2 / ( 0.0118046*T2 + zeta2 )"] },
    ],
  },
  {
    title: "混合训练 MIX（43 维特征：T1/T3 共享 + T2/T4..T10 三变体分裂 + 5 不变量×3 + ret/roloc）",
    items: [
      { name: "MIX PySR", meta: "complexity=11 · loss_asm=0.3610 · ρ_a=+0.610 · 通道使用: v=1, g=1, e=2",
        code: ["b_ij = T2_v * 2.4634178 / ( (T5_e - eta2_e)/0.1575619 - zeta2_g )"] },
      { name: "MIX LGF-go (MiniMax-M2.7) — 新 SOTA", meta: "edges=13 · loss_asm=0.3378 · ρ_a=+0.819 · 通道使用: v=3, g=0, e=0",
        code: ["b_ij = -0.29577 * tanh(0.09563 * T2_v)",
               "        + 0.04726 * tanh(33.964 * T6_v)",
               "        + 0.06915 * tanh(6.0287 * T10_v)"] },
    ],
  },
];

// ---------- document ----------
const children = [];

// 标题
children.push(new Paragraph({ spacing: { before: 2400, after: 240 }, alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "旋转槽道流代数应力模型（ASM）符号回归对比报告", font: ZH, bold: true, size: 44 })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "涡量变体 v（W̄）/ g（W̄+Ω）/ e（W̄+2Ω）与混合训练 MIX", font: ZH, size: 26, color: "444444" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 60 },
  children: [new TextRun({ text: "PySR vs LGF（MiniMax-M2.7 后端）· 11 组 run 对比", font: ZH, size: 22, color: "444444" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 },
  children: [new TextRun({ text: "数据: Yang & Wang (2018 JFM 838) DNS · Re_τ = 180 · Ro_τ = 7.5 / 15 / 30 / 75 / 150", font: ZH, size: 20, color: "666666" })] }));
children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 2400 },
  children: [new TextRun({ text: "2026-10-09 · LLM-Feynman-G 项目", font: ZH, size: 20, color: "666666" })] }));
children.push(new Paragraph({ children: [new PageBreak()] }));

// 1 任务设定
children.push(h1("1  任务设定"));
children.push(p([bold("框架："), t("Weatheritt & Sandberg (2016 JCP 325:22-37) eq.(19)/(20) 真 ASM——单一闭式标量函数对全部 6 个各向异性分量 × 5 个旋转工况统一，仅允许全局常数 C#（无逐工况参数 K#）：")]));
children.push(...codeBlock([
  "b_ij = f( T1_ij..T10_ij, eta2, zeta2, s3, so2, s2o2, ret, roloc )",
  "  T1..T10 : Pope (1975) 十个对称无迹张量基 (S, W 已乘 k/eps 归一化)",
  "  eta2=tr(S^2), zeta2=tr(W^2), s3=tr(S^3), so2=tr(S W^2), s2o2=tr(S^2 W^2)",
  "  ret = k^2/(nu*eps)          (湍流雷诺数)",
  "  roloc = 2*Om_eff*k/eps      (带符号局部旋转数)",
]));
children.push(p("三个涡量变体与混合训练的特征空间定义："));
children.push(makeTable(
  ["设定", "W 的构造", "特征维数", "说明"],
  [
    ["v 纯涡量", "W = W̄", "17", "帧旋转仅经 roloc 标量进入；经典 RANS 闭包量"],
    ["g GS", "W = W̄ + 1·Ω", "17", "Gatski–Speziale (1993) 形式"],
    ["e 扩展内禀", "W = W̄ + 2·Ω = ω_a", "17", "绝对涡量张量"],
    ["MIX 混合", "T1/T3 共享 + T2,T4..T10 三变体分裂 + 不变量×3", "43", "让算法在 v/g/e 三通道间自由选择"],
  ],
  [1400, 3600, 1000, 3360]
));
children.push(p([bold("评估指标："), t("loss_asm（6 分量 × 5 工况联合 NMSE）与 ρ_a,case（W&S eq.22 逐点 3×3 张量余弦，对有效壁面点平均）。参考基线 cc10（10 基全局常数）：v 0.696 / g 0.798 / e 0.840。")]));

// 2 主对比表
children.push(h1("2  主对比表（11 组 run）"));
children.push(makeTable(
  ["run", "loss_asm ↓", "ρ_a ↑", "Ro7.5", "Ro15", "Ro30", "Ro75", "Ro150"],
  runsTable,
  [1600, 1100, 1100, 1112, 1112, 1112, 1112, 1112]
));
children.push(p([bold("读表要点："), t("① 每个变体内 LGF-go 全面优于 LGF-mg（loss 与 ρ 双赢）——M_GUIDE 对 MiniMax-M2.7 后端结构性损害、mm 后端必配 ginit_only 的结论在真 ASM 任务复现；② 变体排序 ρ：v > g > e（LGF-go 列 0.746 / 0.725 / 0.380）；③ MIX LGF-go 同时刷新 loss 下限（0.338）与 ρ 上限（0.819），且逐工况 ρ 全部 ≥ 0.78，高 Ro 不衰减。")]));

// 3 全部最优方程
children.push(h1("3  全部最优方程（常数已代入拟合值）"));
for (const grp of equations) {
  children.push(h2(grp.title));
  for (const it of grp.items) {
    children.push(p([bold(it.name + "："), new TextRun({ text: it.meta, font: ZH, size: 18, color: "666666" })]));
    children.push(...codeBlock(it.code));
    children.push(p("", { spacing: { after: 40 } }));
  }
}
children.push(p([bold("公式说明："), t("全部 11 条 = 单公式同时覆盖 6 分量 × 5 工况，常数一律跨工况全局。LGF 公式为 final refit 后的 DSL 代入 θ 值；PySR 公式为 Pareto 前沿最优复杂度点。ret / roloc 没有被任何一条最优公式选用（负面结果，见 §7）。")]));

// 4 剖面比较
children.push(h1("4  剖面比较（b_ij 六分量 × 5 工况）"));
children.push(p("曲线：DNS（黑，粗）/ PySR（蓝）/ LGF-mg（橙）/ LGF-go（红）。"));
children.push(img("fig_profile_v.png", 620, 512, "v 变体剖面"));
children.push(caption("图 1  变体 v：b_ij 剖面对比。b12（剪切）LGF-go 红线几乎完美贴合 DNS 的 S 形；b13/b23 近壁双峰各方法只能抓形状、幅度不足。"));
children.push(img("fig_profile_g.png", 620, 512, "g 变体剖面"));
children.push(caption("图 2  变体 g：b_ij 剖面对比。LGF-mg 橙线近壁出现 tanh(1.34e6·T2) 饱和阶跃造成的方波/爆值台阶——mg 臂差的可视化证据。"));
children.push(img("fig_profile_e.png", 620, 512, "e 变体剖面"));
children.push(caption("图 3  变体 e：b_ij 剖面对比。全方法最差，LGF-go 公式 T2/(0.012·T2+zeta2) 在高 Ro 幅度不足。"));

// 5 迭代收敛
children.push(h1("5  迭代收敛过程"));
children.push(p([bold("口径声明："), t("LGF 逐代日志只保存 best_graph（无 θ），每个去重后的逐代最优图用与 final refit 相同设置（8 启动 / 150 迭代）在全量行重新拟合后评估，保证与最终模型同一口径；粗线 = best-so-far 包络，细淡线 = 逐代原值，星号 = final refit 模型。PySR 未保存逐迭代历史，以复杂度阶梯（Pareto 前沿）代替。")]));
children.push(img("fig_iter_lgf.png", 620, 228, "LGF 迭代"));
children.push(caption("图 4  LGF 六 run 逐代 loss_asm（log 轴）+ mean ρ_a；虚线点线为各变体 cc10 全局常数基线。"));
children.push(img("fig_iter_lgf_rho_cases.png", 500, 478, "LGF 逐工况 rho"));
children.push(caption("图 5  3×2 分面：每 run 逐代 ρ_a,case（5 工况 viridis + 黑 mean）。g-go / v-go 的高 Ro 曲线随代际上抬并稳住 0.7+；e-mg 的 Ro=150 曲线长期贴 0。"));
children.push(img("fig_iter_pysr.png", 620, 331, "PySR Pareto"));
children.push(caption("图 6  PySR Pareto 前沿：横轴 complexity，上 loss_asm（semilogy）、下 ρ_a（含 5 工况散点），金星 = best。"));

// 6 混合训练
children.push(h1("6  混合训练（MIX）与假设验证"));
children.push(p("把 v/g/e 三套张量基与不变量混合为 43 维特征（T1/T3 共享 + T2/T4..T10 三变体分裂 + 5 不变量×3 + ret/roloc），PySR 与 LGF-go 各跑一轮，验证假设：v 的 W̄ 是 RANS 教材的母语量；g/e 把 Ω 塞进 W̄ 是冗余表示。"));
children.push(img("fig_mix_compare.png", 460, 344, "mix 对比"));
children.push(caption("图 7  11 run 的 loss–ρ 散点对比；MIX LGF-go（★）双指标登顶。"));
children.push(img("fig_mix_percase_rho.png", 540, 293, "mix 逐工况 rho"));
children.push(caption("图 8  逐工况 ρ_a：MIX LGF-go 全工况 0.78+ 平稳；MIX PySR 与单变体 PySR 一样随 Ro 单调衰减（0.80→0.45）。"));
children.push(img("fig_profile_mix.png", 620, 512, "mix 剖面"));
children.push(caption("图 9  MIX 剖面对比（6 分量 × 5 工况；DNS 黑 / mix_pysr 蓝 / mix_lgf_go 红）。LGF-go 红线在 b12 全工况贴合 DNS 且随 Ro 增大更好（NMSE 0.167→0.069）；b23 红线呈“振荡 + 近壁尖峰”，丢了真值的正偏置与中心单峰（真值为全 y 恒正的偶对称单峰）；b33 高 Ro 幅度衰减。PySR 蓝线在 b12/b13/b23 全部工况塌缩为 0（NMSE=1.000），仅学会 b11/b22。"));
children.push(p([bold("通道使用统计（真实 token 次数；原 v=36/g=0/e=0 与 12/12/24 系计数 bug，已修）：")]));
children.push(makeTable(
  ["run", "v 通道", "g 通道", "e 通道", "共享", "结论"],
  [
    ["MIX LGF-go", "3", "0", "0", "0", "最终公式只含 v 通道特征"],
    ["MIX PySR", "1", "1", "2", "0", "三通道混用"],
  ],
  [1700, 1100, 1100, 1100, 1000, 3360]
));
children.push(p([t("MIX LGF-go 的最终公式只含 v 通道特征（"), bold("T2_v / T6_v / T10_v；v=3, g=0, e=0"), t("）。"), bold("重要修正："), t("该公式来自 G_INIT 一次生成（逐代日志显示 best_graph 从 gen0 到 gen39 一字未变，且 llm_ginit_only=True、M_GUIDE 全程 0 调用），"), bold("不是 GP 逐代淘汰 g/e 的达尔文选择结果"), t("。可以说的只是“同时用 g/e 特征没有产生更好公式”。此外特征名列表中每组三元组 _v 恒排首位，存在 LLM 位置偏好混淆，需“打乱特征名顺序”的对照实验才能排除（本次未做）。MIX LGF-go 公式形态从单变体的“5 基线性”升级为“3-tanh 调制求和”，ρ 从 0.746 升至 0.819。")]));

// 7 逐分量分析 b12 vs b23
children.push(h1("7  逐分量分析：为什么 b12 与 b23 差异大"));
children.push(p([t("b12/b23 的差异存在于两个层面，必须分开看：")]));
children.push(bullet("真值层（差 2.4–4.2 倍）：b12 = −uv̄/k 是主剪切通道，由 dU/dy 直接驱动，全 y 向都有强信号，且为奇对称 S 形、零均值（std≈RMS），RMS 0.092–0.109；b23 = −vw̄/k 是二次流通道，剖面为偶对称单峰（yh=0 处 0.0375、两壁面约 0.001）、全 y 恒正、偏置主导（mean +0.0196~+0.0389，std 0.0085–0.0188，mean/std=1.6–2.9），RMS 仅 0.023–0.043，且随 Ro 非单调（Ro30 最大）。旋转 Coriolis 项 2Ω(0,−W,V) 在 V–W 平面激发二次流 → b23 的正偏置与中心峰即其产物。"));
children.push(bullet("模型层（差约 5.6 倍）：mix_lgf_go 在 b12 上 NMSE=0.110 / ρ=+0.944，在 b23 上 NMSE=0.612 / ρ=+0.452。分量难度（NMSE 升序）b12 < b11 < b22 < b23 < b33 < b13(=1.0 退化)。"));
children.push(p([bold("机制（修正版）："), t("首要机制是损失加权与幅值尺度，而非非局部输运。① 一个公式 g 同时覆盖 6 分量 × 5 工况行、共享同一组全局常数，优化目标是全局标量 loss_asm，由 b11/b22（RMS 0.3–0.5）等大幅值分量主导；② b23 的 RMS 仅为 b11 的 1/10 量级，在共享损失里几乎无权重；③ 结果模型给 b23 输出“偶对称振荡 + 近壁尖峰”——std 反比真值大（0.019–0.022 vs 0.009–0.019）、正偏置只抓到 1/3~1/2（0.009–0.014 vs 0.020–0.039）、近壁峰 0.026–0.059（真值约 0.001）；而 b12 的 std 精确对齐（0.092/0.104/0.108 → 0.101/0.103/0.094）、误差 0.028–0.038 ≪ RMS 0.09–0.11。")]));
children.push(p([bold("可检验推论："), t("若该机制成立，逐分量加权 loss（按 1/RMS² 归一化）或 b23 独立拟合应显著压低 b23 NMSE；若加权后 b23 仍差，才回到“Pope 局部基不含非局部输运（压力应变/TG 涡对流输运）”的更强解释——该解释本次未被证实，仅作备择假设。b13 (uw) 数据随 Ro 增大衰减至近零（RMS 0.072→0.029），模型输出 0 → NMSE=1.0，属数据退化非模型失败。")]));
children.push(p([bold("PySR 对照："), t("同一 43 维空间下，PySR 在 b12 上 NMSE=1.000（预测塌缩为 0），只学会 b11 (0.179) 与 b22 (0.066)；其 ρ_a 仅 0.610。LGF 纯 v 公式在 b12 上 NMSE=0.110 —— 这是两者的分水岭。")]));
children.push(img("fig_profile_mix_b12b23.png", 620, 224, "mix b12/b23 剖面"));
children.push(caption("图 10  MIX 剖面聚焦：b12（上排，主剪切，奇对称 S 形、过零）vs b23（下排，二次流，偶对称单峰、全 y 恒正）。每格标注该工况真值 RMS 与两模型 NMSE。b12：LGF-go 红线几乎完整复现 DNS 主信号，仅近壁略微过冲，Ro 越高越好。b23：真值是平滑正偏置 + 中心峰，LGF-go 红线给出的是围绕零的振荡与近壁尖峰（偏置只抓到 1/3~1/2、std 反超真值）——这正是共享损失中弱分量无权重导致的结构性失真。"));

// 8 物理解读
children.push(h1("8  物理解读：为何 v（纯涡量）最优"));
children.push(p([bold("(1) S̄ 改写链条——旋转信息如何进入模型。"), t("沿 x₁ 旋转时 Coriolis 项 = 2Ω(0, −U₃, U₂) 出现在 RANS 动量方程中，它改写 U(y) 剖面（稳定壁 log 层变薄、不稳定壁增厚；高 Ro 出现 W≠0），因此 dU/dy 的幅值与符号都是 Ro 的函数。S̄_ij = 0.5(∂Ū_i/∂x_j + ∂Ū_j/∂x_i) 跨工况真的不一样——三个变体的 S 完全相同，SR 算法拿到 S̄ 张量基的数值即可“看到” Ro 差异。")]));
children.push(p([bold("(2) roloc / ret 的负面结果。"), t("两个标量都是候选特征，但没有被任何一条最优公式选用——所谓“两个独立旋转通道”只在候选层面成立，有效层面只有 S̄ 一条。跨工况差异被 eta2/s2o2 等不变量自身吸收。")]));
children.push(p([bold("(3) v 胜出的机制。"), t("三个变体的差别只在张量基的 W：v 用 W̄（标准 RANS 闭包量、母语量），g 用 W̄+Ω（无标准名字的中间量），e 用 W̄+2Ω = ω_a（绝对涡量，但在 Ro=7.5 几乎退化为常数、被算法误用为可学习强度）。g/e 把“几何 W̄”和“外加 Ω”混进同一个张量基，没有增加新信号（新信号在 S̄ 里已经在了），但结构变模糊——算法为解耦被迫学出额外非线性（g-go 出现 2.92·T2/(eta2+zeta2)，e-go 出现 T2/(0.012·T2+zeta2)）。v 的 W̄ 让符号回归找 f(T1..T10) 的结构最自然：v-go 最优公式就是五基线性组合。")]));
children.push(p([bold("(4) MIX 实验定论（修订版）。"), t("v 看到的不是“更多旋转信号”，而是“结构最干净的旋转信号”——S̄ 负责跨工况差异，W̄ 负责张量基几何，两条互不混淆。MIX 中纯 v 公式未被 g/e 公式击败，与“g/e 信息冗余”相容；但"), bold("该结论不能表述为“算法主动拒绝 g/e”"), t("——公式源自 G_INIT 单次生成，且未排除位置偏好混淆（见 §6 通道使用统计说明）。")]));

// 8 方法学声明
children.push(h1("9  方法学声明与局限"));
children.push(bullet("复杂度口径不同：PySR 按节点数，LGF 按边数；跨方法比较以 loss/ρ 为准。"));
children.push(bullet("单 seed：11 组 run 均未做 cross-seed 验证（MIX LGF-go ρ=0.819 为单 seed；C4=33.96 偏大待验证）。"));
children.push(bullet("无 hold-out：与 W&S DNS 拟合设定一致；高 Ro 泛化结论基于逐工况 ρ 而非留出工况。"));
children.push(bullet("算力不对等：PySR Julia 多线程约 40 s vs LGF Python 约 800 s（MIX 43 min）。"));
children.push(bullet("s3/so2 在本二维平均流恒为零，却被 g 变体 PySR 公式当常数发生器使用（s3+3.31）；迁移三维平均流前需人工审查。"));
children.push(bullet("vw/b23 分量仍最差：单变体（Pope 1-formula 共享口径）逐工况 NMSE 2.2–6.1；MIX 口径下 b23 逐工况 NMSE 0.53–0.78。首要机制为“单公式共享 6 分量 + 幅值尺度悬殊 → 弱分量在全局损失中无权重”（见 §7 失效形态表），非局部输运仅为备择假设（未证实）。可执行的下一步：逐分量加权 loss / b23 独立拟合对照。"));
children.push(bullet("MIX LGF-go 的逐工况 uw NMSE 恒为 1.0：二维平均流下 uw 分量 DNS 数据退化，非模型问题。"));

// 9 文件清单
children.push(h1("10  文件清单"));
children.push(bullet([t("数据集："), new TextRun({ text: "outputs/asm/dataset_trueasm_{v,g,e,mix}.npz", font: CODE, size: 18 })]));
children.push(bullet([t("结果："), new TextRun({ text: "outputs/asm/runs_trueasm_pysr/{v,g,e}/ · runs_trueasm_lgf/{v,g,e}_{mg,go}/ · runs_trueasm_mix_{pysr,lgf_go}/", font: CODE, size: 18 })]));
children.push(bullet([t("图（本报告全部插图）："), new TextRun({ text: "outputs/asm/figs_trueasm/fig_profile_{v,g,e,mix}.png · fig_profile_mix_b12b23.png · fig_iter_lgf.png · fig_iter_lgf_rho_cases.png · fig_iter_pysr.png · fig_mix_compare.png · fig_mix_percase_rho.png", font: CODE, size: 18 })]));
children.push(bullet([t("Markdown 报告："), new TextRun({ text: "outputs/asm/ASM_trueasm_report.md · ASM_trueasm_equations.md · ASM_trueasm_mix_report.md", font: CODE, size: 18 })]));
children.push(bullet([t("生成器："), new TextRun({ text: "scripts/rc_asm_true_{prep,pysr,lgf,eval,report,figs,mix,writeup,mix_figs}.py · scripts/gen_asm_word_report.js（本 Word）", font: CODE, size: 18 })]));

// ---------- assemble ----------
const doc = new Document({
  styles: {
    default: { document: { run: { font: ZH, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 30, bold: true, font: ZH, color: "1F3864" },
        paragraph: { spacing: { before: 300, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 25, bold: true, font: ZH, color: "2E5395" },
        paragraph: { spacing: { before: 220, after: 140 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 620, hanging: 300 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    footers: {
      default: new Footer({ children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "第 ", font: ZH, size: 16, color: "888888" }),
                   new TextRun({ children: [PageNumber.CURRENT], font: ZH, size: 16, color: "888888" }),
                   new TextRun({ text: " 页", font: ZH, size: 16, color: "888888" })],
      })] }),
    },
    children,
  }],
});

if (!fs.existsSync(OUTDIR)) fs.mkdirSync(OUTDIR, { recursive: true });
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(OUTFILE, buf);
  console.log("WROTE " + OUTFILE + " (" + (buf.length / 1024).toFixed(1) + " KB)");
});
