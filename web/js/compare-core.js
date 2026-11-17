/* N07 对比页的纯函数层：API 聚合数据 → Chart.js datasets / 摘要表。
 *
 * 在 SPA 中的位置：compare.js 的唯一数据变换依赖；本模块不碰 DOM、不读全局状态。
 * 数据来源：GET /api/v1/compare?runs=a,b,c（端点构造见 api.js 的 endpoints.compare），
 * 由后端聚合 runs/<id>/metrics.csv 与 training_report.json。
 * 依赖：无（纯 ESM），因此可在 node 中直接单测（tests/compare_core_check.mjs）。
 *
 * 关键约束：
 *  - 色板是固定基准色；compare.js 会按当前主题把它们映射到 --chart-* 变量；
 *  - 数据点缺失（null / "" / 非数字）时跳过而不是画 0，避免曲线出现假零点；
 *  - 表结构与后端字段名一一对应，改字段需同步后端 compare 端点。
 */

/**
 * 固定 8 色基准色板（train 实线 / val 虚线共用同一色）。
 * 与 style.css 的 --chart-1..8 默认值保持一致，便于无主题时降级。
 */
const PALETTE = [
  "#7c93f0",
  "#d3a44e",
  "#9d8cc4",
  "#5fb88f",
  "#c98aa6",
  "#cf9463",
  "#6fb3b8",
  "#d97a72",
];

/**
 * 按序号取色，序号可为负或超过色板长度（取模回绕）。
 * @param {number} index run 在响应数组中的下标。
 * @returns {string} 十六进制颜色，形如 "#7c93f0"。
 */
export function colorForIndex(index) {
  return PALETTE[((index % PALETTE.length) + PALETTE.length) % PALETTE.length];
}

/**
 * 取出响应里的 runs 数组。
 * @param {object|null} payload /api/v1/compare 的响应体。
 * @returns {Array<object>} 结构异常时返回空数组（调用方无需再判空）。
 */
function runsOf(payload) {
  if (!payload || !Array.isArray(payload.runs)) return [];
  return payload.runs;
}

/**
 * 把一列数值转成 Chart.js 需要的 {x: step, y: value} 点集。
 * 缺失值（null / undefined / 空串 / 非有限数）直接跳过，曲线留空而不是画 0。
 * @param {{step?: Array<number>, [key: string]: any}} series 单个 run 的序列对象。
 * @param {string} key 取值列名，如 "train_loss"、"val_ppl"。
 * @returns {Array<{x: number, y: number}>} 按原顺序排列（不排序，调用方按 step 递增假设绘制）。
 */
export function pointsFor(series, key) {
  const steps = Array.isArray(series && series.step) ? series.step : [];
  const values = Array.isArray(series && series[key]) ? series[key] : [];
  const points = [];
  for (let index = 0; index < steps.length; index += 1) {
    const rawX = steps[index];
    const rawY = values[index];
    // null/undefined/空串不是有效数据点（Number(null)===0 会画假零点）。
    if (rawX === null || rawX === undefined || rawX === "") continue;
    if (rawY === null || rawY === undefined || rawY === "") continue;
    const x = Number(rawX);
    const y = Number(rawY);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    points.push({ x, y });
  }
  return points;
}

/**
 * loss 图数据集：每个 run 两条 dataset（train 实线、val 虚线），共用一个颜色。
 * @param {object|null} payload /api/v1/compare 的响应体。
 * @returns {Array<object>} 可直接赋给 Chart.js 的 datasets（borderColor 仍为基准色）。
 */
export function buildLossDatasets(payload) {
  const datasets = [];
  runsOf(payload).forEach((run, index) => {
    const color = colorForIndex(index);
    datasets.push({
      label: `${run.run_id} · train_loss`,
      data: pointsFor(run.series, "train_loss"),
      borderColor: color,
      backgroundColor: "transparent",
    });
    datasets.push({
      label: `${run.run_id} · val_loss`,
      data: pointsFor(run.series, "val_loss"),
      borderColor: color,
      backgroundColor: "transparent",
      borderDash: [6, 4],
    });
  });
  return datasets;
}

/**
 * 验证困惑度图数据集：每个 run 一条 val_ppl 曲线。
 * @param {object|null} payload /api/v1/compare 的响应体。
 * @returns {Array<object>} Chart.js datasets。
 */
export function buildPplDatasets(payload) {
  return runsOf(payload).map((run, index) => ({
    label: `${run.run_id} · val_ppl`,
    data: pointsFor(run.series, "val_ppl"),
    borderColor: colorForIndex(index),
    backgroundColor: "transparent",
  }));
}

/**
 * 单元格格式化：缺失值给 "—"，数字按位数截断，非数字也给 "—"。
 * @param {any} value 原始值。
 * @param {number} digits 小数位。
 * @returns {string}
 */
function fmtCell(value, digits) {
  if (value === null || value === undefined || value === "") return "—";
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "—";
}

/**
 * 评估/训练摘要并排表：行=指标，列=run。
 * @param {object|null} payload /api/v1/compare 的响应体。
 * @returns {{headers: string[], rows: Array<Array<string|number>>}} 纯数据表描述，
 *   由 compare.js 用 ui.js 的 table() 渲染（本模块不创建 DOM）。
 */
export function buildSummaryTable(payload) {
  const runs = runsOf(payload);
  const headers = ["指标", ...runs.map((run) => String(run.run_id))];
  // 每行：[行标题, 取值函数, 备注占位]；备注列保留为 null 以便后续扩展。
  const metricRows = [
    ["阶段", (run) => run.stage || "—", null],
    ["状态", (run) => run.display_status || "—", null],
    ["指标行数", (run) => (Number.isFinite(Number(run.metrics_count)) ? String(run.metrics_count) : "—"), null],
    ["steps", (run) => (run.report && run.report.steps !== undefined ? String(run.report.steps) : "—"), null],
    ["best_val_loss", (run) => fmtCell(run.report && run.report.best_val_loss, 4), null],
    // best_step=-1 是「无最优检查点」哨兵值，不应当作真实步数展示。
    ["best_step", (run) => {
      const step = run.report ? Number(run.report.best_step) : NaN;
      return Number.isFinite(step) && step >= 0 ? String(step) : "—";
    }, null],
    ["final_train_loss", (run) => fmtCell(run.report && run.report.final_train_loss, 4), null],
    ["耗时(s)", (run) => fmtCell(run.report && run.report.elapsed_s, 1), null],
    ["早停", (run) => (run.report ? (run.report.stopped_early ? "是" : "否") : "—"), null],
  ];
  const rows = metricRows.map(([label, getter]) => [label, ...runs.map(getter)]);
  return { headers, rows };
}
