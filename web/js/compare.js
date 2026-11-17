/* N07 多 run 对比页：勾选任意多个 run → 同图叠加 loss/ppl 曲线 + 摘要并排表。
 *
 * 数据来自 GET /api/v1/compare?runs=a,b,c（聚合 metrics.csv + training_report.json）。
 * 按可见页签轮询：只在 #/compare 可见时定时刷新；全局「暂停刷新」同样生效。
 * 纯数据转换在 compare-core.js（可 node 单测）；本模块只做 DOM 与 Chart.js。
 * 勾选变化用序号守卫：在途请求不丢弃新选择，晚到响应不覆盖新结果。
 */

import { byId, el, getJson, table, loadingBlock, emptyBlock, errorBlock, chartPalette, invalidateChartPalette, chartFont, fontTokens, invalidateFontTokens } from "./ui.js";
import { buildLossDatasets, buildPplDatasets, buildSummaryTable } from "./compare-core.js";
import { endpoints } from "./api.js";
import { store, on } from "./store.js";

const POLL_MS = 5000;

const state = {
  visible: false,
  selected: new Set(),
  runs: [],
  charts: { loss: null, ppl: null },
  loading: false,
  pending: false,
  timer: null,
  lastPayload: null,
  compareSeq: 0,
};

/**
 * 显示对比页状态提示。
 */
function setStatus(message, isError = false) {
  const node = byId("compare-status");
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("error", Boolean(isError));
}

/* ------------------------------------------------------------------ 图表 */
function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/**
 * 生成对比图表的通用配置（含 Y 轴标题）。
 */
function chartOptions(yTitle) {
  const tick = cssVar("--chart-tick", "#9a9da6");
  const grid = cssVar("--chart-grid", "rgba(255,255,255,0.06)");
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    normalized: true,
    interaction: { mode: "index", intersect: false },
    elements: { point: { radius: 0, hitRadius: 8 }, line: { borderWidth: 1.8, tension: 0.25 } },
    plugins: {
      legend: { labels: { color: cssVar("--chart-legend", "#c4c6cc"), boxWidth: 12, boxHeight: 2, font: chartFont("data", fontTokens().size.xs) } },
      tooltip: {
        enabled: true,
        titleFont: chartFont("ui", fontTokens().size.sm, fontTokens().weight.medium),
        bodyFont: chartFont("data", fontTokens().size.sm),
        padding: 8,
        cornerRadius: 4,
      },
    },
    scales: {
      x: {
        type: "linear",
        title: { display: true, text: "step", color: tick, font: chartFont("data", fontTokens().size.xs) },
        ticks: { color: tick, maxTicksLimit: 6, font: chartFont("data", fontTokens().size.xs) },
        grid: { color: grid },
      },
      y: {
        title: { display: true, text: yTitle, color: tick, font: chartFont("data", fontTokens().size.xs) },
        ticks: { color: tick, maxTicksLimit: 5, font: chartFont("data", fontTokens().size.xs) },
        grid: { color: grid },
      },
    },
  };
}

/** 数据集配色跟随主题：compare-core 的固定色板映射到当前 --chart-*。 */
function themedDatasets(datasets) {
  const palette = chartPalette();
  const colorMap = new Map();
  datasets.forEach((dataset) => {
    if (!colorMap.has(dataset.borderColor)) {
      colorMap.set(dataset.borderColor, palette[colorMap.size % palette.length]);
    }
    dataset.borderColor = colorMap.get(dataset.borderColor);
  });
  return datasets;
}

/**
 * 创建或更新指定 canvas 上的 Chart.js 图表。
 */
function renderChart(key, canvasId, datasets, yTitle) {
  const canvas = byId(canvasId);
  if (!canvas || typeof Chart === "undefined") return;
  if (!state.charts[key]) {
    state.charts[key] = new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets: [] },
      options: chartOptions(yTitle),
    });
  }
  const chart = state.charts[key];
  chart.data.datasets = themedDatasets(datasets);
  chart.update("none");
}

/**
 * 主题切换后重设图表配色。
 */
function restyleCharts() {
  invalidateFontTokens();
  invalidateChartPalette();
  ["loss", "ppl"].forEach((key) => {
    const chart = state.charts[key];
    if (!chart) return;
    const options = chartOptions(key === "loss" ? "loss" : "val_ppl");
    chart.options.scales.x = options.scales.x;
    chart.options.scales.y = options.scales.y;
    chart.options.plugins.legend.labels.color = options.plugins.legend.labels.color;
    chart.options.plugins.legend.labels.font = options.plugins.legend.labels.font;
    chart.options.plugins.tooltip = options.plugins.tooltip;
    chart.update("none");
  });
  if (state.lastPayload) renderPayload(state.lastPayload);
}

/* ------------------------------------------------------------------ 渲染 */
let runListSig = "";
/**
 * 渲染可勾选的 run 列表。
 */
function renderRunList() {
  const root = byId("compare-run-list");
  if (!root) return;
  const sig = state.runs.map((run) => `${run.run_id}:${run.display_status}:${run.has_best}`).join("|");
  if (sig === runListSig) {
    // 列表结构没变：只同步勾选态，不重建 DOM（保住焦点）。
    root.querySelectorAll("input[type=checkbox]").forEach((checkbox) => {
      checkbox.checked = state.selected.has(checkbox.value);
    });
    return;
  }
  runListSig = sig;
  root.replaceChildren();
  if (!state.runs.length) {
    root.appendChild(emptyBlock("还没有运行", "先完成一次训练，runs/ 下出现运行目录后即可对比。"));
    return;
  }
  state.runs.forEach((run) => {
    const label = el("label", "compare-run-item");
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = run.run_id;
    checkbox.checked = state.selected.has(run.run_id);
    checkbox.addEventListener("change", () => {
      if (checkbox.checked) state.selected.add(run.run_id);
      else state.selected.delete(run.run_id);
      loadCompare(true);
    });
    label.appendChild(checkbox);
    label.appendChild(el("span", "compare-run-name", run.run_id));
    label.appendChild(el(
      "span",
      "compare-run-meta",
      `${run.stage || "?"} · ${run.display_status || "?"}${run.has_best ? " · best.pt" : ""}`,
    ));
    root.appendChild(label);
  });
}

let summarySig = "";
/**
 * 渲染对比摘要表。
 */
function renderSummary(payload) {
  const root = byId("compare-summary");
  if (!root) return;
  const summary = buildSummaryTable(payload);
  const sig = JSON.stringify(summary);
  if (sig === summarySig) return;
  summarySig = sig;
  root.replaceChildren();
  if (summary.headers.length <= 1) {
    root.appendChild(emptyBlock("未选择运行", "左侧勾选至少一个运行后展示摘要。"));
    return;
  }
  root.appendChild(table(summary.headers, summary.rows));
}

/**
 * 渲染接口原始返回，便于排查。
 */
function renderPayload(payload) {
  state.lastPayload = payload;
  renderChart("loss", "compare-loss-chart", buildLossDatasets(payload), "loss");
  renderChart("ppl", "compare-ppl-chart", buildPplDatasets(payload), "val_ppl");
  renderSummary(payload);
  const count = payload && payload.count ? payload.count : 0;
  setStatus(count ? `对比 ${count} 个运行` : "");
}

/* ------------------------------------------------------------------ 数据 */
async function loadRuns() {
  const payload = await getJson(endpoints.runs());
  state.runs = (payload && payload.runs) || [];
  // 清理已不存在的选择。
  const known = new Set(state.runs.map((run) => run.run_id));
  state.selected.forEach((runId) => {
    if (!known.has(runId)) state.selected.delete(runId);
  });
  renderRunList();
}

/**
 * 拉取并渲染选中 run 的对比数据。
 */
async function loadCompare(force = false) {
  // 在途请求不丢弃新选择：标记 pending，当前请求结束后自动补一次。
  if (state.loading) {
    state.pending = true;
    return;
  }
  if (!state.selected.size) {
    state.compareSeq += 1;
    state.lastPayload = null;
    renderChart("loss", "compare-loss-chart", [], "loss");
    renderChart("ppl", "compare-ppl-chart", [], "val_ppl");
    renderSummary({ runs: [] });
    setStatus("勾选左侧运行开始对比");
    return;
  }
  state.loading = true;
  const seq = ++state.compareSeq;
  if (force) setStatus("加载对比数据…");
  try {
    const runs = [...state.selected].join(",");
    const payload = await getJson(endpoints.compare(runs));
    if (seq !== state.compareSeq) return;
    renderPayload(payload);
  } catch (error) {
    if (seq !== state.compareSeq) return;
    setStatus(`对比加载失败：${error.message}`, true);
    const root = byId("compare-summary");
    if (root && force) {
      root.replaceChildren(errorBlock(error.message, "重试", () => loadCompare(true)));
    }
  } finally {
    state.loading = false;
    if (state.pending) {
      state.pending = false;
      loadCompare(true);
    }
  }
}

/**
 * 刷新列表与对比结果。
 */
async function refreshAll(force = false) {
  try {
    await loadRuns();
  } catch (error) {
    setStatus(`运行列表加载失败：${error.message}`, true);
    const root = byId("compare-run-list");
    if (root) root.replaceChildren(errorBlock(error.message, "重试", () => refreshAll(true)));
    return;
  }
  await loadCompare(force);
}

/* ------------------------------------------------------------------ 轮询 */
function tick() {
  if (!state.visible || store.paused) return;
  refreshAll(false);
}

/**
 * 初始化对比页（绑定事件并首次加载）。
 */
export function initCompareView() {
  const root = byId("compare-run-list");
  if (!root) return;
  const refresh = byId("compare-refresh-btn");
  if (refresh) refresh.addEventListener("click", () => refreshAll(true));
  window.addEventListener("spa:view", (event) => {
    const detail = (event && event.detail) || {};
    state.visible = detail.view === "compare";
    if (state.visible) refreshAll(true);
  });
  on("theme", () => {
    if (state.visible) restyleCharts();
  });
  state.timer = window.setInterval(tick, POLL_MS);
}