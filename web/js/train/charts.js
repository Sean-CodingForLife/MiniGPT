/* 训练曲线：Loss / 学习率 / 梯度范数与吞吐。
 *
 * 数据来源：events.jsonl 的 step/eval 事件（最细）、metrics.csv 的 eval 行、
 * status.json 的实时值。{x: step, y: value} 点集，x 轴线性 step，
 * train（每 log_interval）与 val（每 eval_interval）不同密度共存。
 *
 * 性能：按签名差量更新——数据没变就完全不碰 Chart；
 * 主题：配色从 CSS 变量读取，切换亮暗时重建配色并刷新。
 */

import { byId, el, chartPalette, invalidateChartPalette, chartFont, fontTokens, invalidateFontTokens } from "../ui.js";
import { on } from "../store.js";

const MAX_CHART_POINTS = 800;
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * 读取 CSS 变量并带回退值。
 */
function cssVar(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/**
 * 把十六进制颜色转换成带透明度的 rgba。
 */
function hexToRgba(hex, alpha) {
  const m = String(hex).match(/^#([0-9a-f]{6})$/i);
  if (!m) return `rgba(124,147,240,${alpha})`;
  const h = m[1];
  return `rgba(${parseInt(h.slice(0, 2), 16)},${parseInt(h.slice(2, 4), 16)},${parseInt(h.slice(4, 6), 16)},${alpha})`;
}

/**
 * 创建训练曲线控制器（创建/更新/主题重绘）。
 */
export function createTrainCharts() {
  const charts = { loss: null, lr: null, grad: null };
  let lastSig = "";

  function setChartOverlay(id, mode, text) {
    const overlay = byId(`${id}-overlay`);
    if (!overlay) return;
    overlay.replaceChildren();
    if (mode === "hidden") {
      overlay.hidden = true;
      return;
    }
    overlay.hidden = false;
    overlay.className = `chart-overlay${mode === "error" ? " error-state" : ""}`;
    if (mode === "loading") overlay.appendChild(el("span", "spinner"));
    overlay.appendChild(el("span", null, text || ""));
  }

  function setAllOverlays(mode, text) {
    ["loss-chart", "lr-chart", "grad-chart"].forEach((id) => setChartOverlay(id, mode, text));
  }

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
          title: yTitle ? { display: true, text: yTitle, color: tick, font: chartFont("data", fontTokens().size.xs) } : { display: false },
          ticks: { color: tick, maxTicksLimit: 5, font: chartFont("data", fontTokens().size.xs) },
          grid: { color: grid },
        },
      },
    };
  }

  function createChart(canvasId, datasets, yTitle) {
    const canvas = byId(canvasId);
    if (!canvas || typeof Chart === "undefined") return null;
    return new Chart(canvas.getContext("2d"), {
      type: "line",
      data: { datasets },
      options: chartOptions(yTitle),
    });
  }

  function datasets() {
    const palette = chartPalette();
    return {
      loss: [
        { label: "train_loss", data: [], borderColor: palette[0], backgroundColor: hexToRgba(palette[0], 0.12), fill: true },
        { label: "val_loss", data: [], borderColor: palette[1], backgroundColor: "transparent", borderDash: [] },
      ],
      lr: [
        { label: "lr", data: [], borderColor: palette[2], backgroundColor: hexToRgba(palette[2], 0.12), fill: true },
      ],
      grad: [
        { label: "grad_norm", data: [], borderColor: palette[3], yAxisID: "y" },
        { label: "tokens_per_sec", data: [], borderColor: palette[4], borderDash: [4, 3], yAxisID: "y1" },
      ],
    };
  }

  function init() {
    if (typeof Chart === "undefined") return;
    Chart.defaults.color = cssVar("--chart-legend", "#c4c6cc");
    const ft = fontTokens();
    Chart.defaults.font.family = ft.ui;
    Chart.defaults.font.size = ft.size.sm;
    Chart.defaults.font.weight = ft.weight.regular;
    const ds = datasets();
    charts.loss = createChart("loss-chart", ds.loss, "loss");
    charts.lr = createChart("lr-chart", ds.lr, "lr");
    charts.grad = createChart("grad-chart", ds.grad, "grad_norm");
    if (charts.grad) {
      const tick = cssVar("--chart-tick", "#9a9da6");
      charts.grad.options.scales.y1 = {
        position: "right",
        title: { display: true, text: "tokens/s", color: tick, font: chartFont("data", fontTokens().size.xs) },
        ticks: { color: tick, maxTicksLimit: 5, font: chartFont("data", fontTokens().size.xs) },
        grid: { drawOnChartArea: false },
      };
      charts.grad.update("none");
    }
    on("theme", applyTheme);
  }

  function applyTheme() {
    invalidateChartPalette();
    invalidateFontTokens();
    if (typeof Chart === "undefined") return;
    Chart.defaults.color = cssVar("--chart-legend", "#c4c6cc");
    const ft = fontTokens();
    Chart.defaults.font.family = ft.ui;
    Chart.defaults.font.size = ft.size.sm;
    const ds = datasets();
    [["loss", "loss"], ["lr", "lr"], ["grad", "grad"]].forEach(([key, dsKey]) => {
      const chart = charts[key];
      if (!chart) return;
      chart.data.datasets.forEach((dataset, index) => {
        const source = ds[dsKey][index];
        if (!source) return;
        dataset.borderColor = source.borderColor;
        dataset.backgroundColor = source.backgroundColor;
      });
      const options = chartOptions(key === "loss" ? "loss" : key === "lr" ? "lr" : "grad_norm");
      chart.options.scales.x = options.scales.x;
      chart.options.scales.y = options.scales.y;
      if (key === "grad" && chart.options.scales.y1) {
        chart.options.scales.y1.title.color = options.scales.y.title.color;
        chart.options.scales.y1.ticks.color = options.scales.y.ticks.color;
      }
      chart.options.plugins.legend.labels.color = options.plugins.legend.labels.color;
      chart.update("none");
    });
    lastSig = "";
  }

  function setChartData(chart, seriesList) {
    if (!chart) return;
    seriesList.forEach((points, index) => {
      if (chart.data.datasets[index]) chart.data.datasets[index].data = points;
    });
    chart.update("none");
  }

  function reset() {
    lastSig = "";
    const empty = [];
    setChartData(charts.loss, [empty, empty]);
    setChartData(charts.lr, [empty]);
    setChartData(charts.grad, [empty, empty]);
  }

  function resize() {
    Object.keys(charts).forEach((key) => {
      if (charts[key]) charts[key].resize();
    });
  }

  /** 签名差量：metrics 行数 + 事件数 + 实时步数/损失没变就完全不动图表。 */
  function update(metricsPayload, summary, events) {
    const rows = (metricsPayload && metricsPayload.rows) || [];
    const status = (summary && summary.status) || {};
    const liveStep = Number(status.step);
    const sig = `${rows.length}|${events.length}|${Number.isFinite(liveStep) ? liveStep : ""}|${status.train_loss}|${status.val_loss}|${status.lr}`;
    if (sig === lastSig) return;
    lastSig = sig;

    const series = { train: new Map(), val: new Map(), lr: new Map(), grad: new Map(), tps: new Map() };
    function add(name, x, y) {
      if (isNum(x) && isNum(y)) series[name].set(x, y);
    }
    function addStatusFields(step, payload) {
      add("train", step, payload.train_loss);
      add("lr", step, payload.lr);
      add("grad", step, payload.grad_norm);
      add("tps", step, payload.tokens_per_sec);
    }
    rows.forEach((row) => {
      addStatusFields(row.step, row);
      add("val", row.step, row.val_loss);
    });
    events.forEach((event) => {
      const payload = event.payload || {};
      if (event.event === "step" || event.event === "eval") {
        addStatusFields(event.step, payload);
        if (event.event === "eval") add("val", event.step, payload.val_loss);
      }
    });
    if (Number.isFinite(liveStep)) {
      addStatusFields(liveStep, status);
      add("val", liveStep, status.val_loss);
    }
    function points(name) {
      const list = [];
      series[name].forEach((y, x) => list.push({ x, y }));
      list.sort((a, b) => a.x - b.x);
      return list.slice(-MAX_CHART_POINTS);
    }
    const trainPts = points("train");
    const valPts = points("val");
    const lrPts = points("lr");
    const gradPts = points("grad");
    const tpsPts = points("tps");
    setChartData(charts.loss, [trainPts, valPts]);
    setChartData(charts.lr, [lrPts]);
    setChartData(charts.grad, [gradPts, tpsPts]);
    const emptyText = "暂无曲线数据 · 启动训练后自动绘制";
    setChartOverlay("loss-chart", trainPts.length || valPts.length ? "hidden" : "empty", emptyText);
    setChartOverlay("lr-chart", lrPts.length ? "hidden" : "empty", emptyText);
    setChartOverlay("grad-chart", gradPts.length || tpsPts.length ? "hidden" : "empty", emptyText);
  }

  return { init, update, reset, resize, setAllOverlays };
}