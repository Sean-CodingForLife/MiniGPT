/* 注意力视图运行时检查：用最小 DOM 桩加载真实 attention.js，
 * 喂入 1024 个 token 的上下文与矩阵，验证画布尺寸固定、只渲染 Top-K、
 * 未展示权重被正确汇总。输出 JSON 供 Python 测试断言。
 */

import { readFile } from "node:fs/promises";

const listeners = new Map();

function makeCtx() {
  const calls = { fillText: 0, fillRect: 0, bezierCurveTo: 0, stroke: 0 };
  const noop = () => {};
  return {
    calls,
    canvas: null,
    setTransform: noop,
    clearRect: noop,
    save: noop,
    restore: noop,
    translate: noop,
    rotate: noop,
    beginPath: noop,
    closePath: noop,
    moveTo: noop,
    lineTo: noop,
    arc: noop,
    arcTo: noop,
    fill: noop,
    strokeRect: noop,
    measureText: () => ({ width: 10 }),
    fillRect: () => { calls.fillRect += 1; },
    stroke: () => { calls.stroke += 1; },
    bezierCurveTo: () => { calls.bezierCurveTo += 1; },
    fillText: () => { calls.fillText += 1; },
    drawImage: noop,
  };
}

function makeEl(id) {
  const el = {
    id,
    children: [],
    style: {},
    dataset: {},
    value: "",
    checked: false,
    disabled: false,
    hidden: false,
    textContent: "",
    className: "",
    clientWidth: 900,
    clientHeight: 380,
    width: 0,
    height: 0,
    classList: {
      toggle() {},
      add() {},
      remove() {},
      contains: () => false,
    },
    appendChild(child) { this.children.push(child); return child; },
    replaceChildren(...kids) { this.children = kids.slice(); },
    addEventListener(type, fn) {
      const key = `${id}:${type}`;
      if (!listeners.has(key)) listeners.set(key, []);
      listeners.get(key).push(fn);
    },
    removeEventListener() {},
    remove() {},
    select() {},
    focus() {},
    click() {},
    getBoundingClientRect: () => ({ left: 0, top: 0, width: 900, height: 380 }),
    toDataURL: () => "data:image/png;base64,",
    getContext() {
      if (!this._ctx) this._ctx = makeCtx();
      return this._ctx;
    },
    get firstElementChild() { return this.children[0] || null; },
    get parentElement() { return elements.get("attn-stage-wrap"); },
  };
  return el;
}

const ids = [
  "attn-canvas", "attn-stage-wrap", "attn-tooltip", "attn-source", "attn-query-current",
  "attn-focus-hint", "attn-metrics", "attn-query-summary", "attn-selected-detail",
  "attn-screen-reader", "attn-query-prev", "attn-query-next", "attn-query-latest",
  "attn-query-select", "attn-layer", "attn-head", "attn-topk", "attn-threshold",
  "attn-threshold-out", "attn-order", "attn-palette", "attn-window", "attn-follow",
  "attn-values", "attn-normalize", "attn-mode-links", "attn-mode-matrix", "attn-mode-strip",
  "attn-search", "attn-reset", "attn-export", "attn-legend-bar", "attn-legend-min", "attn-legend-max",
];
const elements = new Map(ids.map((id) => [id, makeEl(id)]));

// 从真实页面读取控件初始值，保证测试与产品默认一致。
const html = await readFile(new URL("../web/index.html", import.meta.url), "utf8");
function inputValue(id, fallback) {
  const re = new RegExp(`id="${id}"[^>]*value="([^"]*)"`);
  const m = html.match(re);
  return m ? m[1] : fallback;
}
function isChecked(id) {
  const re = new RegExp(`id="${id}"[^>]*>`);
  const m = html.match(re);
  return m ? /\bchecked\b/.test(m[0]) : false;
}
elements.get("attn-topk").value = inputValue("attn-topk", "8");
elements.get("attn-threshold").value = inputValue("attn-threshold", "0");
elements.get("attn-order").value = "weight";
elements.get("attn-palette").value = "blue";
elements.get("attn-window").value = inputValue("attn-window", "12");
elements.get("attn-follow").checked = isChecked("attn-follow");
elements.get("attn-values").checked = isChecked("attn-values");
elements.get("attn-normalize").checked = isChecked("attn-normalize");
// select 的选中项来自 selected 属性
const topkSelected = html.match(/id="attn-topk"[\s\S]*?<option value="(\d+)" selected/);
if (topkSelected) elements.get("attn-topk").value = topkSelected[1];

globalThis.document = {
  getElementById: (id) => elements.get(id) || null,
  createElement: (tag) => makeEl(`<${tag}>`),
  createTextNode: (text) => ({ nodeType: 3, textContent: String(text) }),
  body: makeEl("body"),
  addEventListener() {},
  removeEventListener() {},
};
globalThis.window = {
  devicePixelRatio: 2,
  addEventListener() {},
  setTimeout: (fn) => { fn(); return 0; },
  clearTimeout() {},
};
globalThis.requestAnimationFrame = (fn) => { fn(0); return 0; };
globalThis.ResizeObserver = class { observe() {} disconnect() {} };

const { initAttentionDemo } = await import(new URL("../web/js/attention.js", import.meta.url).href);
const demo = initAttentionDemo({ onNavigate() {}, onLatest() {}, onFollowChange() {} });

function buildRows(count, layers = 2, heads = 4, sum = 1) {
  const rows = [];
  for (let l = 0; l < layers; l += 1) {
    const hs = [];
    for (let h = 0; h < heads; h += 1) {
      const raw = Array.from({ length: count }, (_, i) => 1 + ((i * 7 + h * 13 + l * 3) % 11));
      const total = raw.reduce((a, b) => a + b, 0);
      hs.push(raw.map((v) => (v / total) * sum));
    }
    rows.push(hs);
  }
  return rows;
}

function metricsText() {
  return elements.get("attn-metrics").children.map((c) => {
    const label = c.children[0] ? c.children[0].textContent : "";
    const value = c.children.slice(1).map((n) => n.textContent).join("");
    return `${label}${value}`;
  }).join(" | ");
}

const canvas = elements.get("attn-canvas");
const results = {};

// 基线：短上下文
demo.setData({
  context: Array.from({ length: 10 }, (_, i) => `t${i}`),
  rows: buildRows(10),
  queryToken: "t9",
  predictedToken: "next",
  step: 1,
  seriesId: "short",
});
results.shortHeight = canvas.height;
results.shortMetrics = metricsText();

// 长序列：1024 个 Key
const long = 1024;
demo.setData({
  context: Array.from({ length: long }, (_, i) => `tok${i}`),
  rows: buildRows(long),
  queryToken: `tok${long - 1}`,
  predictedToken: "next",
  step: long,
  seriesId: "long",
});
results.longHeight = canvas.height;
results.longMetrics = metricsText();
results.longSummary = elements.get("attn-query-summary").textContent;
results.longSr = elements.get("attn-screen-reader").textContent;

// 窗口外权重：只传输 60% 的注意力质量
demo.setData({
  context: Array.from({ length: long }, (_, i) => `tok${i}`),
  rows: buildRows(long, 2, 4, 0.6),
  queryToken: `tok${long - 1}`,
  predictedToken: "next",
  step: long,
  seriesId: "partial",
});
results.partialMetrics = metricsText();

// 局部矩阵：1024×1024 输入也只渲染固定窗口
const matrix = [[Array.from({ length: long }, () => Array.from({ length: long }, (_, j) => (j % 7) / 70))]];
elements.get("attn-mode-matrix").textContent = "";
const matrixListeners = listeners.get("attn-mode-matrix:click") || [];
demo.setData({
  context: Array.from({ length: long }, (_, i) => `tok${i}`),
  rows: buildRows(long),
  queryToken: `tok${long - 1}`,
  predictedToken: "next",
  step: long,
  matrix,
  matrixContext: Array.from({ length: long }, (_, i) => `tok${i}`),
  seriesId: "matrix",
});
matrixListeners.forEach((fn) => fn());
results.matrixHeight = canvas.height;
results.matrixMetrics = metricsText();

// 回到连线视图后再次切换长/短，确认尺寸稳定
const linkListeners = listeners.get("attn-mode-links:click") || [];
linkListeners.forEach((fn) => fn());
demo.setData({
  context: Array.from({ length: 10 }, (_, i) => `t${i}`),
  rows: buildRows(10),
  queryToken: "t9",
  predictedToken: "next",
  step: 1,
  seriesId: "short-again",
});
results.returnHeight = canvas.height;

// 交互回归：芯片为 Canvas 绘制内容，用坐标事件验证点击 / 键盘 / 行放大 / 导出。
const stripListeners = listeners.get("attn-mode-strip:click") || [];
stripListeners.forEach((fn) => fn());
const clickListeners = listeners.get("attn-canvas:click") || [];
clickListeners.forEach((fn) => fn({ clientX: 254, clientY: 124 }));
results.clickDetail = elements.get("attn-selected-detail").textContent;
const keyListeners = listeners.get("attn-canvas:keydown") || [];
keyListeners.forEach((fn) => fn({ key: "ArrowRight", preventDefault() {} }));
results.keyDetail = elements.get("attn-selected-detail").textContent;
elements.get("attn-normalize").checked = true;
(listeners.get("attn-normalize:change") || []).forEach((fn) => fn());
results.legendMax = elements.get("attn-legend-max").textContent;
let exportOk = true;
try {
  (listeners.get("attn-export:click") || []).forEach((fn) => fn());
} catch (error) {
  exportOk = false;
}
results.exportOk = exportOk;

// 窄屏回归：700px 宽度下条带视图仍正常渲染且高度固定。
const wrapEl = elements.get("attn-stage-wrap");
wrapEl.clientWidth = 700;
wrapEl.getBoundingClientRect = () => ({ left: 0, top: 0, width: 700, height: 380 });
demo.refresh();
results.narrowHeight = canvas.height;
results.narrowMetrics = metricsText();

console.log(JSON.stringify(results));