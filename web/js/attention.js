/* 固定尺寸的 Transformer 自注意力重点视图。
 *
 * 主视图只回答一个问题：当前预测位置 Q 正在关注哪些 K？
 *   - 默认展示权重最高的 8 个 Key，线宽/颜色 = 真实 softmax 权重；
 *   - 未展示项汇总为“其余窗口内权重”，不置零、不伪装成不存在；
 *   - 后端只传输最近 max_context 个 Key，窗口外权重单独标注；
 *   - 局部矩阵模式只渲染固定窗口，完整矩阵放在按需打开的弹窗里。
 *
 * 画布高度由 CSS 固定，不随生成长度增长；无持续动画、无音效。
 * 数据契约：{ context, rows:[layer][head][weight], queryToken, step,
 *            history?:[{abs,start,context,rows}], promptLen?,
 *            matrix?:[layer][head][row][col], matrixContext?, live, label }
 */

import { on } from "./store.js";
import { canvasFont, fontTokens, invalidateFontTokens } from "./ui.js";

const PALETTES = {
  blue: { name: "靛蓝", stops: [[18, 20, 32], [40, 50, 96], [86, 108, 200], [154, 171, 245]], accent: "#9aabf5" },
  magma: { name: "暖色", stops: [[26, 18, 24], [92, 40, 60], [190, 100, 90], [224, 170, 120]], accent: "#cf9463" },
  gray: { name: "灰阶", stops: [[22, 24, 28], [74, 78, 86], [154, 157, 166], [236, 236, 239]], accent: "#ececef" },
};

const DEFAULTS = {
  layer: 0,
  head: 0,
  topK: 8,
  threshold: 0,
  mode: "strip",
  order: "weight",
  palette: "blue",
  stroke: 2.5,
  window: 12,
  follow: true,
  showValues: true,
  normalizeColor: false,
};

const MAX_MATRIX_WINDOW = 16;

/**
 * 生成演示用注意力数据（无后端时展示）。
 */
function exampleData() {
  const context = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too"];
  const rows = [];
  for (let l = 0; l < 2; l += 1) {
    const heads = [];
    for (let h = 0; h < 4; h += 1) {
      const raw = context.map((token, i) => {
        let w = 0.015 + 0.01 * ((i + h + l) % 3);
        const pattern = (l * 4 + h) % 4;
        if (pattern === 0 && token === "animal") w += 0.42;
        if (pattern === 1 && token === "street") w += 0.38;
        if (pattern === 2 && token === "it") w += 0.34;
        if (pattern === 3 && i === context.length - 1) w += 0.3;
        if (i === context.length - 1) w += 0.08;
        return w;
      });
      const sum = raw.reduce((a, b) => a + b, 0) || 1;
      heads.push(raw.map((v) => v / sum));
    }
    rows.push(heads);
  }
  return {
    context,
    rows,
    queryToken: "too",
    predictedToken: "tired",
    step: 0,
    live: false,
    label: "示例数据",
    seriesId: "example",
  };
}

/**
 * 把 token 转成可读文本（空格/换行用符号表示）。
 */
function tokenText(raw) {
  const text = raw === undefined || raw === null ? "" : String(raw);
  if (/^\\u[0-9a-fA-F]{4}$/.test(text)) return String.fromCharCode(parseInt(text.slice(2), 16));
  const map = { "\\n": "\n", "\\r": "\r", "\\t": "\t", "\\\\": "\\" };
  return Object.prototype.hasOwnProperty.call(map, text) ? map[text] : text;
}

/**
 * 截断 token 标签以便绘制坐标轴。
 */
function tokenLabel(raw, max = 8) {
  const text = tokenText(raw);
  if (text === "") return "∅";
  if (text === " ") return "␠";
  if (text === "\n") return "↵";
  if (text === "\r") return "␍";
  if (text === "\t") return "⇥";
  return text.length > max ? `${text.slice(0, max)}…` : text;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/**
 * 在色带中按 t 采样颜色。
 */
function samplePalette(stops, t) {
  const x = clamp(t, 0, 1) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

/**
 * 拼装带透明度的 rgb 字符串。
 */
function rgb(c, alpha) {
  return alpha === undefined ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
}

function pctText(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return "—";
  return `${(n * 100).toFixed(n >= 0.1 ? 1 : 2)}%`;
}

/**
 * 初始化独立注意力演示组件（自包含，不依赖 SPA 状态）。
 */
export function initAttentionDemo(options = {}) {
  const canvas = document.getElementById("attn-canvas");
  if (!canvas) return null;
  const ctx = canvas.getContext("2d");
  const wrap = document.getElementById("attn-stage-wrap") || canvas.parentElement;
  const tooltip = document.getElementById("attn-tooltip");
  const $ = (id) => document.getElementById(id);

  const settings = { ...DEFAULTS };
  let data = exampleData();
  let searchTerm = "";
  let selectedKey = null;
  let hoverKey = null;
  let dpr = 1;
  let cssW = 0;
  let cssH = 0;
  let hitRows = [];
  let lastLayerCount = -1;
  let lastHeadCount = -1;
  let resizeScheduled = false;
  // 渲染缓存：数据/层/头不变时不重复构造候选行（hover 高频路径）。
  let rowsCache = null;
  let candCache = null;
  let hoverRaf = 0;
  let destroyed = false;
  const disposers = [];
  const canvasStyle = typeof window.getComputedStyle === "function" ? window.getComputedStyle(canvas) : null;
  let ink = (canvasStyle && canvasStyle.getPropertyValue("--text").trim()) || "#ececef";
  let inkSoft = (canvasStyle && canvasStyle.getPropertyValue("--text-soft").trim()) || "#c4c6cc";
  let inkMuted = (canvasStyle && canvasStyle.getPropertyValue("--muted").trim()) || "#9a9da6";

  function refreshInk() {
    if (!canvasStyle) return;
    ink = canvasStyle.getPropertyValue("--text").trim() || ink;
    inkSoft = canvasStyle.getPropertyValue("--text-soft").trim() || inkSoft;
    inkMuted = canvasStyle.getPropertyValue("--muted").trim() || inkMuted;
  }

  /** 登记监听器，destroy 时统一注销。 */
  function listen(target, type, handler) {
    target.addEventListener(type, handler);
    disposers.push(() => target.removeEventListener(type, handler));
  }

  function invalidateCaches() {
    rowsCache = null;
    candCache = null;
  }

  function palette() { return PALETTES[settings.palette] || PALETTES.blue; }

  function currentRows() {
    if (rowsCache) return rowsCache;
    const layers = data.rows || [];
    const l = clamp(settings.layer, 0, Math.max(0, layers.length - 1));
    const heads = layers[l] || [];
    const h = clamp(settings.head, 0, Math.max(0, heads.length - 1));
    rowsCache = (heads[h] || []).map((v) => {
      const n = Number(v);
      return Number.isFinite(n) && n > 0 ? n : 0;
    });
    return rowsCache;
  }

  function currentMatrix() {
    if (!data.matrix || !data.matrix.length) return null;
    const l = clamp(settings.layer, 0, data.matrix.length - 1);
    const heads = data.matrix[l] || [];
    const h = clamp(settings.head, 0, Math.max(0, heads.length - 1));
    return heads[h] || null;
  }

  function candidates() {
    const candKey = `${data.seriesId}|${settings.layer}|${settings.head}|${settings.topK}|${settings.threshold}|${settings.order}|${searchTerm}`;
    if (candCache && candCache.key === candKey) return candCache.value;
    const values = currentRows();
    const context = data.context || [];
    let items = values.map((weight, index) => ({
      index,
      weight,
      token: context[index] !== undefined ? context[index] : `#${index}`,
    }));
    if (searchTerm) {
      const q = searchTerm.toLowerCase();
      items = items.filter((item) => tokenText(item.token).toLowerCase().includes(q));
    }
    items.sort((a, b) => b.weight - a.weight || a.index - b.index);
    items = items.slice(0, settings.topK);
    if (settings.threshold > 0) items = items.filter((item) => item.weight >= settings.threshold);
    const topRank = new Map(items.map((item, i) => [item.index, i + 1]));
    if (settings.order === "position") items.sort((a, b) => a.index - b.index);
    const result = { items, values, topRank };
    candCache = { key: candKey, value: result };
    return result;
  }

  function populateSelectors(force) {
    const layers = data.rows || [];
    const heads = layers[settings.layer] || layers[0] || [];
    if (!force && layers.length === lastLayerCount && heads.length === lastHeadCount) return;
    lastLayerCount = layers.length;
    lastHeadCount = heads.length;
    const layerSel = $("attn-layer");
    const headSel = $("attn-head");
    if (layerSel) {
      layerSel.replaceChildren();
      layers.forEach((_, i) => {
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = `L${i}`;
        layerSel.appendChild(o);
      });
      settings.layer = clamp(settings.layer, 0, Math.max(0, layers.length - 1));
      layerSel.value = String(settings.layer);
    }
    if (headSel) {
      headSel.replaceChildren();
      heads.forEach((_, i) => {
        const o = document.createElement("option");
        o.value = String(i);
        o.textContent = `H${i}`;
        headSel.appendChild(o);
      });
      settings.head = clamp(settings.head, 0, Math.max(0, heads.length - 1));
      headSel.value = String(settings.head);
    }
  }

  function updateText() {
    const badge = $("attn-source");
    if (badge) {
      badge.textContent = data.label || (data.live ? "实时生成" : "真实数据");
      badge.classList.toggle("attn-chip-example", Boolean(data.label));
    }
    const current = $("attn-query-current");
    if (current) {
      const stepText = data.step ? `第 ${data.step} 步` : "示例";
      const predicted = data.predictedToken || data.queryToken;
      current.textContent = `${stepText} · Q「${tokenLabel(data.queryToken, 8)}」→ 预测「${tokenLabel(predicted, 8)}」`;
    }
    const hint = $("attn-focus-hint");
    if (hint) hint.textContent = "固定尺寸 · 不随生成长度增长";

    const { items, values } = candidates();
    const windowSum = values.reduce((a, b) => a + b, 0);
    const shownSum = items.reduce((a, item) => a + item.weight, 0);
    const restInWindow = Math.max(0, windowSum - shownSum);
    const outsideWindow = Math.max(0, 1 - windowSum);
    const metrics = $("attn-metrics");
    if (metrics) {
      metrics.replaceChildren();
      const add = (label, value, cls) => {
        const node = document.createElement("span");
        node.className = `attn-metric${cls ? ` ${cls}` : ""}`;
        const b = document.createElement("b");
        b.textContent = label;
        node.appendChild(b);
        node.appendChild(document.createTextNode(value));
        metrics.appendChild(node);
      };
      add("展示 Key", `${items.length} / ${values.length}`);
      add("展示覆盖", pctText(shownSum), "attn-metric-accent");
      add("其余窗口内", pctText(restInWindow));
      add("窗口外未传输", pctText(outsideWindow));
    }
    const summary = $("attn-query-summary");
    if (summary) {
      const source = data.context && data.context.length ? tokenLabel(data.context[data.context.length - 1], 10) : "—";
      summary.textContent = `Q 来自位置「${source}」 · 线宽 = 真实权重${searchTerm ? ` · 搜索“${searchTerm}”` : ""}`;
    }
    const detail = $("attn-selected-detail");
    if (detail) {
      const item = items.find((x) => x.index === selectedKey) || (hoverKey !== null ? items.find((x) => x.index === hoverKey) : null);
      if (item) {
        detail.textContent = `K[${item.index}]「${tokenLabel(item.token, 12)}」 · α=${item.weight.toFixed(4)}（${pctText(item.weight)}） · 贡献 α·V`;
      } else if (searchTerm && !items.length) {
        detail.textContent = "当前 Top-K / 阈值下没有匹配的 Key；可降低阈值或清空搜索。";
      } else {
        detail.textContent = "点击或用 ↑↓ 选择一个 Key 查看精确权重";
      }
    }
    const sr = $("attn-screen-reader");
    if (sr) {
      sr.textContent = items.length
        ? `Query ${tokenLabel(data.queryToken, 10)} 预测 ${tokenLabel(data.predictedToken || data.queryToken, 10)}，展示 ${items.length} 个 Key，覆盖 ${pctText(shownSum)}。最高权重 ${tokenLabel(items[0].token, 10)} ${pctText(items[0].weight)}。`
        : "当前没有达到阈值的 Key。";
    }
    const prev = $("attn-query-prev");
    const next = $("attn-query-next");
    const latest = $("attn-query-latest");
    const nav = data.nav || {};
    const hasRealQuery = Boolean(data.step);
    if (prev) prev.disabled = !options.onNavigate || !hasRealQuery || nav.canPrev === false;
    if (next) next.disabled = !options.onNavigate || !hasRealQuery || nav.canNext === false;
    if (latest) latest.disabled = !options.onLatest || !hasRealQuery || nav.atLatest !== false;
  }

  function resize() {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    cssW = Math.max(200, wrap.clientWidth);
    cssH = Math.max(240, wrap.clientHeight);
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    canvas.style.width = `${cssW}px`;
    canvas.style.height = `${cssH}px`;
    render();
  }

  function scheduleResize() {
    if (resizeScheduled) return;
    resizeScheduled = true;
    requestAnimationFrame(() => {
      resizeScheduled = false;
      resize();
    });
  }

  function drawEmpty(message) {
    ctx.fillStyle = inkMuted;
    ctx.font = canvasFont("ui", fontTokens().size.base);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(message, cssW / 2, cssH / 2);
  }

  function renderStrip() {
    const pal = palette();
    const { items, values, topRank } = candidates();
    const context = data.context || [];
    const n = values.length;
    hitRows = [];
    if (!n) {
      drawEmpty(searchTerm ? "没有匹配的 Key" : "等待生成");
      return;
    }
    const pad = cssW < 480 ? 10 : 18;
    const maxW = Math.max(1e-9, ...values);
    const colorScale = settings.normalizeColor ? maxW : 1;
    const promptLen = clamp(Number(data.promptLen) || 0, 0, n);
    const qIndex = n - 1;
    const heat = (w) => Math.sqrt(clamp(w / colorScale, 0, 1));
    const matches = (i) => !searchTerm || tokenText(context[i] !== undefined ? context[i] : `#${i}`).toLowerCase().includes(searchTerm.toLowerCase());

    ctx.textBaseline = "middle";
    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.fillStyle = inkMuted;
    ctx.textAlign = "left";
    ctx.fillText("上下文 · 芯片热度 = 注意力权重", pad, 16);
    ctx.textAlign = "right";
    ctx.fillText("末端 = 当前 Query → 预测", cssW - pad, 16);

    const predW = clamp(cssW * 0.14, 76, 120);
    const predX = cssW - pad - predW;
    const predY = 30;
    const predH = 40;
    const drawPred = () => {
      ctx.fillStyle = (canvasStyle && canvasStyle.getPropertyValue("--bg-raised").trim()) || "rgba(33,36,41,0.92)";
      ctx.strokeStyle = rgb(samplePalette(pal.stops, 0.9), 0.8);
      ctx.lineWidth = 1.4;
      roundRect(predX, predY, predW, predH, 4);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = pal.accent;
      ctx.font = canvasFont("data", fontTokens().size.md, fontTokens().weight.bold);
      ctx.textAlign = "center";
      ctx.fillText(`→ ${tokenLabel(data.predictedToken || data.queryToken, 6)}`, predX + predW / 2, predY + 14);
      ctx.fillStyle = inkSoft;
      ctx.font = canvasFont("data", fontTokens().size.xs);
      ctx.fillText("预测 token", predX + predW / 2, predY + 29);
    };
    const drawTokenChip = (x, y, w, h, i, wgt, compact) => {
      const c = samplePalette(pal.stops, 0.18 + 0.82 * heat(wgt));
      const isQ = i === qIndex;
      const rank = topRank.get(i);
      const isSel = selectedKey === i;
      const isHover = hoverKey === i;
      ctx.globalAlpha = matches(i) ? 1 : 0.25;
      ctx.fillStyle = rgb(c, 0.92);
      roundRect(x, y, w, h, 4);
      ctx.fill();
      ctx.lineWidth = isQ || isSel ? 2 : rank ? 1.4 : 1;
      ctx.strokeStyle = isQ ? "#ececef" : isSel ? pal.accent : rank ? rgb(samplePalette(pal.stops, 0.95), 0.9) : i < promptLen ? "rgba(154,157,166,0.5)" : "rgba(124,147,240,0.4)";
      roundRect(x, y, w, h, 4);
      ctx.stroke();
      if (isHover) {
        ctx.fillStyle = "rgba(248,250,252,0.12)";
        roundRect(x, y, w, h, 4);
        ctx.fill();
      }
      const lum = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
      ctx.fillStyle = lum > 0.55 ? "rgba(8,12,20,0.92)" : "rgba(235,242,250,0.95)";
      ctx.font = canvasFont("data", compact ? fontTokens().size.xs : fontTokens().size.sm, isQ || rank === 1 ? fontTokens().weight.bold : fontTokens().weight.regular);
      ctx.textAlign = "center";
      ctx.fillText(tokenLabel(context[i] !== undefined ? context[i] : `#${i}`, compact ? 2 : 4), x + w / 2, y + (compact ? h / 2 : 16));
      if (!compact && settings.showValues) {
        ctx.font = canvasFont("data", fontTokens().size.xs);
        ctx.fillText(pctText(wgt), x + w / 2, y + 34);
      }
      if (!compact && rank) {
        ctx.fillStyle = lum > 0.55 ? "rgba(8,12,20,0.7)" : "rgba(235,242,250,0.7)";
        ctx.font = canvasFont("data", fontTokens().size.xs);
        ctx.textAlign = "left";
        ctx.fillText(`#${rank}`, x + 4, y + h - 8);
      }
      if (isQ) {
        ctx.fillStyle = pal.accent;
        ctx.font = canvasFont("data", fontTokens().size.xs);
        ctx.textAlign = "center";
        ctx.fillText("Q", x + w / 2, y - 7);
      }
      ctx.globalAlpha = 1;
      hitRows.push({ index: i, x, y, width: w, height: h, weight: wgt, token: context[i] !== undefined ? context[i] : `#${i}` });
    };

    const gap = 4;
    const maxChips = Math.max(4, Math.floor((predX - pad - 16 + gap) / (26 + gap)));
    if (n <= maxChips) {
      // 短上下文：整段按位置排成一行芯片，末端接预测芯片。
      const chipW = clamp((predX - pad - 16 - gap * (n - 1)) / n, 26, 64);
      const chipH = 56;
      const chipY = 96;
      for (let i = 0; i < n; i += 1) {
        drawTokenChip(pad + i * (chipW + gap), chipY, chipW, chipH, i, values[i], chipW < 34);
      }
      const qX = pad + qIndex * (chipW + gap) + chipW;
      ctx.strokeStyle = rgb(samplePalette(pal.stops, 0.9), 0.8);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.moveTo(qX + 2, chipY + chipH / 2);
      ctx.bezierCurveTo(qX + 10, chipY + chipH / 2, predX - 10, predY + predH / 2, predX - 2, predY + predH / 2);
      ctx.stroke();
      drawPred();
      // 底部：权重最高的 Key 排名行，直接回答「看了谁」。
      ctx.fillStyle = inkMuted;
      ctx.font = canvasFont("ui", fontTokens().size.xs);
      ctx.textAlign = "left";
      ctx.fillText("权重最高的 Key", pad, 190);
      const topSlots = Math.max(2, Math.min(8, Math.floor((cssW - pad * 2 + gap) / (60 + gap))));
      const topItems = items.slice(0, topSlots);
      const tW = clamp((cssW - pad * 2 - gap * (topSlots - 1)) / topSlots, 48, 150);
      topItems.forEach((item, r) => {
        const x = pad + r * (tW + gap);
        const y = 204;
        const h = 52;
        const c = samplePalette(pal.stops, 0.18 + 0.82 * heat(item.weight));
        const isSel = selectedKey === item.index;
        ctx.fillStyle = rgb(c, 0.92);
        roundRect(x, y, tW, h, 4);
        ctx.fill();
        ctx.strokeStyle = isSel ? pal.accent : rgb(samplePalette(pal.stops, 0.95), 0.7);
        ctx.lineWidth = isSel ? 2 : 1.2;
        roundRect(x, y, tW, h, 4);
        ctx.stroke();
        const lum = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
        ctx.fillStyle = lum > 0.55 ? "rgba(8,12,20,0.92)" : "rgba(235,242,250,0.95)";
        ctx.font = canvasFont("data", fontTokens().size.sm, fontTokens().weight.bold);
        ctx.textAlign = "center";
        ctx.fillText(`${r + 1} · ${tokenLabel(item.token, 4)}`, x + tW / 2, y + 16);
        if (settings.showValues) {
          ctx.font = canvasFont("data", fontTokens().size.xs);
          ctx.fillText(pctText(item.weight), x + tW / 2, y + 34);
        }
        hitRows.push({ index: item.index, x, y, width: tW, height: h, weight: item.weight, token: item.token });
      });
      ctx.fillStyle = "rgba(154,157,166,0.7)";
      ctx.font = canvasFont("ui", fontTokens().size.xs);
      ctx.textAlign = "left";
      ctx.fillText("灰边框 = prompt · 靛边框 = 已生成 · 白描边 = 当前 Query · 亮描边 = Top-K", pad, cssH - 14);
      return;
    }

    // 长上下文：聚合热带（≤64 bin，bin 内权重求和）+ Top-K 芯片行 + Q/预测芯片。
    const bins = Math.min(64, n);
    const bandX = pad;
    const bandW = predX - pad - 16;
    const binW = bandW / bins;
    const bandY = 96;
    const bandH = 44;
    const sums = new Array(bins).fill(0);
    for (let i = 0; i < n; i += 1) sums[Math.min(bins - 1, Math.floor((i * bins) / n))] += values[i];
    const maxSum = Math.max(1e-9, ...sums);
    for (let b = 0; b < bins; b += 1) {
      const lo = Math.floor((b * n) / bins);
      const hi = Math.max(lo + 1, Math.floor(((b + 1) * n) / bins));
      const c = samplePalette(pal.stops, 0.15 + 0.85 * Math.sqrt(clamp(sums[b] / (settings.normalizeColor ? maxSum : 1), 0, 1)));
      ctx.fillStyle = rgb(c, 0.92);
      ctx.fillRect(bandX + b * binW, bandY, Math.max(1, binW - 1), bandH);
      hitRows.push({ index: lo, x: bandX + b * binW, y: bandY, width: binW, height: bandH, weight: sums[b], token: `${tokenLabel(context[lo], 4)}…${tokenLabel(context[hi - 1], 4)}`, bin: [lo, hi] });
    }
    ctx.strokeStyle = "rgba(154,157,166,0.35)";
    ctx.lineWidth = 1;
    ctx.strokeRect(bandX - 0.5, bandY - 0.5, bandW + 1, bandH + 1);
    ctx.fillStyle = "rgba(154,157,166,0.7)";
    ctx.font = canvasFont("data", fontTokens().size.xs);
    ctx.textAlign = "left";
    ctx.fillText(`位置 0（${bins} 段聚合 · 段内权重求和）`, bandX, bandY + bandH + 14);
    ctx.textAlign = "right";
    ctx.fillText(`位置 ${n - 1}`, bandX + bandW, bandY + bandH + 14);
    drawPred();
    ctx.fillStyle = inkMuted;
    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.textAlign = "left";
    ctx.fillText("权重最高的 Key", pad, 190);
    const topSlots = Math.max(2, Math.min(8, Math.floor((cssW - pad * 2 + gap) / (60 + gap))));
    const topItems = items.slice(0, topSlots);
    const tW = clamp((cssW - pad * 2 - gap * (topSlots - 1)) / topSlots, 48, 150);
    topItems.forEach((item, r) => {
      const x = pad + r * (tW + gap);
      const y = 204;
      const h = 52;
      const c = samplePalette(pal.stops, 0.18 + 0.82 * heat(item.weight));
      const isSel = selectedKey === item.index;
      ctx.fillStyle = rgb(c, 0.92);
      roundRect(x, y, tW, h, 4);
      ctx.fill();
      ctx.strokeStyle = isSel ? pal.accent : rgb(samplePalette(pal.stops, 0.95), 0.7);
      ctx.lineWidth = isSel ? 2 : 1.2;
      roundRect(x, y, tW, h, 4);
      ctx.stroke();
      const lum = (0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]) / 255;
      ctx.fillStyle = lum > 0.55 ? "rgba(8,12,20,0.92)" : "rgba(235,242,250,0.95)";
      ctx.font = canvasFont("data", fontTokens().size.sm, fontTokens().weight.bold);
      ctx.textAlign = "center";
      ctx.fillText(`${r + 1} · ${tokenLabel(item.token, 4)}`, x + tW / 2, y + 16);
      if (settings.showValues) {
        ctx.font = canvasFont("data", fontTokens().size.xs);
        ctx.fillText(pctText(item.weight), x + tW / 2, y + 34);
      }
      hitRows.push({ index: item.index, x, y, width: tW, height: h, weight: item.weight, token: item.token });
    });
    // 当前 Query 芯片（长序列下单独标出）。
    drawTokenChip(pad, 286, 64, 48, qIndex, values[qIndex], false);
    ctx.fillStyle = "rgba(154,157,166,0.7)";
    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.textAlign = "left";
    ctx.fillText("上下文过长：热带按段聚合，精确权重看 Top-K 芯片或悬浮热带", pad + 76, 310);
  }

  function renderLinks() {
    const pal = palette();
    const { items, values } = candidates();
    hitRows = [];
    if (!items.length) {
      drawEmpty(searchTerm ? "没有匹配的 Key" : "没有 Key 达到当前阈值");
      return;
    }
    const pad = cssW < 480 ? 10 : 18;
    const gap = cssW < 480 ? 16 : 34;
    const qW = clamp(cssW * (cssW < 480 ? 0.24 : 0.2), cssW < 480 ? 64 : 96, 160);
    const outW = clamp(cssW * (cssW < 480 ? 0.18 : 0.16), cssW < 480 ? 54 : 76, 130);
    const kX = pad + qW + gap;
    const kWidth = Math.max(80, cssW - kX - outW - pad - (cssW < 480 ? 12 : 26));
    const top = 34;
    const bottom = cssH - 18;
    const rowH = clamp((bottom - top) / items.length, 18, 56);
    const qY = top + ((bottom - top) / 2);
    const maxW = Math.max(1e-9, ...values);
    const topMax = Math.max(1e-9, ...items.map((item) => item.weight));
    const colorScale = settings.normalizeColor ? maxW : 1;

    ctx.textBaseline = "middle";
    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.fillStyle = inkMuted;
    ctx.textAlign = "left";
    ctx.fillText("Query", pad, 16);
    ctx.fillText("Key · 真实注意力权重", kX, 16);
    ctx.textAlign = "right";
    ctx.fillText("加权 Value", cssW - pad, 16);

    // Q 卡片
    const qH = 64;
    ctx.fillStyle = (canvasStyle && canvasStyle.getPropertyValue("--bg-raised").trim()) || "rgba(33,36,41,0.92)";
    ctx.strokeStyle = rgb(samplePalette(pal.stops, 0.8), 0.65);
    ctx.lineWidth = 1.2;
    roundRect(pad, qY - qH / 2, qW, qH, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = pal.accent;
    ctx.font = canvasFont("data", fontTokens().size.lg, fontTokens().weight.bold);
    ctx.textAlign = "center";
    ctx.fillText(tokenLabel(data.queryToken, Math.max(3, Math.floor(qW / 9))), pad + qW / 2, qY - 8);
    ctx.fillStyle = "rgba(196,198,204,0.8)";
    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.fillText(qW < 90 ? "下一 token" : "预测下一个 token", pad + qW / 2, qY + 12);

    const outX = cssW - pad - outW / 2;
    const outY = qY;

    items.forEach((item, i) => {
      const y = top + i * rowH + rowH / 2;
      const trueRatio = clamp(item.weight, 0, 1);
      const rel = clamp(item.weight / topMax, 0, 1);
      const ratio = settings.normalizeColor ? rel : Math.sqrt(trueRatio);
      const c = samplePalette(pal.stops, 0.25 + 0.75 * ratio);
      const isSelected = selectedKey === item.index;
      const isHover = hoverKey === item.index;
      const lineWidth = settings.stroke * (0.5 + 1.1 * rel);

      // Q → K
      ctx.strokeStyle = rgb(c, isSelected || isHover ? 0.98 : 0.85);
      ctx.lineWidth = lineWidth;
      ctx.beginPath();
      ctx.moveTo(pad + qW, qY);
      const midX = (pad + qW + kX) / 2;
      ctx.bezierCurveTo(midX, qY, midX, y, kX, y);
      ctx.stroke();

      // K 行：按可用宽度分配标签 / 索引 / 权重条 / 数值，窄屏自动压缩。
      const valueW = settings.showValues ? (kWidth < 150 ? 34 : 46) : 0;
      const indexW = kWidth < 150 ? 0 : 30;
      const barW = clamp(kWidth * 0.42, 24, 260);
      const labelW = Math.max(24, kWidth - barW - valueW - indexW - 8);
      if (isSelected || isHover) {
        ctx.fillStyle = rgb(c, 0.1);
        roundRect(kX - 6, y - rowH / 2 + 2, kWidth + 12, rowH - 4, 4);
        ctx.fill();
      }
      ctx.fillStyle = isSelected ? pal.accent : ink;
      ctx.font = canvasFont("data", fontTokens().size.sm, isSelected ? fontTokens().weight.bold : fontTokens().weight.regular);
      ctx.textAlign = "left";
      ctx.fillText(tokenLabel(item.token, Math.max(3, Math.floor(labelW / 8))), kX, y);
      if (indexW) {
        ctx.fillStyle = "rgba(154,157,166,0.75)";
        ctx.font = canvasFont("data", fontTokens().size.xs);
        ctx.fillText(`[${item.index}]`, kX + labelW + 4, y);
      }

      const barX = kX + labelW + indexW;
      ctx.fillStyle = "rgba(154,157,166,0.14)";
      roundRect(barX, y - 4, barW, 8, 2);
      ctx.fill();
      ctx.fillStyle = rgb(c, 0.95);
      roundRect(barX, y - 4, Math.max(3, barW * rel), 8, 2);
      ctx.fill();

      if (settings.showValues) {
        ctx.fillStyle = ink;
        ctx.font = canvasFont("ui", fontTokens().size.xs);
        ctx.textAlign = "right";
        ctx.fillText(pctText(item.weight), kX + kWidth, y);
      }

      // K → ΣαV
      ctx.strokeStyle = rgb(c, 0.35 + 0.5 * rel);
      ctx.lineWidth = Math.max(1, lineWidth * 0.7);
      ctx.beginPath();
      ctx.moveTo(kX + kWidth + 6, y);
      const mid2 = (kX + kWidth + outX) / 2;
      ctx.bezierCurveTo(mid2, y, mid2, outY, outX - outW / 2, outY);
      ctx.stroke();

      hitRows.push({ index: item.index, y, height: rowH, weight: item.weight, token: item.token });
    });

    // 输出节点
    ctx.fillStyle = (canvasStyle && canvasStyle.getPropertyValue("--bg-raised").trim()) || "rgba(33,36,41,0.92)";
    ctx.strokeStyle = rgb(samplePalette(pal.stops, 0.9), 0.7);
    ctx.lineWidth = 1.2;
    roundRect(outX - outW / 2, outY - 32, outW, 64, 4);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = pal.accent;
    ctx.font = canvasFont("data", fontTokens().size.md, fontTokens().weight.bold);
    ctx.textAlign = "center";
    ctx.fillText(`→ ${tokenLabel(data.predictedToken || data.queryToken, 6)}`, outX, outY - 6);
    ctx.fillStyle = "rgba(196,198,204,0.8)";
    ctx.font = canvasFont("data", fontTokens().size.xs);
    ctx.fillText("Σ αᵢ·V · 预测", outX, outY + 14);
  }

  function renderMatrix() {
    const pal = palette();
    hitRows = [];
    const matrix = currentMatrix();
    const values = currentRows();
    const win = clamp(settings.window, 4, MAX_MATRIX_WINDOW);
    let grid;
    let rowLabels;
    let colLabels;
    let title;
    if (matrix && matrix.length) {
      // 生成完成：final 全量矩阵，行/列均为窗口内绝对位置。
      const context = data.matrixContext || data.context || [];
      const n = matrix.length;
      const start = Math.max(0, n - win);
      // 因果遮罩：列绝对位置 > 行绝对位置。
      grid = matrix.slice(start).map((row, i) => (row || []).slice(start, start + win).map((v, j) => ({
        v: Number(v) || 0,
        masked: start + j > start + i,
        untransmitted: false,
      })));
      rowLabels = context.slice(start, start + win);
      colLabels = context.slice(start, start + win);
      title = `局部矩阵 · 最近 ${grid.length}×${win} 窗口`;
    } else if (data.history && data.history.length) {
      // 播放中：用已揭示步的 attn_last 按绝对位置累积多行。
      const l = clamp(settings.layer, 0, Math.max(0, (data.rows || []).length - 1));
      const h = clamp(settings.head, 0, Math.max(0, ((data.rows || [])[l] || []).length - 1));
      const hist = data.history.slice(-win);
      const last = hist[hist.length - 1];
      const colStart = Math.max(0, last.abs + 1 - win);
      const tokenAt = new Map();
      hist.forEach((entry) => (entry.context || []).forEach((tok, i) => tokenAt.set(entry.start + i, tok)));
      grid = hist.map((entry) => {
        const rowVals = ((entry.rows || [])[l] || [])[h] || [];
        return Array.from({ length: win }, (_, j) => {
          const absCol = colStart + j;
          if (absCol > entry.abs) return { v: 0, masked: true, untransmitted: false };
          if (absCol < entry.start) return { v: 0, masked: false, untransmitted: true };
          return { v: Number(rowVals[absCol - entry.start]) || 0, masked: false, untransmitted: false };
        });
      });
      rowLabels = hist.map((entry) => tokenAt.get(entry.abs));
      colLabels = Array.from({ length: win }, (_, j) => tokenAt.get(colStart + j));
      title = `局部矩阵 · 播放中累积 ${hist.length}×${win} · 斜线 = 因果遮罩`;
    } else {
      const start = Math.max(0, values.length - win);
      grid = [values.slice(start, start + win).map((v, j) => ({ v, masked: false, untransmitted: false }))];
      rowLabels = [data.queryToken || "Q"];
      colLabels = (data.context || []).slice(start, start + win);
      title = "实时仅传输当前 Query 行";
    }
    const rowsN = grid.length;
    const colsN = grid[0] ? grid[0].length : 0;
    if (!colsN) {
      drawEmpty("暂无可显示的矩阵窗口");
      return;
    }
    const left = 86;
    const top = 64;
    const cell = clamp(Math.floor(Math.min((cssW - left - 24) / colsN, (cssH - top - 40) / rowsN)), 8, 30);
    const gridW = cell * colsN;
    const gridH = cell * rowsN;
    const mx = left;
    const my = top;
    const flat = grid.flat().filter((c) => !c.masked && !c.untransmitted).map((c) => c.v);
    const maxVal = settings.normalizeColor ? Math.max(1e-9, ...flat) : 1;

    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.fillStyle = inkMuted;
    ctx.textAlign = "left";
    ctx.fillText(title, 12, 18);

    for (let i = 0; i < rowsN; i += 1) {
      for (let j = 0; j < colsN; j += 1) {
        const cellV = grid[i][j];
        const ratio = settings.normalizeColor ? clamp(cellV.v / maxVal, 0, 1) : Math.sqrt(clamp(cellV.v, 0, 1));
        const c = samplePalette(pal.stops, ratio);
        const x = mx + j * cell;
        const y = my + i * cell;
        ctx.fillStyle = cellV.masked ? "rgba(33,36,41,0.55)" : cellV.untransmitted ? "rgba(11,12,14,0.7)" : rgb(c, 0.95);
        ctx.fillRect(x, y, cell - 1, cell - 1);
        if (cellV.masked || cellV.untransmitted) {
          ctx.strokeStyle = cellV.masked ? "rgba(154,157,166,0.25)" : "rgba(154,157,166,0.12)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x, y + cell - 1);
          ctx.lineTo(x + cell - 1, y);
          ctx.stroke();
        }
        if (settings.showValues && cell >= 20 && !cellV.masked && !cellV.untransmitted && cellV.v >= settings.threshold) {
          ctx.fillStyle = ratio > 0.55 ? "rgba(8,12,20,0.9)" : "rgba(236,236,239,0.85)";
          // 字号随格子缩放：小格用更小的字，避免数值横向粘连。
          ctx.font = canvasFont("data", Math.max(7, Math.min(fontTokens().size.xs, Math.round(cell * 0.3))));
          ctx.textAlign = "center";
          ctx.textBaseline = "middle";
          ctx.fillText(cellV.v.toFixed(2), x + cell / 2, y + cell / 2);
        }
      }
    }

    ctx.font = canvasFont("ui", fontTokens().size.xs);
    ctx.textBaseline = "middle";
    const labelEvery = Math.max(1, Math.ceil(rowsN / 10));
    for (let i = 0; i < rowsN; i += 1) {
      if (i % labelEvery !== 0 && i !== rowsN - 1) continue;
      ctx.textAlign = "right";
      ctx.fillStyle = inkSoft;
      ctx.fillText(tokenLabel(rowLabels[i], 8), mx - 8, my + i * cell + cell / 2);
    }
    const colEvery = Math.max(1, Math.ceil(colsN / 10));
    for (let j = 0; j < colsN; j += 1) {
      if (j % colEvery !== 0 && j !== colsN - 1) continue;
      ctx.save();
      ctx.translate(mx + j * cell + cell / 2, my - 8);
      ctx.rotate(-Math.PI / 4);
      ctx.textAlign = "left";
      ctx.fillStyle = inkSoft;
      ctx.fillText(tokenLabel(colLabels[j], 6), 0, 0);
      ctx.restore();
    }
    ctx.strokeStyle = "rgba(154,157,166,0.3)";
    ctx.lineWidth = 1;
    ctx.strokeRect(mx - 0.5, my - 0.5, gridW + 1, gridH + 1);
  }

  function roundRect(x, y, w, h, r) {
    const rr = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.arcTo(x + w, y, x + w, y + h, rr);
    ctx.arcTo(x + w, y + h, x, y + h, rr);
    ctx.arcTo(x, y + h, x, y, rr);
    ctx.arcTo(x, y, x + w, y, rr);
    ctx.closePath();
  }

  function render(canvasOnly) {
    if (destroyed) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    if (settings.mode === "strip") renderStrip();
    else if (settings.mode === "links") renderLinks();
    else renderMatrix();
    // hover 高频路径只重绘画布，不重建 DOM 文本。
    if (!canvasOnly) {
      updateText();
      syncLegend();
    }
  }

  function hitTest(x, y) {
    for (const row of hitRows) {
      if (row.x !== undefined) {
        if (x >= row.x && x <= row.x + row.width && y >= row.y && y <= row.y + row.height) return row;
      } else if (Math.abs(y - row.y) <= row.height / 2) return row;
    }
    return null;
  }

  function showTooltip(row, x, y) {
    if (!tooltip || !row) return;
    tooltip.hidden = false;
    tooltip.textContent = row.bin
      ? `位置 ${row.bin[0]}–${row.bin[1] - 1} · 段内权重和=${row.weight.toFixed(4)}`
      : `K[${row.index}] ${tokenLabel(row.token, 12)} · α=${row.weight.toFixed(4)}`;
    tooltip.style.left = `${clamp(x + 12, 4, cssW - 180)}px`;
    tooltip.style.top = `${clamp(y - 30, 4, cssH - 30)}px`;
  }

  listen(canvas, "mousemove", (event) => {
    if (settings.mode === "matrix") return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const row = hitTest(x, y);
    const nextHover = row ? row.index : null;
    canvas.style.cursor = row ? "pointer" : "default";
    if (row) showTooltip(row, x, y);
    else if (tooltip) tooltip.hidden = true;
    if (nextHover !== hoverKey) {
      hoverKey = nextHover;
      // RAF 节流：一帧内多次 mousemove 只重绘一次。
      if (!hoverRaf) {
        hoverRaf = requestAnimationFrame(() => {
          hoverRaf = 0;
          render(true);
          // 详情条同步悬浮项的精确权重（不重建指标 DOM）。
          const detail = $("attn-selected-detail");
          if (detail) {
            const { items } = candidates();
            const item = items.find((x) => x.index === selectedKey) || (hoverKey !== null ? items.find((x) => x.index === hoverKey) : null);
            if (item) {
              detail.textContent = `K[${item.index}]「${tokenLabel(item.token, 12)}」 · α=${item.weight.toFixed(4)}（${pctText(item.weight)}） · 贡献 α·V`;
            }
          }
        });
      }
    }
  });

  listen(canvas, "mouseleave", () => {
    hoverKey = null;
    if (tooltip) tooltip.hidden = true;
    if (settings.mode !== "matrix") render(true);
  });

  listen(canvas, "click", (event) => {
    if (settings.mode === "matrix") return;
    const rect = canvas.getBoundingClientRect();
    const row = hitTest(event.clientX - rect.left, event.clientY - rect.top);
    if (!row) return;
    selectedKey = selectedKey === row.index ? null : row.index;
    render();
  });

  listen(canvas, "keydown", (event) => {
    if (settings.mode === "matrix" || !hitRows.length) return;
    const forward = event.key === "ArrowDown" || event.key === "ArrowRight";
    const backward = event.key === "ArrowUp" || event.key === "ArrowLeft";
    if (!forward && !backward && event.key !== "Enter" && event.key !== " ") return;
    event.preventDefault();
    let idx = hitRows.findIndex((r) => r.index === selectedKey);
    if (forward) idx = idx < 0 ? 0 : Math.min(hitRows.length - 1, idx + 1);
    else if (backward) idx = idx < 0 ? hitRows.length - 1 : Math.max(0, idx - 1);
    else idx = idx < 0 ? 0 : idx;
    selectedKey = hitRows[idx].index;
    render();
  });

  function bindRange(id, key, fmt) {
    const input = $(id);
    const out = $(`${id}-out`);
    if (!input) return;
    const apply = () => {
      settings[key] = Number(input.value);
      invalidateCaches();
      if (out) out.textContent = fmt ? fmt(settings[key]) : String(settings[key]);
      render();
    };
    listen(input, "input", apply);
    apply();
  }

  function bindSelect(id, key, onChange) {
    const select = $(id);
    if (!select) return;
    const apply = () => {
      settings[key] = select.multiple ? select.value : (isNaN(Number(select.value)) ? select.value : Number(select.value));
      invalidateCaches();
      if (onChange) onChange();
      render();
    };
    listen(select, "change", apply);
    apply();
  }

  function bindCheck(id, key, onChange) {
    const input = $(id);
    if (!input) return;
    const apply = () => {
      settings[key] = input.checked;
      invalidateCaches();
      if (onChange) onChange();
      render();
    };
    listen(input, "change", apply);
    apply();
  }

  function syncLegend() {
    const legend = $("attn-legend-bar");
    if (legend) {
      const stops = palette().stops.map((c) => rgb(c)).join(", ");
      legend.style.background = `linear-gradient(90deg, ${stops})`;
    }
    const legendMax = $("attn-legend-max");
    if (legendMax) {
      const vals = currentRows();
      const maxW = vals.length ? Math.max(...vals) : 1;
      legendMax.textContent = settings.normalizeColor ? `行最大 ${pctText(maxW)}` : "100%";
    }
  }

  function syncControls() {
    const setVal = (id, value) => { const n = $(id); if (n) n.value = String(value); };
    const setCheck = (id, value) => { const n = $(id); if (n) n.checked = Boolean(value); };
    setVal("attn-topk", settings.topK);
    setVal("attn-threshold", settings.threshold);
    setVal("attn-order", settings.order);
    setVal("attn-palette", settings.palette);
    setVal("attn-window", settings.window);
    setCheck("attn-follow", settings.follow);
    setCheck("attn-values", settings.showValues);
    setCheck("attn-normalize", settings.normalizeColor);
    const thrOut = $("attn-threshold-out");
    if (thrOut) thrOut.textContent = pctText(settings.threshold);
    const links = $("attn-mode-links");
    const matrix = $("attn-mode-matrix");
    const strip = $("attn-mode-strip");
    if (links) links.classList.toggle("active", settings.mode === "links");
    if (matrix) matrix.classList.toggle("active", settings.mode === "matrix");
    if (strip) strip.classList.toggle("active", settings.mode === "strip");
    syncLegend();
    populateSelectors(true);
  }

  function bind() {
    bindSelect("attn-layer", "layer", () => { settings.head = 0; populateSelectors(true); });
    bindSelect("attn-head", "head");
    bindSelect("attn-topk", "topK");
    bindRange("attn-threshold", "threshold", (v) => pctText(v));
    bindSelect("attn-order", "order");
    bindSelect("attn-palette", "palette", syncLegend);
    bindSelect("attn-window", "window");
    bindCheck("attn-follow", "follow", () => { if (options.onFollowChange) options.onFollowChange(settings.follow); });
    bindCheck("attn-values", "showValues");
    bindCheck("attn-normalize", "normalizeColor");

    const links = $("attn-mode-links");
    const matrix = $("attn-mode-matrix");
    const strip = $("attn-mode-strip");
    const setMode = (mode) => {
      settings.mode = mode;
      if (links) links.classList.toggle("active", mode === "links");
      if (matrix) matrix.classList.toggle("active", mode === "matrix");
      if (strip) strip.classList.toggle("active", mode === "strip");
      selectedKey = null;
      render();
    };
    if (links) listen(links, "click", () => setMode("links"));
    if (matrix) listen(matrix, "click", () => setMode("matrix"));
    if (strip) listen(strip, "click", () => setMode("strip"));

    const search = $("attn-search");
    if (search) listen(search, "input", () => { searchTerm = search.value.trim(); selectedKey = null; invalidateCaches(); render(); });

    const prev = $("attn-query-prev");
    const next = $("attn-query-next");
    const latest = $("attn-query-latest");
    if (prev) listen(prev, "click", () => { if (options.onNavigate) options.onNavigate(-1); });
    if (next) listen(next, "click", () => { if (options.onNavigate) options.onNavigate(1); });
    if (latest) listen(latest, "click", () => { if (options.onLatest) options.onLatest(); });

    const reset = $("attn-reset");
    if (reset) listen(reset, "click", () => {
      Object.assign(settings, DEFAULTS);
      searchTerm = "";
      selectedKey = null;
      invalidateCaches();
      const searchInput = $("attn-search");
      if (searchInput) searchInput.value = "";
      syncControls();
      if (options.onFollowChange) options.onFollowChange(true);
      render();
    });

    const exportBtn = $("attn-export");
    if (exportBtn) listen(exportBtn, "click", () => {
      try {
        const headerH = 56;
        const footerH = 40;
        const off = document.createElement("canvas");
        off.width = canvas.width;
        off.height = canvas.height + (headerH + footerH) * dpr;
        const octx = off.getContext("2d");
        octx.setTransform(dpr, 0, 0, dpr, 0, 0);
        octx.fillStyle = (canvasStyle && canvasStyle.getPropertyValue("--bg-inset").trim()) || "#0b0c0e";
        octx.fillRect(0, 0, cssW, cssH + headerH + footerH);
        octx.fillStyle = ink;
        octx.font = canvasFont("data", fontTokens().size.md, fontTokens().weight.bold);
        octx.textAlign = "left";
        octx.textBaseline = "middle";
        const modeName = settings.mode === "strip" ? "上下文条带" : settings.mode === "links" ? "Top-K 连线" : "局部矩阵";
        octx.fillText(`MiniGPT 自注意力 · ${modeName} · L${settings.layer}H${settings.head}`, 16, 20);
        octx.fillStyle = inkSoft;
        octx.font = canvasFont("ui", fontTokens().size.xs);
        octx.fillText(`第 ${data.step || 0} 步 · Q「${tokenLabel(data.queryToken, 8)}」→ 预测「${tokenLabel(data.predictedToken || data.queryToken, 8)}」`, 16, 40);
        octx.drawImage(canvas, 0, headerH, cssW, cssH);
        const pal = palette();
        const barX = 60;
        const barW = Math.max(80, cssW - 220);
        for (let x = 0; x < barW; x += 2) {
          octx.fillStyle = rgb(samplePalette(pal.stops, x / barW), 1);
          octx.fillRect(barX + x, cssH + headerH + 14, 2, 10);
        }
        octx.fillStyle = inkSoft;
        octx.font = canvasFont("ui", fontTokens().size.xs);
        octx.textAlign = "left";
        octx.fillText("权重", 16, cssH + headerH + 19);
        octx.fillText("0", barX, cssH + headerH + 32);
        octx.textAlign = "right";
        const vals = currentRows();
        const maxW = vals.length ? Math.max(...vals) : 1;
        octx.fillText(settings.normalizeColor ? `行最大 ${pctText(maxW)}` : "100%", barX + barW, cssH + headerH + 32);
        const a = document.createElement("a");
        a.href = off.toDataURL("image/png");
        a.download = `minigpt-attention-${settings.mode}-L${settings.layer}H${settings.head}.png`;
        document.body.appendChild(a);
        a.click();
        a.remove();
      } catch (error) {
        try {
          const a = document.createElement("a");
          a.href = canvas.toDataURL("image/png");
          a.download = `minigpt-attention-${settings.mode}-L${settings.layer}H${settings.head}.png`;
          document.body.appendChild(a);
          a.click();
          a.remove();
        } catch (fallbackError) { /* 导出失败时保持页面可用 */ }
      }
    });
  }

  bind();
  syncControls();
  populateSelectors(true);
  resize();

  let resizeObserver = null;
  if (typeof ResizeObserver !== "undefined") {
    resizeObserver = new ResizeObserver(scheduleResize);
    resizeObserver.observe(wrap);
  } else {
    listen(window, "resize", scheduleResize);
  }
  // 主题切换：画布文字/卡片颜色与字体令牌取自 CSS 变量，需要重取并重绘。
  disposers.push(on("theme", () => {
    if (destroyed) return;
    refreshInk();
    invalidateFontTokens();
    invalidateCaches();
    render();
  }));

  return {
    setData(next) {
      const incoming = next && next.rows && next.rows.length ? next : exampleData();
      if (incoming.seriesId !== data.seriesId) {
        selectedKey = null;
        hoverKey = null;
      }
      data = incoming;
      invalidateCaches();
      if (selectedKey !== null && selectedKey >= (data.context || []).length) selectedKey = null;
      populateSelectors(false);
      render();
    },
    setFollow(value) {
      settings.follow = Boolean(value);
      const input = $("attn-follow");
      if (input) input.checked = settings.follow;
      updateText();
    },
    refresh() { scheduleResize(); },
    destroy() {
      destroyed = true;
      if (hoverRaf) cancelAnimationFrame(hoverRaf);
      if (resizeObserver) resizeObserver.disconnect();
      disposers.forEach((dispose) => dispose());
      disposers.length = 0;
    },
  };
}