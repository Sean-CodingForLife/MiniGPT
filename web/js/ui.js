/* 各视图共用的 UI 基座：DOM/格式化、三态渲染、toast、应用内对话框、
 * 差量渲染与下拉同步、列表键盘操作。
 * 全部原生 API，无外部依赖；数据一律 textContent 写入，不执行训练产物里的文本。
 */

import { getJson, postJson, putJson, deleteJson, requestJson, ApiError } from "./api.js";

/* 兼容旧调用方：请求函数统一从 api.js 出口，这里再导出一次。 */
export { getJson, postJson, putJson, deleteJson, requestJson, ApiError };

export const byId = (id) => document.getElementById(id);

/**
 * 创建元素并设置 className 与文本（文本走 textContent，防 XSS）。
 */
export function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

/**
 * 格式化普通数字，非法值显示占位符。
 */
export function fmtNumber(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "?";
  return number.toFixed(digits);
}

/** 智能数字：整数原样、小数按位；非数值返回占位符（训练面板用）。 */
export function fmtSmart(value, digits = 4, fallback = "—") {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return fallback;
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(digits);
  }
  if (typeof value === "string" && value.trim() !== "") return value;
  return fallback;
}

/**
 * 整数千分位格式化。
 */
export function fmtInt(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "?";
  return number.toLocaleString();
}

/**
 * 把字节数转换成 KB/MB/GB 文本。
 */
export function bytesText(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = number;
  let unit = 0;
  while (size >= 1024 && unit < units.length - 1) {
    size /= 1024;
    unit += 1;
  }
  return `${size.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

/**
 * 把比率格式化为百分比文本。
 */
export function pct(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "?";
  return `${(number * 100).toFixed(digits)}%`;
}

/**
 * 把 ISO 时间戳格式化为本地可读时间。
 */
export function fmtTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (isNaN(date.getTime())) return String(value);
  return date.toLocaleString();
}

/**
 * 把秒数格式化为 1h02m03s 形式。
 */
export function fmtElapsed(seconds) {
  const value = Number(seconds);
  if (!Number.isFinite(value) || value < 0) return "—";
  const h = Math.floor(value / 3600);
  const m = Math.floor((value % 3600) / 60);
  const s = Math.floor(value % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/**
 * 生成带语义色调的状态徽标。
 */
export function badge(text, tone) {
  return el("span", `badge ${tone || "badge-wait"}`, text);
}

/** 两列键值表；值可以是字符串、数字或已有 DOM 节点。 */
export function kv(rows, className) {
  const box = el("div", className || "stage-kv");
  rows.forEach(([key, value]) => {
    const row = el("div", "stage-kv-row");
    row.appendChild(el("span", "stage-kv-key", key));
    if (value instanceof Node) {
      row.appendChild(value);
    } else {
      row.appendChild(el("code", "stage-kv-value", value === undefined || value === null ? "—" : String(value)));
    }
    box.appendChild(row);
  });
  return box;
}

/** 带表头的简单表格；cells 可以是字符串数组或 DOM 数组。 */
export function table(headers, rows) {
  const wrap = el("table", "stage-table");
  const head = el("tr");
  headers.forEach((header) => head.appendChild(el("th", null, header)));
  wrap.appendChild(head);
  rows.forEach((cells) => {
    const tr = el("tr");
    cells.forEach((cell) => {
      const td = el("td");
      if (cell instanceof Node) td.appendChild(cell);
      else td.textContent = cell === undefined || cell === null ? "" : String(cell);
      tr.appendChild(td);
    });
    wrap.appendChild(tr);
  });
  return wrap;
}

/**
 * 创建标题区块容器。
 */
export function section(title) {
  return el("h3", "subhead stage-subhead", title);
}

/* ------------------------------------------------------------------ 三态渲染 */

/**
 * 渲染加载中占位块。
 */
export function loadingBlock(text) {
  const box = el("div", "state-block");
  box.setAttribute("role", "status");
  box.appendChild(el("span", "spinner"));
  box.appendChild(el("span", null, text || "加载中…"));
  return box;
}

/**
 * 渲染空状态占位块。
 */
export function emptyBlock(title, detail) {
  const box = el("div", "state-block");
  box.appendChild(el("span", "state-title", title));
  if (detail) box.appendChild(el("span", null, detail));
  return box;
}

/**
 * 渲染错误信息块，可选重试按钮。
 */
export function errorBlock(message, retryLabel, onRetry) {
  const box = el("div", "state-block error-state");
  box.setAttribute("role", "alert");
  box.appendChild(el("span", "state-title", "加载失败"));
  box.appendChild(el("span", null, message));
  if (retryLabel && typeof onRetry === "function") {
    const button = el("button", "btn btn-small", retryLabel);
    button.type = "button";
    button.addEventListener("click", onRetry);
    box.appendChild(button);
  }
  return box;
}

/* ------------------------------------------------------------------ 差量渲染 */

/** 数据签名相同就完全不动 DOM（避免轮询重建夺走焦点/滚动位置）。
 * 返回 true 表示执行了重建。 */
export function renderSigned(node, signature, build) {
  if (!node) return false;
  if (node.dataset.sig === signature) return false;
  node.dataset.sig = signature;
  node.replaceChildren();
  const content = build();
  if (content instanceof Node) node.appendChild(content);
  else if (Array.isArray(content)) content.forEach((item) => item && node.appendChild(item));
  return true;
}

/** 差量更新 <select>：签名不变完全不动 DOM；聚焦/展开时暂缓，失焦后补上。 */
export function syncSelectOptions(select, items, signature, preferredValue) {
  if (!select) return;
  if (select.dataset.sig === signature) return;
  if (document.activeElement === select) {
    select.dataset.pendingSig = signature;
    select._pendingItems = items;
    select._pendingPreferred = preferredValue;
    return;
  }
  applySelectOptions(select, items, signature, preferredValue);
}

/**
 * 真正重建 <select> 选项并尽量恢复选中值。
 */
export function applySelectOptions(select, items, signature, preferredValue) {
  const previous = select.value;
  select.replaceChildren();
  items.forEach((item) => {
    const option = el("option", null, item.label);
    option.value = item.value;
    if (item.disabled) option.disabled = true;
    select.appendChild(option);
  });
  const target = preferredValue !== undefined && preferredValue !== null ? preferredValue : previous;
  const values = items.map((item) => item.value);
  if (target !== "" && values.indexOf(target) >= 0) select.value = target;
  else if (values.indexOf(previous) >= 0) select.value = previous;
  else if (items.length) select.selectedIndex = 0;
  select.dataset.sig = signature;
  delete select.dataset.pendingSig;
  select._pendingItems = null;
  select._pendingPreferred = undefined;
}

/**
 * 提交延迟到下一次交互才应用的选中项。
 */
export function flushPendingSelect(select) {
  if (select && select.dataset.pendingSig && select._pendingItems) {
    applySelectOptions(select, select._pendingItems, select.dataset.pendingSig, select._pendingPreferred);
  }
}

/** 给一组 select 绑定失焦补齐（幂等）。 */
export function bindSelectFlush(ids) {
  ids.forEach((id) => {
    const select = byId(id);
    if (select && !select.dataset.flushWired) {
      select.dataset.flushWired = "1";
      select.addEventListener("blur", () => flushPendingSelect(select));
    }
  });
}

/* ------------------------------------------------------------------ toast */

const TOAST_ICONS = { error: "✕", ok: "✓", warn: "!", info: "i" };
const TOAST_MAX = 5;

/**
 * 显示带类型与自动消失的提示条。
 */
export function showToast(type, title, message, options) {
  const region = byId("toast-region");
  if (!region) return null;
  const opts = options || {};
  const kind = TOAST_ICONS[type] ? type : "info";
  const toast = el("div", `toast toast-${kind}`);
  // live 区整体播报，单条不再加 role，避免双重朗读。
  toast.appendChild(el("span", "toast-icon", TOAST_ICONS[kind]));
  const body = el("div", "toast-body");
  body.appendChild(el("span", "toast-title", title || ""));
  if (message) body.appendChild(el("span", "toast-message", message));
  if (opts.actionLabel && typeof opts.onAction === "function") {
    const action = el("button", "btn btn-small toast-action", opts.actionLabel);
    action.type = "button";
    action.addEventListener("click", () => {
      dismissToast(toast);
      opts.onAction();
    });
    body.appendChild(action);
  }
  toast.appendChild(body);
  const close = el("button", "toast-close", "✕");
  close.type = "button";
  close.title = "关闭提示";
  close.setAttribute("aria-label", "关闭提示");
  close.addEventListener("click", () => dismissToast(toast));
  toast.appendChild(close);
  region.appendChild(toast);
  // 挤掉最旧的可自动消失提示；常驻（duration:0）失败提示不被挤掉。
  const live = Array.from(region.children);
  if (live.length > TOAST_MAX) {
    const removable = live.find((node) => node !== toast && !node.dataset.sticky);
    if (removable) dismissToast(removable, true);
  }
  const duration = opts.duration !== undefined ? opts.duration : kind === "error" ? 8000 : 3500;
  if (duration > 0) {
    toast._timer = window.setTimeout(() => dismissToast(toast), duration);
  } else {
    toast.dataset.sticky = "1";
  }
  return toast;
}

/**
 * 关闭指定提示条（可立即移除）。
 */
export function dismissToast(toast, immediate) {
  if (!toast || toast._leaving) return;
  toast._leaving = true;
  if (toast._timer) window.clearTimeout(toast._timer);
  if (immediate) {
    toast.remove();
    return;
  }
  toast.classList.add("leaving");
  window.setTimeout(() => toast.remove(), 200);
}

/* ------------------------------------------------------------------ 应用内对话框 */
/* 用页面内 <dialog> 替代原生 confirm/prompt；排队处理，连开不互相吞掉。 */

let dialogState = null;
const dialogQueue = [];

/**
 * 对话框确认按钮的统一提交入口。
 */
function dialogSubmit() {
  if (!dialogState) return;
  const input = byId("app-dialog-input");
  const errorNode = byId("app-dialog-error");
  if (dialogState.mode === "prompt") {
    const value = input ? input.value : "";
    if (dialogState.validate) {
      const problem = dialogState.validate(value);
      if (problem) {
        if (errorNode) errorNode.textContent = problem;
        if (input) input.focus();
        return;
      }
    }
    dialogFinish(value);
  } else {
    dialogFinish(true);
  }
}

/**
 * 结束当前对话框并把结果回传给调用方。
 */
function dialogFinish(result) {
  const pending = dialogState;
  dialogState = null;
  const dialog = byId("app-dialog");
  if (dialog && dialog.open) dialog.close();
  if (pending && pending.resolve) pending.resolve(result);
  // 关闭动画后处理队列中的下一个。
  window.setTimeout(pumpDialogQueue, 0);
}

/**
 * 串行处理对话框队列，避免叠层。
 */
function pumpDialogQueue() {
  if (dialogState || !dialogQueue.length) return;
  const next = dialogQueue.shift();
  openAppDialogNow(next.mode, next.options, next.resolve);
}

/**
 * 真正打开并聚焦一个对话框。
 */
function openAppDialogNow(mode, options, resolve) {
  const dialog = byId("app-dialog");
  if (!dialog) {
    resolve(null);
    return;
  }
  if (!dialog.dataset.wired) {
    dialog.dataset.wired = "1";
    const cancel = byId("app-dialog-cancel");
    const ok = byId("app-dialog-ok");
    const inputNode = byId("app-dialog-input");
    if (cancel) cancel.addEventListener("click", () => dialogFinish(null));
    if (ok) ok.addEventListener("click", dialogSubmit);
    if (inputNode) {
      inputNode.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          dialogSubmit();
        }
      });
    }
    dialog.addEventListener("close", () => {
      if (dialogState) dialogFinish(null);
    });
  }
  dialogState = { mode, validate: options.validate || null, resolve };
  const title = byId("app-dialog-title");
  const body = byId("app-dialog-body");
  const input = byId("app-dialog-input");
  const errorNode = byId("app-dialog-error");
  const okButton = byId("app-dialog-ok");
  const cancelButton = byId("app-dialog-cancel");
  const titleText = options.title || (mode === "prompt" ? "请输入" : "请确认");
  if (title) title.textContent = titleText;
  dialog.setAttribute("aria-label", titleText);
  if (body) {
    body.textContent = options.message || "";
    body.className = `app-dialog-body${options.danger ? " danger" : ""}`;
  }
  if (errorNode) errorNode.textContent = "";
  if (input) {
    input.hidden = mode !== "prompt";
    input.value = options.defaultValue || "";
    input.placeholder = options.placeholder || "";
    input.setAttribute("aria-label", titleText);
  }
  if (okButton) {
    okButton.textContent = options.okText || (mode === "prompt" ? "保存" : "确定");
    okButton.className = `btn ${options.danger ? "btn-danger" : "btn-primary"}`;
  }
  if (cancelButton) cancelButton.textContent = options.cancelText || "取消";
  dialog.showModal();
  window.setTimeout(() => {
    if (mode === "prompt" && input) {
      input.focus();
      input.select();
    } else if (options.danger && cancelButton) {
      // 危险操作默认焦点放在「取消」，回车不会误删。
      cancelButton.focus();
    } else if (okButton) {
      okButton.focus();
    }
  }, 0);
}

/**
 * 把对话框请求入队并返回 Promise 结果。
 */
export function openAppDialog(mode, options = {}) {
  return new Promise((resolve) => {
    if (dialogState) {
      dialogQueue.push({ mode, options, resolve });
      return;
    }
    openAppDialogNow(mode, options, resolve);
  });
}

export const confirmDialog = (options) => openAppDialog("confirm", options);
export const promptDialog = (options) => openAppDialog("prompt", options);

/** 复制文本：优先 Clipboard API，降级 execCommand。 */
export async function copyText(text) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch (error) {
    /* 降级 */
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.style.position = "fixed";
  area.style.opacity = "0";
  document.body.appendChild(area);
  area.select();
  let ok = false;
  try {
    ok = document.execCommand("copy");
  } catch (error) {
    ok = false;
  }
  area.remove();
  return ok;
}

/* ------------------------------------------------------------------ 列表键盘操作 */

/**
 * 让表格行可点击/键盘激活。
 */
export function makeRowActivatable(row) {
  row.setAttribute("role", "button");
  row.tabIndex = 0;
}

/**
 * 为列表绑定上下键与 Enter 的键盘导航。
 */
export function bindListKeyboard(list) {
  if (!list || list.dataset.kbdWired) return;
  list.dataset.kbdWired = "1";
  list.addEventListener("keydown", (event) => {
    if (event.key !== "Enter" && event.key !== " ") return;
    const target = event.target;
    if (!target || !target.closest) return;
    const row = target.closest(".stage-row");
    if (!row) return;
    event.preventDefault();
    row.click();
  });
}

/* ------------------------------------------------------------------ 图表配色 */

let chartPaletteCache = null;

/** 图表 8 色从 CSS 变量读取，亮暗主题切换后自动跟随。 */
export function chartPalette() {
  if (chartPaletteCache) return chartPaletteCache;
  const style = getComputedStyle(document.documentElement);
  const colors = [];
  for (let i = 1; i <= 8; i += 1) {
    const value = style.getPropertyValue(`--chart-${i}`).trim();
    colors.push(value || ["#7c93f0", "#d3a44e", "#9d8cc4", "#5fb88f", "#cf9463", "#c98aa6", "#6fb3b8", "#d97a72"][i - 1]);
  }
  chartPaletteCache = colors;
  return colors;
}

/**
 * 主题变化时使配色缓存失效。
 */
export function invalidateChartPalette() {
  chartPaletteCache = null;
}

/* ------------------------------------------------------------------ 字体令牌 */

let fontTokenCache = null;

/** 从 CSS 令牌读取三类字体族与字号/字重，供 Chart.js 与 canvas 共用。
 *  纯本地字体，不发起外部请求；读取失败时回退到系统等价栈。 */
export function fontTokens() {
  if (fontTokenCache) return fontTokenCache;
  const style =
    typeof getComputedStyle === "function" && typeof document !== "undefined" && document.documentElement
      ? getComputedStyle(document.documentElement)
      : null;
  const read = (name, fallback) => (style && style.getPropertyValue(name).trim()) || fallback;
  fontTokenCache = {
    ui: read("--font-ui", 'system-ui, "Segoe UI", "Microsoft YaHei", "PingFang SC", sans-serif'),
    display: read("--font-display", 'system-ui, "Segoe UI", "Microsoft YaHei", sans-serif'),
    data: read("--font-data", '"Cascadia Mono", "Cascadia Code", Consolas, "SF Mono", Menlo, "Courier New", monospace'),
    size: {
      micro: parseInt(read("--fs-2xs", "10"), 10) || 10,
      xs: parseInt(read("--fs-xs", "11"), 10) || 11,
      sm: parseInt(read("--fs-sm", "12"), 10) || 12,
      base: parseInt(read("--fs-base", "13"), 10) || 13,
      md: parseInt(read("--fs-md", "14"), 10) || 14,
      lg: parseInt(read("--fs-lg", "16"), 10) || 16,
    },
    weight: {
      regular: parseInt(read("--fw-regular", "400"), 10) || 400,
      medium: parseInt(read("--fw-medium", "500"), 10) || 500,
      semibold: parseInt(read("--fw-semibold", "600"), 10) || 600,
      bold: parseInt(read("--fw-bold", "700"), 10) || 700,
    },
  };
  return fontTokenCache;
}

/**
 * 字体变化时使字体缓存失效。
 */
export function invalidateFontTokens() {
  fontTokenCache = null;
}

/** 构造 canvas ctx.font 字符串：fontToken("data", 12, 600) -> "600 12px <族>"。 */
export function canvasFont(kind, size, weight) {
  const t = fontTokens();
  const family = kind === "ui" ? t.ui : kind === "display" ? t.display : t.data;
  const w = weight ? `${weight} ` : "";
  return `${w}${size}px ${family}`;
}

/** Chart.js 字体配置对象：chartFont("data", 11) -> { family, size, weight }。 */
export function chartFont(kind, size, weight) {
  const t = fontTokens();
  const family = kind === "ui" ? t.ui : kind === "display" ? t.display : t.data;
  return { family, size, weight: weight || t.weight.regular };
}