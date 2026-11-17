/* 全局 UI 状态：暂停刷新 / 主题 / 当前视图与阶段。
 *
 * 在 SPA 中的位置：最底层的状态模块。dashboard、各阶段视图、train/charts.js、
 * attention.js 都从这里读取状态或订阅变化，本模块不反向依赖任何视图（避免 import 环）。
 * 依赖：仅浏览器原生 API（document / localStorage）。
 *
 * 关键约束：
 *  - 轻量发布-订阅：模块只订阅自己关心的切片（"paused" / "theme" / "route"）；
 *  - 状态持久化到 localStorage（键前缀 minigpt:），失败（隐私模式、配额）静默降级，
 *    此时偏好只在本次会话生效；
 *  - emit 为同步派发，订阅者在 set* 的调用栈内执行。
 */

/** 订阅者表：切片名 → 处理函数集合。 */
const listeners = new Map();

/**
 * 读取持久化偏好。
 * @param {string} key 不含前缀的键名（函数内部自动拼上 minigpt:）。
 * @param {string} fallback localStorage 不可用或键不存在时返回的默认值。
 * @returns {string} 原始字符串值（取值范围由调用方校验）。
 */
function readPref(key, fallback) {
  try {
    const value = window.localStorage.getItem(`minigpt:${key}`);
    return value === null ? fallback : value;
  } catch (error) {
    return fallback;
  }
}

/**
 * 写入持久化偏好。
 * @param {string} key 不含前缀的键名。
 * @param {string|number|boolean} value 会被 String() 后存储。
 * @returns {void} 写入失败时不抛出。
 */
function writePref(key, value) {
  try {
    window.localStorage.setItem(`minigpt:${key}`, String(value));
  } catch (error) {
    /* 隐私模式等场景：仅本次会话生效 */
  }
}

/**
 * 全局可变状态快照。
 * - paused：暂停自动轮询（顶栏开关，dashboard/compare/eval 都会检查）。
 * - theme： "dark" | "light"，初值来自 localStorage，只允许这两个值。
 * - view：当前视图容器名（train/data/eval/...），由 router 写入。
 * - stage：训练阶段 "pretrain" | "sft"，仅 view=train 时有意义。
 * - route：当前 hash 路由名，比 view 更细（区分 pretrain 与 sft）。
 */
export const store = {
  paused: false,
  theme: readPref("theme", "dark") === "light" ? "light" : "dark",
  view: "train",
  stage: "pretrain",
  route: "pretrain",
};

/**
 * 订阅某个状态切片。
 * @param {string} key 切片名："paused" | "theme" | "route"。
 * @param {(value: any) => void} handler 状态变化回调（同一切片可挂多个）。
 * @returns {() => void} 取消订阅函数，视图销毁时调用以免泄漏。
 */
export function on(key, handler) {
  if (!listeners.has(key)) listeners.set(key, new Set());
  listeners.get(key).add(handler);
  return () => listeners.get(key).delete(handler);
}

/**
 * 同步派发某个切片的当前值。
 * @param {string} key 切片名。
 * @param {any} value 新值（原样传给订阅者）。
 * @returns {void}
 */
function emit(key, value) {
  const set = listeners.get(key);
  if (set) set.forEach((handler) => handler(value));
}

/**
 * 设置「暂停刷新」。
 * @param {boolean} value true=暂停轮询；值未变化时不派发，避免重复触发请求。
 * @returns {void}
 */
export function setPaused(value) {
  const next = Boolean(value);
  if (store.paused === next) return;
  store.paused = next;
  emit("paused", next);
}

/**
 * 切换亮/暗主题，并持久化到 localStorage。
 * @param {string} value 只接受 "light"；其它值一律按 "dark" 处理。
 * @returns {void}
 */
export function setTheme(value) {
  const next = value === "light" ? "light" : "dark";
  if (store.theme === next) return;
  store.theme = next;
  writePref("theme", next);
  // 主题实际生效靠 <html data-theme>，CSS 变量按该属性切换。
  document.documentElement.setAttribute("data-theme", next);
  emit("theme", next);
}

/**
 * 记录当前路由切片并广播，供各视图复用自己的状态。
 * @param {{route?: string, view?: string, stage?: string|null}} detail
 *   router.js 解析出的路由信息；缺省字段回退到 pretrain 工作区。
 * @returns {void}
 */
export function setRoute(detail) {
  store.view = detail.view || "train";
  store.stage = detail.stage || "pretrain";
  store.route = detail.route || "pretrain";
  emit("route", detail);
}

/**
 * 初始化主题与暂停开关（顶栏控件在 index.html，这里只做同步与绑定）。
 * @returns {void} 控件缺失时静默跳过（便于在测试页局部复用）。
 */
export function initStore() {
  document.documentElement.setAttribute("data-theme", store.theme);
  const themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    // sync：把当前主题映射到按钮文案/提示/aria-pressed，主题变化时复用。
    const sync = () => {
      themeBtn.textContent = store.theme === "dark" ? "☀ 亮色" : "☾ 暗色";
      themeBtn.title = store.theme === "dark" ? "切换到亮色主题" : "切换到暗色主题";
      themeBtn.setAttribute("aria-pressed", store.theme === "light" ? "true" : "false");
    };
    sync();
    themeBtn.addEventListener("click", () => setTheme(store.theme === "dark" ? "light" : "dark"));
    on("theme", sync);
  }
  const pauseToggle = document.getElementById("pause-toggle");
  if (pauseToggle) {
    pauseToggle.addEventListener("change", () => setPaused(pauseToggle.checked));
    // 反向同步：其它模块（如「立即刷新」按钮）恢复刷新时，复选框跟着回到未勾选。
    on("paused", (value) => {
      pauseToggle.checked = value;
    });
  }
}
