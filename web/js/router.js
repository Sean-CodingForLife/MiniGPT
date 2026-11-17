/* Hash 路由器：按流水线阶段分页签，单页多视图切换。
 *
 * 在 SPA 中的位置：由 app.js 最后调用 initRouter() 启动；它负责
 *   1) 解析 #/xxx（可带 ?run= 之类的深链查询）；
 *   2) 显示对应的 section[data-view]，同步页签高亮、document.title 与 body[data-active-view]；
 *   3) 通过 store.setRoute() 和 window 上的 "spa:view" 自定义事件通知各视图。
 *
 * 路由表： #/projects #/data #/pretrain #/sft #/eval #/generate #/compare #/export
 * 预训练与 SFT 共用同一个训练工作区（view=train），由 stage 决定过滤条件。
 *
 * 关键约束：
 *  - 只有路由真正变化时才派发 spa:view，同页签重复 hashchange 不会触发重渲染；
 *  - 非法 hash 用 replaceState 规范化回 #/pretrain，保证地址栏与实际视图一致；
 *  - hash 内的查询串会合并进地址栏 search，供 dashboard / generation 读取 ?run=；
 *  - 所有 history 操作都包 try/catch：file:// 或受限浏览器下退化为纯视图切换。
 */

import { setRoute } from "./store.js";

/**
 * 路由定义表：route 名 → 视图容器 + 页面标题，训练类路由额外带 stage。
 * view 对应 index.html 里 section[data-view] 的取值。
 */
const ROUTES = {
  projects: { view: "projects", title: "项目" },
  data: { view: "data", title: "数据" },
  pretrain: { view: "train", title: "预训练", stage: "pretrain" },
  sft: { view: "train", title: "SFT", stage: "sft" },
  eval: { view: "eval", title: "评测" },
  generate: { view: "generate", title: "生成可视化" },
  compare: { view: "compare", title: "对比" },
  export: { view: "export", title: "导出" },
};

/** 上一次成功应用的路由名；用于判断是否需要派发 spa:view。 */
let lastRoute = null;

/**
 * 解析当前 hash 得到合法路由名。
 * 非法或空 hash 会被规范化（replaceState）到 #/pretrain，并保留原查询串。
 * @returns {string} ROUTES 中存在的路由名。
 */
function currentRoute() {
  const body = (window.location.hash || "").replace(/^#\/?/, "").trim();
  const raw = body.split("?")[0].split("/")[0] || "";
  if (Object.prototype.hasOwnProperty.call(ROUTES, raw)) return raw;
  // 非法 hash 规范化回默认路由，地址栏与实际视图保持一致；保留深链查询。
  const queryIndex = body.indexOf("?");
  const query = queryIndex >= 0 ? body.slice(queryIndex) : "";
  try {
    window.history.replaceState(null, "", `${window.location.pathname}${window.location.search}#/pretrain${query}`);
  } catch (error) {
    /* 受限环境：仅视图回退 */
  }
  return "pretrain";
}

/**
 * 应用当前路由：切视图、同步页签与文档标题、广播状态。
 * @returns {void}
 */
function applyRoute() {
  const route = currentRoute();
  const definition = ROUTES[route];
  // hash 内深链查询（#/sft?run=x）同步到地址栏查询，供各视图消费。
  // 注意：这里只改 search，不改 hash，因此不会再次触发 hashchange 造成递归。
  const hashBody = (window.location.hash || "").replace(/^#\/?/, "");
  const queryIndex = hashBody.indexOf("?");
  if (queryIndex >= 0) {
    try {
      const url = new URL(window.location.href);
      const params = new URLSearchParams(hashBody.slice(queryIndex + 1));
      params.forEach((value, key) => url.searchParams.set(key, value));
      window.history.replaceState(null, "", url.pathname + url.search + url.hash);
    } catch (error) {
      /* 受限环境：深链只是便利 */
    }
  }
  // stage 只有训练类路由才有值；其它路由为 null，store 会回退到 pretrain。
  const stage = definition.stage || null;

  // 视图切换靠 hidden 属性，CSS 不参与显示/隐藏判断。
  document.querySelectorAll("section[data-view]").forEach((node) => {
    node.hidden = node.getAttribute("data-view") !== definition.view;
  });
  document.querySelectorAll("[data-route]").forEach((link) => {
    const active = link.getAttribute("data-route") === route;
    link.classList.toggle("active", active);
    if (active) {
      link.setAttribute("aria-current", "page");
      // 移动端横向页签：让当前项滚入可视区。
      if (link.scrollIntoView) link.scrollIntoView({ block: "nearest", inline: "nearest" });
    } else {
      link.removeAttribute("aria-current");
    }
  });
  document.title = `MiniGPT · ${definition.title}`;
  document.body.setAttribute("data-active-view", definition.view);
  if (stage) document.body.setAttribute("data-stage", stage);
  else document.body.removeAttribute("data-stage");
  const detail = { route, view: definition.view, stage };
  // 先写 store（同步广播 "route"），再派发一次性的 "spa:view" 供视图做较重渲染。
  setRoute(detail);
  if (route !== lastRoute) {
    lastRoute = route;
    window.dispatchEvent(new CustomEvent("spa:view", { detail }));
  }
  // 滚动发生在各视图内部容器，这里同时归位窗口与当前视图。
  window.scrollTo(0, 0);
  const view = document.querySelector(`section[data-view="${definition.view}"]:not([hidden])`);
  if (view) view.scrollTop = 0;
}

/**
 * 启动路由：监听 hashchange 并立即应用一次当前 hash。
 * @returns {void} 重复调用会重复绑定监听（调用方保证只调用一次）。
 */
export function initRouter() {
  window.addEventListener("hashchange", applyRoute);
  applyRoute();
}
