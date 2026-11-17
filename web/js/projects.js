/* 项目视图：项目 = 数据 → 预训练 → SFT → 评测 → 导出 的全流程容器。
 *
 * 在 SPA 中的位置：路由 #/projects 对应的 section#view-projects；app.js 调用 initProjectsView()。
 * 数据来源：GET/POST /api/v1/projects（列表 / 创建）与 GET /api/v1/projects/<id>（聚合详情），
 *           POST /api/v1/projects/<id>/runs（把已有运行挂入项目）。
 * 依赖：api.js（端点）、ui.js（DOM、toast、请求出口）。
 *
 * 关键约定：
 *  - 成员关系的权威数据在 projects/<id>/project.json；本模块只做展示与挂载操作，不推导成员；
 *  - 链节点点击复用既有 ?run= 深链机制跳转到对应阶段页（会触发整页加载）；
 *  - 详情请求与运行列表并行拉取，带序号守卫，快速切换项目时旧响应丢弃；
 *  - 所有项目/运行文本走 textContent 渲染。
 */

import {
  byId, el, badge, kv, section, loadingBlock, emptyBlock, errorBlock,
  makeRowActivatable, bindListKeyboard, getJson, postJson, showToast,
} from "./ui.js";
import { endpoints } from "./api.js";

/** 视图局部状态：list/runs 为最近一次全量数据，detailSeq 用于详情竞态守卫。 */
const state = { loaded: false, loading: false, selected: null, list: [], runs: [], detailSeq: 0 };

/** 运行状态 → 徽章色调（未知状态回落到 badge-wait，由调用方兜底）。 */
const STATUS_TONE = {
  completed: "badge-ok",
  running: "badge-wait",
  starting: "badge-wait",
  created: "badge-wait",
  stopped: "badge-warn",
  failed: "badge-bad",
};

/**
 * 更新列表区标题右侧的状态行。
 * @param {string} message 展示文本，空串表示清空。
 * @param {boolean} [isError=false] true 时加 error 样式。
 * @returns {void}
 */
function setStatus(message, isError = false) {
  const node = byId("projects-count");
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("error", Boolean(isError));
}

/* ------------------------------------------------------------------ 列表 */

/**
 * 构造列表中的一行项目卡片。
 * @param {object} item /api/v1/projects 的 projects[] 元素：
 *   { project_id, name, run_ids[], created_at, note }。
 * @returns {HTMLElement} 可点击/可键盘激活的 .stage-row。
 */
function rowFor(item) {
  const row = el("div", "stage-row");
  makeRowActivatable(row);
  row.dataset.projectId = item.project_id;
  const active = item.project_id === state.selected;
  if (active) row.classList.add("active");
  row.setAttribute("aria-pressed", active ? "true" : "false");
  const head = el("div", "stage-row-head");
  head.appendChild(el("span", "stage-row-title", item.name || item.project_id));
  head.appendChild(badge(`${(item.run_ids || []).length} 个运行`, "badge-ok"));
  row.appendChild(head);
  row.appendChild(el("div", "stage-row-meta", `${item.project_id}${item.created_at ? ` · ${item.created_at}` : ""}`));
  if (item.note) row.appendChild(el("div", "stage-row-sub", item.note));
  return row;
}

/**
 * 只同步列表行的选中态（列表签名未变时的轻量路径）。
 * @returns {void}
 */
function markActive() {
  const list = byId("projects-list");
  if (!list) return;
  list.querySelectorAll(".stage-row").forEach((row) => {
    const active = row.dataset.projectId === state.selected;
    row.classList.toggle("active", active);
    row.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

/** 列表签名：项目数/成员数未变时跳过重建。 */
let listSig = "";

/**
 * 渲染左侧项目列表。
 * @param {object} payload /api/v1/projects 响应体：{ count, projects[] }。
 * @returns {void}
 */
function renderList(payload) {
  const root = byId("projects-list");
  if (!root) return;
  state.list = payload.projects || [];
  setStatus(`共 ${payload.count} 个项目`);
  // 只把「项目 + 成员数量」纳入签名：note/时间变化不影响列表结构。
  const sig = state.list.map((item) => `${item.project_id}:${(item.run_ids || []).length}`).join("|");
  if (sig === listSig) {
    markActive();
    return;
  }
  listSig = sig;
  root.replaceChildren();
  if (!state.list.length) {
    root.appendChild(emptyBlock("还没有项目", "点「新建项目」把数据→预训练→SFT→评测→导出串成一条流水线。"));
    return;
  }
  state.list.forEach((item) => root.appendChild(rowFor(item)));
}

/* ------------------------------------------------------------------ 链视图 */

/**
 * 构造生命周期链上的一个节点（有 onClick 才可点，否则禁用）。
 * @param {string} label 节点文字。
 * @param {string} status 节点状态样式："done" | "current" | "failed" | ""。
 * @param {string} [detail] 悬停提示（通常是 run_id / 说明）。
 * @param {Function} [onClick] 点击回调；缺省时节点 disabled。
 * @returns {HTMLElement} 复用训练工作区 lifecycle 样式的 button/元素。
 */
function chainNode(label, status, detail, onClick) {
  const node = el("button", `lifecycle-node ${status}`);
  node.type = "button";
  node.appendChild(el("span", "lifecycle-dot"));
  node.appendChild(el("span", null, label));
  if (detail) node.title = detail;
  if (onClick) node.addEventListener("click", onClick);
  else node.disabled = true;
  return node;
}

/**
 * 构造「预训练 / SFT」链节点：点击跳到对应阶段并用 ?run= 深链定位该运行。
 * @param {object|null} run 成员运行摘要；run.missing 表示 runs/ 下目录已不存在。
 * @param {string} label 节点文字前缀（"预训练" / "SFT"）。
 * @param {string} route 目标路由名（"pretrain" / "sft"）。
 * @returns {HTMLElement}
 */
function runNode(run, label, route) {
  if (!run || run.missing) {
    return chainNode(`${label} · 缺失`, "failed", "成员运行目录已不存在");
  }
  const status = run.display_status || run.status || "unknown";
  // 只有 done / failed 有独立样式，其余（运行中、已停止、未开始）统一按 current 呈现。
  const tone = status === "completed" ? "done" : status === "failed" ? "failed" : "current";
  const detail = `${run.run_id} · ${status}`;
  return chainNode(`${label} · ${run.run_id}`, tone, detail, () => {
    // 整页跳转：同时写入 ?run= 查询与目标 hash，dashboard/generation 首屏即可定位。
    window.location.href = `${window.location.pathname}?run=${encodeURIComponent(run.run_id)}#/${route}`;
  });
}

/**
 * 渲染横向流水线链：数据 → 预训练 → SFT → 评测 → 导出。
 * 缺失的阶段仍然显示为可点击节点，点击即跳到该阶段页去创建。
 * @param {object} detail GET /api/v1/projects/<id> 响应体（含 runs / evaluations / exports / data_fingerprints）。
 * @returns {HTMLElement} .project-chain 容器。
 */
function renderChain(detail) {
  const runs = detail.runs || [];
  const pretrains = runs.filter((run) => !run.missing && run.stage === "pretrain");
  const sfts = runs.filter((run) => !run.missing && run.stage === "sft");
  const evaluations = detail.evaluations || [];
  const exports = detail.exports || [];
  const chain = el("div", "project-chain");
  // 数据节点没有独立目录，用「成员运行是否带数据指纹」近似判断是否已准备数据。
  const dataOk = Object.keys(detail.data_fingerprints || {}).length > 0;
  chain.appendChild(chainNode("数据", dataOk ? "done" : "", dataOk ? "成员运行带数据指纹" : "尚无数据指纹", () => {
    window.location.hash = "#/data";
  }));
  chain.appendChild(el("span", "lifecycle-link"));
  chain.appendChild(pretrains.length ? runNode(pretrains[0], "预训练", "pretrain") : chainNode("预训练", "", "尚未创建预训练运行", () => {
    window.location.hash = "#/pretrain";
  }));
  chain.appendChild(el("span", "lifecycle-link"));
  chain.appendChild(sfts.length ? runNode(sfts[0], "SFT", "sft") : chainNode("SFT", "", "尚未创建 SFT 运行", () => {
    window.location.hash = "#/sft";
  }));
  chain.appendChild(el("span", "lifecycle-link"));
  const evalRun = evaluations.length ? evaluations[0].run_id : null;
  chain.appendChild(chainNode(`评测 · ${evaluations.length}`, evaluations.length ? "done" : "", evaluations.length ? `最近评测来自 ${evalRun}` : "尚无评测", () => {
    window.location.hash = "#/eval";
  }));
  chain.appendChild(el("span", "lifecycle-link"));
  chain.appendChild(chainNode(`导出 · ${exports.length}`, exports.length ? "done" : "", exports.length ? exports.map((entry) => entry.name).join(", ") : "尚无导出包", () => {
    window.location.hash = "#/export";
  }));
  return chain;
}

/**
 * 渲染项目详情：流水线链 + 成员运行表 + 挂载控件 + 导出包 + 数据指纹。
 * @param {object} detail GET /api/v1/projects/<id> 响应体。
 * @returns {void}
 */
function renderDetail(detail) {
  const root = byId("projects-detail");
  if (!root) return;
  root.replaceChildren();
  const project = detail.project || {};
  const title = byId("projects-detail-title");
  if (title) title.textContent = project.name || project.project_id || "";
  root.appendChild(renderChain(detail));

  root.appendChild(section("成员运行"));
  const runs = detail.runs || [];
  if (!runs.length) {
    root.appendChild(emptyBlock("还没有成员", "在训练页创建任务时选择该项目，或在下方把已有运行挂进来。"));
  } else {
    const rows = runs.map((run) => {
      // missing：project.json 里登记了但 runs/ 下目录已不存在（被删除或未同步）。
      if (run.missing) return [run.run_id, "缺失", "—", "—"];
      return [
        run.run_id,
        run.stage || "?",
        run.display_status || run.status || "?",
        run.parent_run_id || "—",
      ];
    });
    const wrap = el("div", "project-members");
    wrap.appendChild(renderTable(["运行", "阶段", "状态", "上游"], rows));
    root.appendChild(wrap);
  }

  root.appendChild(section("挂载已有运行"));
  const attachRow = el("div", "project-attach");
  const select = el("select");
  select.id = "project-attach-select";
  // 候选 = 全部运行 - 已在项目内 - 内部运行（下划线前缀由后端过滤）。
  const memberIds = new Set(runs.map((run) => run.run_id));
  const candidates = state.runs.filter((run) => !memberIds.has(run.run_id));
  select.appendChild(el("option", null, candidates.length ? "（选择运行）" : "（没有可挂载的运行）"));
  candidates.forEach((run) => {
    const option = el("option", null, `${run.run_id} · ${run.stage || "?"}`);
    option.value = run.run_id;
    select.appendChild(option);
  });
  const attachBtn = el("button", "btn btn-small", "挂入项目");
  attachBtn.type = "button";
  attachBtn.disabled = !candidates.length;
  attachBtn.addEventListener("click", () => {
    const runId = select.value;
    if (!runId) return;
    // 请求期间禁用按钮，避免重复挂载同一运行。
    attachBtn.disabled = true;
    postJson(endpoints.projectRuns(project.project_id), { run_id: runId })
      .then(() => {
        showToast("ok", "已挂载", `运行 ${runId} 已挂入项目`);
        return openProject(project.project_id);
      })
      .catch((error) => showToast("error", "挂载失败", error.message))
      .finally(() => { attachBtn.disabled = false; });
  });
  attachRow.appendChild(select);
  attachRow.appendChild(attachBtn);
  root.appendChild(attachRow);

  if ((detail.exports || []).length) {
    root.appendChild(section("导出包"));
    detail.exports.forEach((entry) => {
      const row = el("div", "stage-raw-row");
      row.appendChild(el("span", null, entry.name));
      row.appendChild(badge(entry.run_id || "来源未知", "badge-ok"));
      root.appendChild(row);
    });
  }
  const fingerprints = detail.data_fingerprints || {};
  if (Object.keys(fingerprints).length) {
    // 指纹值普遍较长，只展示前 12 个字符便于人眼核对是否同一份数据。
    root.appendChild(section("数据指纹"));
    root.appendChild(kv(Object.entries(fingerprints).map(([runId, fp]) => [runId, Object.entries(fp || {}).map(([key, value]) => `${key}=${String(value).slice(0, 12)}…`).join(" · ")])));
  }
}

/**
 * 简易表格构建器（本模块专用：单元格一律按文本处理）。
 * 注意：ui.js 里已有功能相近的 table()，这里保留是因为历史实现先于 ui.table。
 * @param {string[]} headers 表头。
 * @param {Array<Array<string|number>>} rows 行数据。
 * @returns {HTMLTableElement} .stage-table。
 */
function renderTable(headers, rows) {
  const wrap = el("table", "stage-table");
  const head = el("tr");
  headers.forEach((header) => head.appendChild(el("th", null, header)));
  wrap.appendChild(head);
  rows.forEach((cells) => {
    const tr = el("tr");
    cells.forEach((cell) => tr.appendChild(el("td", null, cell)));
    wrap.appendChild(tr);
  });
  return wrap;
}

/* ------------------------------------------------------------------ 数据 */

/**
 * 打开某个项目详情：并行拉取项目聚合详情与全部运行列表（挂载候选需要）。
 * @param {string} projectId 项目 id（projects/<id>/ 目录名）。
 * @returns {Promise<void>} 失败时右侧显示 errorBlock，不抛出。
 */
async function openProject(projectId) {
  state.selected = projectId;
  markActive();
  const root = byId("projects-detail");
  if (!root) return;
  const seq = ++state.detailSeq;
  root.replaceChildren(loadingBlock("载入项目详情…"));
  try {
    const [detail, runsPayload] = await Promise.all([
      getJson(endpoints.project(projectId)),
      getJson(endpoints.runs()),
    ]);
    // 快速切换项目时，旧响应不得覆盖新选择。
    if (seq !== state.detailSeq || state.selected !== projectId) return;
    state.runs = (runsPayload && runsPayload.runs) || [];
    renderDetail(detail);
  } catch (error) {
    if (seq !== state.detailSeq || state.selected !== projectId) return;
    root.replaceChildren(errorBlock(error.message, "重试", () => openProject(projectId)));
  }
}

/**
 * 拉取并渲染项目列表，必要时自动选中第一个项目。
 * @param {boolean} force true=忽略 loaded 缓存强制重扫。
 * @returns {Promise<void>} 失败时列表区显示 errorBlock，不抛出。
 */
async function load(force) {
  if (state.loading) return;
  if (state.loaded && !force) return;
  state.loading = true;
  setStatus("扫描 projects/ …");
  const root = byId("projects-list");
  if (root && (!state.loaded || force)) root.replaceChildren(loadingBlock("扫描 projects/ 目录…"));
  try {
    const payload = await getJson(endpoints.projects());
    state.loaded = true;
    renderList(payload);
    if (!state.selected && state.list.length) openProject(state.list[0].project_id);
    else if (state.selected) {
      // 刷新后维持选择；项目被删除则回到空态提示。
      const still = state.list.some((item) => item.project_id === state.selected);
      if (still) openProject(state.selected);
      else {
        state.selected = null;
        const detail = byId("projects-detail");
        if (detail) detail.replaceChildren(emptyBlock("选择一个项目", "左侧列表点击查看流水线。"));
      }
    }
  } catch (error) {
    setStatus(`项目加载失败：${error.message}`, true);
    if (root) root.replaceChildren(errorBlock(error.message, "重试", () => load(true)));
  } finally {
    state.loading = false;
  }
}

/**
 * 新建项目：这里用原生 window.prompt 取名字（其它页面已改用应用内对话框）。
 * @returns {void} 名称留空或取消时不发请求。
 */
function promptNewProject() {
  const name = window.prompt("新项目名称（例如：英文莎士比亚流水线）");
  if (!name || !name.trim()) return;
  postJson(endpoints.projects(), { name: name.trim() })
    .then((data) => {
      showToast("ok", "项目已创建", data.message || "");
      // 清空列表签名，强制下一次 load 重建列表（新项目要出现在最前）。
      listSig = "";
      return load(true).then(() => {
        const project = data.project || {};
        if (project.project_id) openProject(project.project_id);
      });
    })
    .catch((error) => showToast("error", "创建失败", error.message));
}

/**
 * 初始化项目视图：列表点击（事件委托）、键盘激活、刷新/新建按钮、spa:view 首次加载。
 * @returns {void} 页面里没有 #projects-list 时直接返回。
 */
export function initProjectsView() {
  const list = byId("projects-list");
  if (!list) return;
  bindListKeyboard(list);
  // 事件委托：行会被整段重建，绑在容器上才不会丢监听。
  list.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !target.closest) return;
    const row = target.closest(".stage-row");
    if (row && row.dataset.projectId) openProject(row.dataset.projectId);
  });
  const refresh = byId("projects-refresh-btn");
  if (refresh) refresh.addEventListener("click", () => load(true));
  const newBtn = byId("projects-new-btn");
  if (newBtn) newBtn.addEventListener("click", promptNewProject);
  window.addEventListener("spa:view", (event) => {
    if (event.detail && event.detail.view === "projects") load(false);
  });
}
