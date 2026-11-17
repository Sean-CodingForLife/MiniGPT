/* 评测阶段视图：扫描 runs/<id>/ 下的 eval_* 报告 + 面板内发起评估（N08）。
 *
 * 在 SPA 中的位置：路由 #/eval 对应的 section#view-eval；app.js 调用 initEvalView()。
 * 数据来源：
 *   GET  /api/v1/evaluations            评测报告列表（每份 evaluation.json 一条）
 *   GET  /api/v1/evaluations/<run_id>   详情，可用 ?eval=eval_test 指定具体目录
 *   POST /api/v1/evaluate               起 evaluate.py 子进程（N08）
 *   GET  /api/v1/evaluate/status        轮询子进程进度
 * 依赖：api.js（端点）、ui.js（DOM 与格式化基座）、store（暂停遮罩，轮询时检查 store.paused）。
 *
 * 关键约定：
 *  - 列表首次进入才拉取（state.loaded 守卫），刷新按钮 force=true 强制重扫；
 *  - 详情请求带自增序号：快速切换报告时旧响应不落地；
 *  - 启动评估后进入 1.5s 轮询；连续 5 次查询失败自动停表，避免无限重试；
 *  - 评估完成后刷新列表并定位到新报告（以列表为准，避免临时结果与列表不一致）；
 *  - 报告里的问题/输出/边界草稿全部走 textContent，不执行模型产物中的文本。
 */

import {
  byId, el, fmtNumber, fmtInt, pct, badge, getJson, postJson, kv, table, section,
  loadingBlock, emptyBlock, errorBlock, makeRowActivatable, bindListKeyboard, syncSelectOptions,
} from "./ui.js";
import { endpoints } from "./api.js";
import { store } from "./store.js";

/** 视图局部状态：selected + selectedDir 共同定位一份报告（同一 run 可能有多份）。 */
const state = { loaded: false, loading: false, selected: null, selectedDir: null, list: [] };
/** 详情请求序号：防止快速切换报告时旧响应覆盖新选择。 */
let detailSeq = 0;

/**
 * 更新列表区标题右侧的状态行。
 * @param {string} message 展示文本，空串表示清空。
 * @param {boolean} [isError=false] true 时加 error 样式。
 * @returns {void}
 */
function setStatus(message, isError = false) {
  const node = byId("eval-count");
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("error", Boolean(isError));
}

/**
 * 把报告的核心指标压成一行 chips 文本，用于列表行摘要。
 * @param {object} metrics evaluation.json 的 metrics（ppl / answer_loss / char_f1 / exact_match…）。
 * @returns {string[]} 形如 ["ppl 12.34", "char_f1 0.412"]，缺失项自动省略。
 */
function metricChips(metrics) {
  const chips = [];
  if (metrics.perplexity !== undefined) chips.push(`ppl ${fmtNumber(metrics.perplexity, 2)}`);
  if (metrics.answer_loss !== undefined) chips.push(`answer_loss ${fmtNumber(metrics.answer_loss, 3)}`);
  if (metrics.char_f1 !== undefined) chips.push(`char_f1 ${fmtNumber(metrics.char_f1, 3)}`);
  if (metrics.exact_match !== undefined) chips.push(`EM ${fmtNumber(metrics.exact_match, 3)}`);
  return chips;
}

/**
 * 构造列表中的一行评测报告卡片。
 * @param {object} entry /api/v1/evaluations 的 evaluations[] 元素：
 *   { run_id, eval_dir, stage, split, metrics, prediction_count, has_boundary }。
 * @returns {HTMLElement} 可点击/可键盘激活的 .stage-row。
 */
function rowFor(entry) {
  const row = el("div", "stage-row");
  makeRowActivatable(row);
  // run_id 与 eval_dir 一起存在 dataset 上：同一运行可能有多份不同切分的报告。
  row.dataset.runId = entry.run_id;
  row.dataset.evalDir = entry.eval_dir;
  const active = entry.run_id === state.selected && entry.eval_dir === state.selectedDir;
  if (active) row.classList.add("active");
  row.setAttribute("aria-pressed", active ? "true" : "false");
  const head = el("div", "stage-row-head");
  head.appendChild(el("span", "stage-row-title", entry.run_id));
  head.appendChild(badge(entry.stage === "sft" ? "SFT" : "预训练", entry.stage === "sft" ? "badge-warn" : "badge-ok"));
  head.appendChild(badge(entry.split || "?", "badge-wait"));
  row.appendChild(head);
  row.appendChild(el("div", "stage-row-meta", metricChips(entry.metrics || {}).join(" · ") || "（指标缺失）"));
  const extra = [];
  if (entry.prediction_count) extra.push(`${fmtInt(entry.prediction_count)} 条预测`);
  if (entry.has_boundary) extra.push("能力边界草稿");
  extra.push(entry.eval_dir);
  row.appendChild(el("div", "stage-row-sub", extra.join(" · ")));
  return row;
}

/**
 * 只同步列表行的选中态（列表签名未变时的轻量路径）。
 * @returns {void}
 */
function markActive() {
  const list = byId("eval-list");
  if (!list) return;
  list.querySelectorAll(".stage-row").forEach((row) => {
    const active = row.dataset.runId === state.selected && row.dataset.evalDir === state.selectedDir;
    row.classList.toggle("active", active);
    row.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

/** 列表签名：结构未变时跳过重建，保住滚动位置与焦点。 */
let listSig = "";

/**
 * 渲染左侧评测报告列表。
 * @param {object} payload /api/v1/evaluations 响应体：{ count, evaluations[] }。
 * @returns {void}
 */
function renderList(payload) {
  const root = byId("eval-list");
  if (!root) return;
  state.list = payload.evaluations || [];
  setStatus(`共 ${payload.count} 份评测报告`);
  const sig = state.list.map((entry) => `${entry.run_id}:${entry.eval_dir}:${entry.metrics && entry.metrics.perplexity}`).join("|");
  if (sig === listSig) {
    markActive();
    return;
  }
  listSig = sig;
  root.replaceChildren();
  if (!state.list.length) {
    root.appendChild(emptyBlock("还没有评测报告", "runs/ 下没有 evaluation.json；训练完成后运行 evaluate.py 生成评测。"));
    return;
  }
  state.list.forEach((entry) => root.appendChild(rowFor(entry)));
}

/**
 * 打开某份评测报告的详情（右侧面板）。
 * @param {string} runId 运行 id（runs/<runId>/）。
 * @param {string|null} evalDir 具体评测目录名（如 eval_test）；为空表示取最新一份。
 * @returns {Promise<void>} 失败时右侧显示 errorBlock，不抛出。
 */
async function openEvaluation(runId, evalDir) {
  state.selected = runId;
  state.selectedDir = evalDir || null;
  markActive();
  const title = byId("eval-detail-title");
  if (title) title.textContent = `${runId} · ${evalDir || "最新"}`;
  const root = byId("eval-detail");
  if (!root) return;
  const seq = ++detailSeq;
  root.replaceChildren(loadingBlock("载入评测详情…"));
  try {
    const query = evalDir ? `?eval=${encodeURIComponent(evalDir)}` : "";
    const detail = await getJson(endpoints.evaluation(runId, query));
    if (seq !== detailSeq || state.selected !== runId) return;
    renderDetail(detail);
  } catch (error) {
    if (seq !== detailSeq || state.selected !== runId) return;
    root.replaceChildren(errorBlock(error.message, "重试", () => openEvaluation(runId, evalDir)));
  }
}

/**
 * 渲染评测详情：元信息 + 指标 + 样本生成 + 分类指标 + 预测样例 + 能力边界草稿。
 * @param {object} detail GET /api/v1/evaluations/<run_id> 响应体。
 * @returns {void}
 */
function renderDetail(detail) {
  const root = byId("eval-detail");
  if (!root) return;
  root.replaceChildren();
  const metrics = detail.metrics || {};
  const report = detail.report || {};
  const rows = [
    ["运行", detail.run_id],
    ["目录", detail.eval_dir],
    ["阶段", detail.stage || "—"],
    ["切分", detail.split || "—"],
    ["checkpoint", detail.checkpoint_name || "—"],
    ["token 数", fmtInt(detail.tokens)],
    ["创建时间", detail.created_at || "—"],
  ];
  // 只展示实际存在的指标：预训练报告没有 answer_loss/EM，SFT 报告没有 ppl。
  const metricRows = [
    ["loss", metrics.loss],
    ["perplexity", metrics.perplexity],
    ["bpc", metrics.bpc],
    ["answer_loss", metrics.answer_loss],
    ["exact_match", metrics.exact_match],
    ["char_f1", metrics.char_f1],
    ["empty_rate", metrics.empty_rate],
    ["repetition_rate", metrics.repetition_rate],
  ];
  metricRows.forEach(([key, value]) => {
    if (value !== undefined && value !== null) rows.push([key, fmtNumber(value, 4)]);
  });
  root.appendChild(kv(rows));

  if (report.sample) {
    root.appendChild(section("样本生成（评估时固定 prompt）"));
    if (report.sample_prompt) {
      root.appendChild(el("pre", "stage-preview", `prompt: ${report.sample_prompt}\n\n${report.sample}`));
    } else {
      root.appendChild(el("pre", "stage-preview", report.sample));
    }
  }

  // SFT 评测会带 by_category 分类指标（EM / char_f1 / 重复率）。
  const byCategory = (report.metrics && report.metrics.by_category) || null;
  if (byCategory) {
    root.appendChild(section("按类别指标"));
    const rowsByCategory = Object.entries(byCategory).map(([name, item]) => [
      name,
      fmtInt(item.count),
      fmtNumber(item.exact_match, 3),
      fmtNumber(item.char_f1, 3),
      item.repetition_rate === undefined ? "—" : pct(item.repetition_rate, 2),
    ]);
    root.appendChild(table(["category", "count", "EM", "char_f1", "重复率"], rowsByCategory));
  }

  if (detail.predictions && detail.predictions.length) {
    root.appendChild(section(`预测样例（前 ${detail.predictions.length} 条）`));
    const rowsPrediction = detail.predictions.map((item) => [
      item.q ?? "",
      item.reference ?? "",
      item.prediction ?? "",
      item.category ?? "",
      item.stopped_early ? "是" : "",
    ]);
    root.appendChild(table(["问题", "参考答案", "模型输出", "类别", "提前停止"], rowsPrediction));
  }

  if (detail.boundary) {
    // 能力边界由评测脚本自动生成草稿，必须人工复核后才能对外引用。
    root.appendChild(section("能力边界草稿（自动生成，需人工复核）"));
    root.appendChild(el("pre", "stage-preview", detail.boundary));
  }
}

/**
 * 拉取并渲染评测列表，必要时自动选中第一份或刷新当前报告。
 * @param {boolean} force true=忽略 loaded 缓存强制重扫。
 * @returns {Promise<void>} 失败时列表区显示 errorBlock，不抛出。
 */
async function load(force) {
  if (state.loading) return;
  if (state.loaded && !force) return;
  state.loading = true;
  setStatus("扫描 runs/ 下的评测报告…");
  const root = byId("eval-list");
  if (root && (!state.loaded || force)) root.replaceChildren(loadingBlock("扫描评测报告…"));
  try {
    const payload = await getJson(endpoints.evaluations());
    state.loaded = true;
    renderList(payload);
    if (!state.selected && state.list.length) {
      openEvaluation(state.list[0].run_id, state.list[0].eval_dir);
    } else if (state.selected) {
      // 优先找回「同一 run + 同一目录」，找不到就退回到该 run 的最新报告。
      const current = state.list.find((item) => item.run_id === state.selected && (!state.selectedDir || item.eval_dir === state.selectedDir))
        || state.list.find((item) => item.run_id === state.selected);
      if (current) openEvaluation(current.run_id, current.eval_dir);
    }
  } catch (error) {
    setStatus(`评测列表加载失败：${error.message}`, true);
    if (root) root.replaceChildren(errorBlock(error.message, "重试", () => load(true)));
  } finally {
    state.loading = false;
  }
}

/* ------------------------------------------------------------------ N08 发起评估 */

/**
 * 面板内评估（N08）的运行状态。
 * - visible：本视图当前是否可见（不可见时轮询跳过，但计时器继续跑）。
 * - polling：是否正在轮询 status 端点。
 * - runId：本次评估所属运行，同时用作 status 查询参数。
 * - optionsLoaded/optionsLoading：下拉选项只加载一次并去重并发请求。
 * - consecutiveFails：连续失败计数，达到阈值自动停表。
 */
const launch = {
  visible: false,
  polling: false,
  timer: null,
  runId: null,
  optionsLoaded: false,
  optionsLoading: false,
  consecutiveFails: 0,
};

/**
 * 更新评估启动区的进度文本（可带 spinner）。
 * @param {string} message 文案。
 * @param {boolean} [isError=false] true 时加 error 样式。
 * @param {boolean} [busy=false] true 时前置转圈图标（进行中）。
 * @returns {void}
 */
function launchStatus(message, isError = false, busy = false) {
  const node = byId("eval-launch-status");
  if (!node) return;
  node.replaceChildren();
  if (busy) node.appendChild(el("span", "spinner"));
  node.appendChild(el("span", null, message || ""));
  node.classList.toggle("error", Boolean(isError));
}

/**
 * 加载发起评估用的「运行」与「数据集」下拉选项。
 * 只列出 has_best（有 best.pt）的运行；选项用签名差量同步，聚焦时不重建。
 * @returns {Promise<void>} 并发请求由 optionsLoading 去重；失败只更新状态文案。
 */
async function loadLaunchOptions() {
  if (launch.optionsLoading) return;
  launch.optionsLoading = true;
  try {
    const [runsPayload, datasetsPayload] = await Promise.all([
      getJson(endpoints.runs()),
      getJson(endpoints.datasets()),
    ]);
    // 评估必须能从 checkpoint 加载权重，所以没有 best.pt 的运行不参与。
    const runs = ((runsPayload && runsPayload.runs) || []).filter((run) => run.has_best);
    syncSelectOptions(
      byId("eval-launch-run"),
      runs.map((run) => ({ value: run.run_id, label: `${run.run_id} · ${run.stage || "?"}` })),
      `eval-run|${runs.map((run) => run.run_id).join(",")}`,
    );
    const datasets = (datasetsPayload && datasetsPayload.datasets) || [];
    syncSelectOptions(
      byId("eval-launch-dataset"),
      datasets.map((item) => ({ value: item.name, label: `${item.name} · ${item.kind}` })),
      `eval-ds|${datasets.map((item) => item.name).join(",")}`,
    );
    launch.optionsLoaded = true;
  } catch (error) {
    launchStatus(`评估选项加载失败：${error.message}`, true);
  } finally {
    launch.optionsLoading = false;
  }
}

/**
 * 查询一次评估子进程状态，并据 running/starting/completed/failed 更新 UI。
 * 完成时顺带刷新报告列表并定位到新报告。
 * @returns {Promise<void>} 查询失败时累计 consecutiveFails，达 5 次自动停表。
 */
async function pollLaunchStatus() {
  if (!launch.runId) return;
  let payload;
  try {
    payload = await getJson(endpoints.evaluateStatus(launch.runId));
    launch.consecutiveFails = 0;
  } catch (error) {
    launch.consecutiveFails += 1;
    // 连续失败停表，避免无限重试；用户可再次发起或刷新。
    if (launch.consecutiveFails >= 5) {
      stopLaunchPolling();
      const button = byId("eval-launch-btn");
      if (button) button.disabled = false;
      launchStatus(`评估状态查询连续失败，已停止轮询：${error.message}`, true);
    } else {
      launchStatus(`评估状态查询失败（重试 ${launch.consecutiveFails}/5）：${error.message}`, true);
    }
    return;
  }
  const status = payload.status;
  if (status === "running" || status === "starting") {
    launchStatus(`评估进行中（${payload.dataset || ""} · ${payload.split || ""}）…`, false, true);
    return;
  }
  // 终态（completed/failed/其它）：先停表再恢复按钮，避免重复轮询。
  stopLaunchPolling();
  const button = byId("eval-launch-btn");
  if (button) button.disabled = false;
  if (status === "completed") {
    launchStatus(`评估完成：${payload.eval_dir || ""}`);
    // 刷新列表并直接定位到新报告（列表渲染会覆盖临时结果，这里以列表为准）。
    state.selected = payload.run_id || launch.runId;
    state.selectedDir = payload.eval_dir || null;
    await load(true);
    const entry = state.list.find((item) => item.run_id === state.selected && item.eval_dir === state.selectedDir);
    if (entry) openEvaluation(entry.run_id, entry.eval_dir);
  } else if (status === "failed") {
    launchStatus(`评估失败：${payload.error || "详见 eval_console.log"}`, true);
  } else {
    launchStatus("");
  }
}

/**
 * 启动评估状态轮询（1.5s 一次）。
 * @param {string} runId 本次评估所属运行 id，作为 status 查询参数。
 * @returns {void} 会先清掉已有计时器，保证同一时刻只有一个轮询循环。
 */
function startLaunchPolling(runId) {
  launch.runId = runId;
  launch.polling = true;
  launch.consecutiveFails = 0;
  if (launch.timer !== null) window.clearInterval(launch.timer);
  launch.timer = window.setInterval(() => {
    // 页面不可见或用户暂停刷新时跳过本轮（计时器保留，恢复后自动继续）。
    if (!launch.visible || store.paused) return;
    pollLaunchStatus();
  }, 1500);
  pollLaunchStatus();
}

/**
 * 停止评估状态轮询并释放计时器。
 * @returns {void}
 */
function stopLaunchPolling() {
  launch.polling = false;
  if (launch.timer !== null) {
    window.clearInterval(launch.timer);
    launch.timer = null;
  }
}

/**
 * 提交「发起评估」表单：POST /api/v1/evaluate 后转入状态轮询。
 * @param {SubmitEvent} event 表单提交事件（会被 preventDefault）。
 * @returns {Promise<void>} 失败时恢复按钮并显示后端错误文本。
 */
async function submitLaunch(event) {
  event.preventDefault();
  const runSelect = byId("eval-launch-run");
  const datasetSelect = byId("eval-launch-dataset");
  const splitSelect = byId("eval-launch-split");
  const runId = runSelect ? runSelect.value : "";
  const dataset = datasetSelect ? datasetSelect.value : "";
  if (!runId || !dataset) {
    launchStatus("请先选择运行与数据集。", true);
    return;
  }
  const button = byId("eval-launch-btn");
  if (button) button.disabled = true;
  launchStatus("正在启动评估子进程…", false, true);
  try {
    const payload = await postJson(endpoints.evaluate(), {
      run_id: runId,
      dataset,
      split: splitSelect ? splitSelect.value : "test",
    });
    launchStatus(`评估已启动${payload.pid ? `（pid ${payload.pid}）` : ""}`, false, true);
    startLaunchPolling(runId);
  } catch (error) {
    // 启动失败要立刻恢复按钮，否则用户无法重试。
    if (button) button.disabled = false;
    launchStatus(`发起评估失败：${error.message}`, true);
  }
}

/**
 * 绑定「发起评估」表单的 submit（表单内按钮为 type=submit，统一走这里）。
 * @returns {void}
 */
function initLaunchForm() {
  const form = byId("eval-launch-form");
  if (!form) return;
  form.addEventListener("submit", submitLaunch);
}

/**
 * 初始化评测视图：报告列表交互 + 面板内评估表单 + spa:view 生命周期。
 * @returns {void} 页面里没有 #eval-list 时直接返回。
 */
export function initEvalView() {
  const list = byId("eval-list");
  if (!list) return;
  initLaunchForm();
  bindListKeyboard(list);
  // 事件委托：行会被整段重建，绑在容器上才不会丢监听。
  list.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !target.closest) return;
    const row = target.closest(".stage-row");
    if (row && row.dataset.runId) openEvaluation(row.dataset.runId, row.dataset.evalDir);
  });
  const refresh = byId("eval-refresh-btn");
  if (refresh) refresh.addEventListener("click", () => {
    load(true);
    loadLaunchOptions();
  });
  window.addEventListener("spa:view", (event) => {
    const visible = Boolean(event.detail && event.detail.view === "eval");
    launch.visible = visible;
    if (visible) {
      load(false);
      if (!launch.optionsLoaded) loadLaunchOptions();
      // 从别的页签回来时补一次状态查询，不必等下一个 1.5s 周期。
      if (launch.polling && launch.runId) pollLaunchStatus();
    }
  });
}
