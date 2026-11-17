/* 数据阶段视图：只读扫描 data/ 下的预训练语料与 SFT 数据集。
 *
 * 在 SPA 中的位置：路由 #/data 对应的 section#view-data；app.js 调用 initDataView() 注册事件。
 * 数据来源：GET /api/v1/datasets（列表 + data/raw 缓存清单）与
 *           GET /api/v1/datasets/<name>（详情，含样例 / 预览）。
 * 依赖：api.js（端点）、ui.js（DOM 与格式化基座）、store 不参与（本页无轮询、无暂停逻辑）。
 *
 * 关键约束：
 *  - 本页只读：不准备数据、不改文件；重新准备请用 prepare.py / prepare_sft.py；
 *  - 布局固定视口：左侧列表、右侧详情，各自滚动（CSS 负责，JS 只切 hidden/内容）；
 *  - 首次进入才拉列表（state.loaded 守卫），刷新按钮 force=true 强制重扫；
 *  - 详情请求带自增序号，连点多个数据集时旧响应一律丢弃；
 *  - 所有训练产物文本统一走 textContent 渲染，防止数据集内容被当作 HTML 执行。
 */

import {
  byId, el, fmtInt, bytesText, pct, badge, getJson, kv, table, section,
  loadingBlock, emptyBlock, errorBlock, makeRowActivatable, bindListKeyboard,
} from "./ui.js";
import { endpoints } from "./api.js";

/** 视图局部状态：loaded/loading 用于去重扫描，selected 记录当前数据集名。 */
const state = { loaded: false, loading: false, selected: null, list: [] };
/** 详情请求序号：防止连点多个数据集时旧响应覆盖新选择。 */
let detailSeq = 0;

/**
 * 更新面板标题右侧的状态行（数据集计数 / 错误信息）。
 * @param {string} message 展示文本，空串表示清空。
 * @param {boolean} [isError=false] true 时加 error 样式。
 * @returns {void}
 */
function setStatus(message, isError = false) {
  const node = byId("data-count");
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("error", Boolean(isError));
}

/**
 * 构造列表中的一行数据集卡片。
 * @param {object} dataset /api/v1/datasets 的 datasets[] 元素：
 *   { name, kind: "sft"|其它, bytes, files[], vocab_size, splits, counts?, coverage? }。
 * @returns {HTMLElement} 可点击/可键盘激活的 .stage-row。
 */
function rowFor(dataset) {
  const row = el("div", "stage-row");
  makeRowActivatable(row);
  // data-* 只存名字：点击时据此回到 openDataset()，不依赖闭包，重建后仍然有效。
  row.dataset.dataset = dataset.name;
  const active = dataset.name === state.selected;
  if (active) row.classList.add("active");
  row.setAttribute("aria-pressed", active ? "true" : "false");
  const head = el("div", "stage-row-head");
  head.appendChild(el("span", "stage-row-title", dataset.name));
  head.appendChild(badge(dataset.kind === "sft" ? "SFT" : "预训练", dataset.kind === "sft" ? "badge-warn" : "badge-ok"));
  head.appendChild(badge(`V=${dataset.vocab_size ?? "?"}`, "badge-wait"));
  row.appendChild(head);
  const splits = dataset.splits || {};
  const parts = [];
  // SFT 数据集按「样本数」统计，预训练语料按「token 数」统计。
  if (dataset.kind === "sft") {
    parts.push(`train ${fmtInt(splits.train?.examples)}`);
  } else {
    parts.push(`train ${fmtInt(splits.train?.tokens)} token`);
  }
  if (dataset.counts && dataset.counts.kept) parts.push(`保留 ${fmtInt(dataset.counts.kept)}`);
  if (dataset.coverage) parts.push(`覆盖率 ${pct(dataset.coverage, 2)}`);
  row.appendChild(el("div", "stage-row-meta", parts.join(" · ")));
  row.appendChild(el("div", "stage-row-sub", `${bytesText(dataset.bytes)} · ${dataset.files.length} 个文件`));
  return row;
}

/**
 * 只同步列表行的选中态（列表结构未变时的轻量路径，不重建 DOM）。
 * @returns {void}
 */
function markActive() {
  const list = byId("data-list");
  if (!list) return;
  list.querySelectorAll(".stage-row").forEach((row) => {
    const active = row.dataset.dataset === state.selected;
    row.classList.toggle("active", active);
    row.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

/** 列表签名：内容未变时跳过整段重建，避免轮询/刷新夺走滚动位置。 */
let listSig = "";

/**
 * 渲染左侧数据集列表（含 data/raw 原始缓存分区）。
 * @param {object} payload /api/v1/datasets 响应体：
 *   { count, pretrain_count, sft_count, datasets[], raw_files[] }。
 * @returns {void}
 */
function renderList(payload) {
  const root = byId("data-list");
  if (!root) return;
  state.list = payload.datasets || [];
  setStatus(`共 ${payload.count} 个数据集（预训练 ${payload.pretrain_count} · SFT ${payload.sft_count}）`);
  // 签名包含 raw_files 名单，因为原始缓存区也在同一容器里渲染。
  const sig = state.list.map((d) => `${d.name}:${d.bytes}:${d.vocab_size}`).join("|") + `#${(payload.raw_files || []).map((f) => f.name).join(",")}`;
  if (sig === listSig) {
    markActive();
    return;
  }
  listSig = sig;
  root.replaceChildren();
  if (!state.list.length) {
    root.appendChild(emptyBlock("还没有数据集", "data/ 下没有带 meta.json / sft_meta.json 的数据集；先用 prepare.py / prepare_sft.py 准备数据。"));
    return;
  }
  state.list.forEach((dataset) => root.appendChild(rowFor(dataset)));
  if (payload.raw_files && payload.raw_files.length) {
    root.appendChild(section("原始缓存 data/raw/"));
    payload.raw_files.forEach((file) => {
      const row = el("div", "stage-raw-row");
      row.appendChild(el("span", "", file.name));
      row.appendChild(el("span", "muted", bytesText(file.bytes)));
      root.appendChild(row);
    });
  }
}

/**
 * 打开某个数据集的详情（右侧面板）。
 * @param {string} name 数据集目录名（data/<name>/）。
 * @returns {Promise<void>} 失败时右侧面板显示 errorBlock，不抛出。
 */
async function openDataset(name) {
  state.selected = name;
  markActive();
  const title = byId("data-detail-title");
  if (title) title.textContent = name;
  const root = byId("data-detail");
  if (!root) return;
  // 每次请求自增序号：连点多个数据集时只认最后一次。
  const seq = ++detailSeq;
  root.replaceChildren(loadingBlock("载入数据集详情…"));
  try {
    const detail = await getJson(endpoints.dataset(name, 10));
    // 连点多个数据集时，旧响应不得覆盖新选择。
    if (seq !== detailSeq || state.selected !== name) return;
    renderDetail(detail);
  } catch (error) {
    if (seq !== detailSeq || state.selected !== name) return;
    root.replaceChildren(errorBlock(error.message, "重试", () => openDataset(name)));
  }
}

/**
 * 渲染数据集详情：元信息表 + 切分表 + tokenizer + 样例/预览。
 * @param {object} detail GET /api/v1/datasets/<name> 响应体：
 *   { kind, meta_file, source, vocab_size, splits, files, counts, tokenizer, samples, preview, ... }。
 * @returns {void}
 */
function renderDetail(detail) {
  const root = byId("data-detail");
  if (!root) return;
  root.replaceChildren();

  const source = detail.source || {};
  const rows = [
    ["类型", detail.kind === "sft" ? "SFT 指令数据" : "预训练语料"],
    ["元信息", detail.meta_file],
    ["来源", source.name || "—"],
    ["来源类型", source.kind || "—"],
    ["许可", source.license || "—"],
    // sha256 太长，只展示前 16 位作为人眼对照，完整值在 meta.json 里。
    ["sha256", source.sha256 ? `${String(source.sha256).slice(0, 16)}…` : "—"],
    ["词表大小", fmtInt(detail.vocab_size)],
    ["创建时间", detail.created_at || "—"],
    ["seed", detail.seed ?? "—"],
    ["文件数量", String((detail.files || []).length)],
    ["总大小", bytesText(detail.bytes)],
  ];
  if (detail.kind === "sft") {
    // SFT：关注输入/有效/保留与去重、同组不跨切分等数据质量指标。
    const counts = detail.counts || {};
    rows.push(["输入 / 有效 / 保留", `${fmtInt(counts.input)} / ${fmtInt(counts.valid)} / ${fmtInt(counts.kept)}`]);
    rows.push(["无效 / 重复", `${fmtInt(counts.invalid)} / ${fmtInt(counts.duplicates)}`]);
    rows.push(["group 完整性", detail.group_integrity ? "true（同组不跨切分）" : "未记录"]);
    rows.push(["tokenizer 覆盖率", detail.coverage ? `${pct(detail.coverage, 4)}` : "—"]);
    rows.push(["格式", detail.format || "—"]);
  } else {
    // 预训练：关注词表补充与 val/test 切分比例。
    const extra = detail.vocab_extra || {};
    rows.push(["词表补充", extra.enabled ? `${extra.files?.length || 0} 个文件 / 新增 ${extra.novel_chars ?? 0} 字` : "未启用"]);
    const meta = detail.meta || {};
    rows.push(["切分比例", `${meta.val_fraction ?? "?"} / ${meta.test_fraction ?? "?"}`]);
  }
  root.appendChild(kv(rows));

  const splits = detail.splits || {};
  const splitRows = Object.keys(splits).map((split) => {
    const item = splits[split] || {};
    // unk 率按切分记录；缺失（旧数据集）时显示破折号。
    const unk = detail.unk_rate ? pct(detail.unk_rate[split], 3) : "—";
    return [split, fmtInt(item.tokens ?? item.examples), bytesText(item.bytes), `${String(item.sha256 || "").slice(0, 12)}…`, unk];
  });
  root.appendChild(section("切分"));
  root.appendChild(table(["split", detail.kind === "sft" ? "样本数" : "token 数", "字节", "sha256", "unk 率"], splitRows));

  if (detail.tokenizer) {
    root.appendChild(section("tokenizer"));
    root.appendChild(
      kv([
        ["类型", `${detail.tokenizer.type || "?"} v${detail.tokenizer.version ?? "?"}`],
        ["词表大小", fmtInt(detail.tokenizer.vocab_size)],
        ["特殊 token", Object.entries(detail.tokenizer.specials || {}).map(([name, id]) => `${name}=${id}`).join("  ")],
      ]),
    );
  }

  if (detail.kind === "sft") {
    root.appendChild(section(`样例（train 前 ${detail.samples.length} 条）`));
    detail.samples.forEach((sample, index) => {
      const card = el("div", "stage-sample");
      card.appendChild(el("div", "stage-sample-head", `#${index + 1}  ${sample.group || ""}`));
      card.appendChild(el("div", "stage-sample-q", `Q: ${sample.q ?? ""}`));
      card.appendChild(el("div", "stage-sample-a", `A: ${sample.a ?? ""}`));
      if (sample.category) card.appendChild(el("div", "stage-sample-meta", `category=${sample.category}`));
      root.appendChild(card);
    });
  } else if (detail.preview) {
    // 预训练语料只给出前若干 token 的纯文本预览（textContent 写入 <pre>）。
    root.appendChild(section(`语料预览（train 前 ${detail.preview.tokens} 个 token）`));
    root.appendChild(el("pre", "stage-preview", detail.preview.text));
  }
}

/**
 * 拉取并渲染数据集列表，必要时自动选中第一项或刷新当前详情。
 * @param {boolean} force true=忽略 loaded 缓存强制重扫（刷新按钮 / 失败重试）。
 * @returns {Promise<void>} 失败时列表区显示 errorBlock，不抛出。
 */
async function load(force) {
  if (state.loading) return;
  if (state.loaded && !force) return;
  state.loading = true;
  setStatus("扫描 data/ …");
  const root = byId("data-list");
  if (root && (!state.loaded || force)) root.replaceChildren(loadingBlock("扫描 data/ 目录…"));
  try {
    const payload = await getJson(endpoints.datasets());
    state.loaded = true;
    renderList(payload);
    if (!state.selected && state.list.length) {
      // 首屏默认展开第一项，省一次点击。
      openDataset(state.list[0].name);
    } else if (state.selected) {
      // 刷新后维持当前选择；数据集被删掉则回到空态提示。
      const stillExists = state.list.some((item) => item.name === state.selected);
      if (stillExists) openDataset(state.selected);
      else {
        state.selected = null;
        const detail = byId("data-detail");
        if (detail) detail.replaceChildren(emptyBlock("选择一个数据集", "左侧列表点击查看详情。"));
      }
    }
  } catch (error) {
    setStatus(`数据集加载失败：${error.message}`, true);
    if (root) root.replaceChildren(errorBlock(error.message, "重试", () => load(true)));
  } finally {
    state.loading = false;
  }
}

/**
 * 初始化数据视图：列表点击（事件委托）、键盘激活、刷新按钮、spa:view 首次加载。
 * @returns {void} 页面里没有 #data-list 时（例如单页测试）直接返回。
 */
export function initDataView() {
  const list = byId("data-list");
  if (!list) return;
  bindListKeyboard(list);
  // 事件委托：行会被整段重建，绑在容器上才不会丢监听。
  list.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !target.closest) return;
    const row = target.closest(".stage-row");
    if (row && row.dataset.dataset) openDataset(row.dataset.dataset);
  });
  const refresh = byId("data-refresh-btn");
  if (refresh) refresh.addEventListener("click", () => load(true));
  // 只在真正切到本视图时加载；已加载过则复用缓存（load 内部有 loaded 守卫）。
  window.addEventListener("spa:view", (event) => {
    if (event.detail && event.detail.view === "data") load(false);
  });
}
