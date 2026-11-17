/* 导出阶段视图：只读扫描 dist/ 下的 MiniGPT Bundle v1 导出包。
 *
 * 在 SPA 中的位置：路由 #/export 对应的 section#view-export；app.js 调用 initExportView()。
 * 数据来源：GET /api/v1/exports（列表 + dist/ 下的 zip 清单）与
 *           GET /api/v1/exports/<name>（详情：MANIFEST、config.json、MODEL_CARD.md 开头）。
 * 依赖：api.js（端点）、ui.js（DOM 与格式化基座）。
 *
 * 关键约束：
 *  - 本页只读；重新导出请运行 export.py。MANIFEST 里的 sha256 是完整性校验依据，
 *    这里只展示前 16 位便于人眼对照；
 *  - 首次进入才拉列表，刷新按钮 force=true 强制重扫；
 *  - 详情请求带自增序号，快速切换导出包时旧响应一律丢弃；
 *  - 模型卡与配置文本统一 textContent 渲染，不当作 HTML。
 */

import {
  byId, el, fmtInt, bytesText, badge, getJson, kv, table, section,
  loadingBlock, emptyBlock, errorBlock, makeRowActivatable, bindListKeyboard,
} from "./ui.js";
import { endpoints } from "./api.js";

/** 视图局部状态：loaded/loading 去重扫描，selected 记录当前导出包目录名。 */
const state = { loaded: false, loading: false, selected: null, list: [] };
/** 详情请求序号：避免快速切换时旧响应覆盖新选择。 */
let detailSeq = 0;

/**
 * 更新列表区标题右侧的状态行。
 * @param {string} message 展示文本，空串表示清空。
 * @param {boolean} [isError=false] true 时加 error 样式。
 * @returns {void}
 */
function setStatus(message, isError = false) {
  const node = byId("export-count");
  if (!node) return;
  node.textContent = message || "";
  node.classList.toggle("error", Boolean(isError));
}

/**
 * 构造列表中的一行导出包卡片。
 * @param {object} item /api/v1/exports 的 exports[] 元素：
 *   { name, format_version, has_zip, parameters, run_id, file_count, total_bytes,
 *     created_at, missing_files[] }。
 * @returns {HTMLElement} 可点击/可键盘激活的 .stage-row。
 */
function rowFor(item) {
  const row = el("div", "stage-row");
  makeRowActivatable(row);
  row.dataset.exportName = item.name;
  const active = item.name === state.selected;
  if (active) row.classList.add("active");
  row.setAttribute("aria-pressed", active ? "true" : "false");
  const head = el("div", "stage-row-head");
  head.appendChild(el("span", "stage-row-title", item.name));
  head.appendChild(badge(item.format_version ? `v${item.format_version}` : "bundle", "badge-ok"));
  head.appendChild(badge(item.has_zip ? "含 zip" : "无 zip", item.has_zip ? "badge-ok" : "badge-warn"));
  row.appendChild(head);
  const meta = [];
  if (item.parameters) meta.push(`${fmtInt(item.parameters)} 参数`);
  if (item.run_id) meta.push(item.run_id);
  meta.push(`${item.file_count} 个文件`);
  meta.push(bytesText(item.total_bytes));
  row.appendChild(el("div", "stage-row-meta", meta.join(" · ")));
  // MANIFEST 声明但磁盘缺失的文件优先提示，比创建时间更有诊断价值。
  if (item.missing_files && item.missing_files.length) {
    row.appendChild(el("div", "stage-row-sub error", `缺少：${item.missing_files.join(", ")}`));
  } else {
    row.appendChild(el("div", "stage-row-sub", item.created_at || ""));
  }
  return row;
}

/**
 * 只同步列表行的选中态（列表签名未变时的轻量路径）。
 * @returns {void}
 */
function markActive() {
  const list = byId("export-list");
  if (!list) return;
  list.querySelectorAll(".stage-row").forEach((row) => {
    const active = row.dataset.exportName === state.selected;
    row.classList.toggle("active", active);
    row.setAttribute("aria-pressed", active ? "true" : "false");
  });
}

/** 列表签名：结构未变时跳过重建，保留滚动位置。 */
let listSig = "";

/**
 * 渲染左侧导出包列表（含 dist/ 下独立 zip 的分区）。
 * @param {object} payload /api/v1/exports 响应体：{ count, exports[], zips[] }。
 * @returns {void}
 */
function renderList(payload) {
  const root = byId("export-list");
  if (!root) return;
  state.list = payload.exports || [];
  setStatus(`共 ${payload.count} 个导出包 · ${(payload.zips || []).length} 个 zip`);
  const sig = state.list.map((item) => `${item.name}:${item.total_bytes}:${item.file_count}`).join("|") + `#${(payload.zips || []).map((f) => f.name).join(",")}`;
  if (sig === listSig) {
    markActive();
    return;
  }
  listSig = sig;
  root.replaceChildren();
  if (!state.list.length) {
    root.appendChild(emptyBlock("还没有导出包", "dist/ 下没有 MANIFEST.json；训练完成后运行 export.py 导出模型。"));
    return;
  }
  state.list.forEach((item) => root.appendChild(rowFor(item)));
  if (payload.zips && payload.zips.length) {
    root.appendChild(section("压缩包"));
    payload.zips.forEach((file) => {
      const row = el("div", "stage-raw-row");
      row.appendChild(el("span", "", file.name));
      row.appendChild(el("span", "muted", bytesText(file.bytes)));
      root.appendChild(row);
    });
  }
}

/**
 * 打开某个导出包的详情（右侧面板）。
 * @param {string} name 导出包目录名（dist/<name>/）。
 * @returns {Promise<void>} 失败时右侧显示 errorBlock，不抛出。
 */
async function openExport(name) {
  state.selected = name;
  markActive();
  const title = byId("export-detail-title");
  if (title) title.textContent = name;
  const root = byId("export-detail");
  if (!root) return;
  const seq = ++detailSeq;
  root.replaceChildren(loadingBlock("载入导出包详情…"));
  try {
    const detail = await getJson(endpoints.export(name));
    if (seq !== detailSeq || state.selected !== name) return;
    renderDetail(detail);
  } catch (error) {
    if (seq !== detailSeq || state.selected !== name) return;
    root.replaceChildren(errorBlock(error.message, "重试", () => openExport(name)));
  }
}

/**
 * 渲染导出包详情：元信息、config.json 摘要、MANIFEST 文件清单、模型卡开头。
 * @param {object} detail GET /api/v1/exports/<name> 响应体。
 * @returns {void}
 */
function renderDetail(detail) {
  const root = byId("export-detail");
  if (!root) return;
  root.replaceChildren();
  // Bundle v1 的模型超参嵌套在 model_config 下；兼容旧版平铺结构。
  const rawConfig = detail.config || {};
  const config = rawConfig.model_config || rawConfig;
  root.appendChild(
    kv([
      ["名称", detail.name],
      ["格式", `${detail.format || "?"} v${detail.format_version ?? "?"}`],
      // 源运行可能已被删除（只留下导出包），这里明确标注而不是显示为正常。
      ["来源运行", detail.run_id ? `${detail.run_id}${detail.run_missing ? "（源运行已缺失）" : ""}` : "—"],
      ["参数量", fmtInt(detail.parameters)],
      ["checkpoint", detail.checkpoint_name || "—"],
      ["创建时间", detail.created_at || "—"],
      ["文件数", String(detail.file_count)],
      ["总计", bytesText(detail.total_bytes)],
      ["zip", detail.has_zip ? bytesText(detail.zip_bytes) : "没有"],
      ["模型卡", detail.model_card_bytes ? bytesText(detail.model_card_bytes) : "没有"],
    ]),
  );

  if (Object.keys(config).length) {
    root.appendChild(section("config.json"));
    root.appendChild(
      kv([
        ["模型", `${config.n_layer}L/${config.n_head}H/${config.n_embd}D · block ${config.block_size}`],
        ["词表", fmtInt(config.vocab_size)],
        ["dropout / bias / tie", `${config.dropout} / ${config.bias} / ${config.tie_weights}`],
        ["backend", String(config.backend_type || "torch")],
      ]),
    );
  }

  if (detail.manifest_files && detail.manifest_files.length) {
    root.appendChild(section("MANIFEST 文件清单（sha256 用于完整性校验）"));
    root.appendChild(
      table(
        ["文件", "字节", "sha256"],
        detail.manifest_files.map((file) => [file.name, bytesText(file.bytes), `${String(file.sha256 || "").slice(0, 16)}…`]),
      ),
    );
  }

  if (detail.model_card) {
    // 后端已截取开头部分（避免把整份模型卡传过来）。
    root.appendChild(section("MODEL_CARD.md（开头）"));
    root.appendChild(el("pre", "stage-preview", detail.model_card));
  }
}

/**
 * 拉取并渲染导出包列表，必要时自动选中第一项或刷新当前详情。
 * @param {boolean} force true=忽略 loaded 缓存强制重扫。
 * @returns {Promise<void>} 失败时列表区显示 errorBlock，不抛出。
 */
async function load(force) {
  if (state.loading) return;
  if (state.loaded && !force) return;
  state.loading = true;
  setStatus("扫描 dist/ …");
  const root = byId("export-list");
  if (root && (!state.loaded || force)) root.replaceChildren(loadingBlock("扫描 dist/ 目录…"));
  try {
    const payload = await getJson(endpoints.exports());
    state.loaded = true;
    renderList(payload);
    if (!state.selected && state.list.length) {
      openExport(state.list[0].name);
    } else if (state.selected) {
      // 刷新后维持选择；包被删除则回到空态提示。
      const stillExists = state.list.some((item) => item.name === state.selected);
      if (stillExists) openExport(state.selected);
      else {
        state.selected = null;
        const detail = byId("export-detail");
        if (detail) detail.replaceChildren(emptyBlock("选择一个导出包", "左侧列表点击查看详情。"));
      }
    }
  } catch (error) {
    setStatus(`导出包加载失败：${error.message}`, true);
    if (root) root.replaceChildren(errorBlock(error.message, "重试", () => load(true)));
  } finally {
    state.loading = false;
  }
}

/**
 * 初始化导出视图：列表点击（事件委托）、键盘激活、刷新按钮、spa:view 首次加载。
 * @returns {void} 页面里没有 #export-list 时直接返回。
 */
export function initExportView() {
  const list = byId("export-list");
  if (!list) return;
  bindListKeyboard(list);
  // 事件委托：行会被整段重建，绑在容器上才不会丢监听。
  list.addEventListener("click", (event) => {
    const target = event.target;
    if (!target || !target.closest) return;
    const row = target.closest(".stage-row");
    if (row && row.dataset.exportName) openExport(row.dataset.exportName);
  });
  const refresh = byId("export-refresh-btn");
  if (refresh) refresh.addEventListener("click", () => load(true));
  window.addEventListener("spa:view", (event) => {
    if (event.detail && event.detail.view === "export") load(false);
  });
}
