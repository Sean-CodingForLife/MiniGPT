/* MiniGPT 控制台 · 训练工作区编排层。
 *
 * 分层：
 *   train/config-form.js  配置编辑器（字段元数据 / 表单 / JSON 双模式）
 *   train/charts.js       训练曲线（签名差量更新 + 主题联动）
 *   本文件                状态 / 轮询 / 运行卡 / 面板 / 动作 / 抽屉
 *
 * 约束：
 *  - 不依赖任何 CDN / 外部网络，Chart.js 从 /static/vendor 本地加载；
 *  - 单一轮询循环：运行详情每秒、任务列表约 2.5 秒，页面隐藏或暂停时停止；
 *  - 请求序号 + AbortController 防乱序：切换运行/阶段后旧响应一律丢弃；
 *  - 事件流按 index 去重合并，游标独占；面板按签名差量渲染，不夺焦点；
 *  - 启动时各端点独立降级，失败项由轮询自动重试；
 *  - 所有渲染都用 textContent，训练产出的文本不会被当作 HTML 执行。
 */
import { endpoints, getJson, postJson, deleteJson } from "./api.js";
import {
  byId, el, badge, fmtSmart, fmtInt, fmtTime, fmtElapsed, showToast, copyText,
  confirmDialog, promptDialog, syncSelectOptions, bindSelectFlush, renderSigned,
} from "./ui.js";
import { store, setPaused, on } from "./store.js";
import { createConfigForm } from "./train/config-form.js";
import { createTrainCharts } from "./train/charts.js";

const POLL_MS = 1000;
const LIST_POLL_MS = 2500;
const CONNECTION_STALE_MS = 8000;
const OFFLINE_FAIL_THRESHOLD = 3;
const RUN_STALE_MS = 60000;
const MAX_RUN_OPTIONS = 80;
const MAX_EVENT_ROWS = 50;
const EVENT_KEEP = 2000;

const STATUS_LABEL = {
  created: "已创建（未启动）",
  starting: "启动中",
  running: "运行中",
  stopping: "停止中",
  completed: "已完成",
  stopped: "已停止",
  failed: "失败",
  unknown: "未知",
};

const STAGE_LABEL = { pretrain: "预训练", sft: "SFT" };
const STAGE_PRESET_DEFAULT = { pretrain: "debug_en", sft: "sft_debug_en" };

const state = {
  currentRun: null,
  lastEventAfter: 0,
  configs: [],
  jobs: [],
  runs: [],
  presets: {},
  presetNames: [],
  events: [],
  eventIndex: new Set(),
  lastSummary: null,
  lastUpdateOk: 0,
  editorConfig: {},
  editorGeneration: 0,
  stage: "pretrain",
  stageReady: false,
  drawerOpen: false,
  activeView: "train",
  lastListFetch: 0,
  consecutiveFails: 0,
  offline: false,
  lastMetrics: null,
  lastMetricsRow: null,
  subView: "run",
  monitorTab: "events",
  initAlignRun: "",
  eventFollow: true,
  drawerReturnFocus: null,
  releaseDrawerTrap: null,
  failureDismissedFor: null,
  failNotified: {},
  pendingSftFrom: null,
  presetsLoaded: false,
  configsLoaded: false,
  systemLoaded: false,
};

/* ------------------------------------------------------------------ 配置字段元数据 */
/* FIELD_GROUPS / FIELD_META 是配置表单的单一数据源，注入 train/config-form.js。
 * 书写格式为测试契约（tests/test_form_ui.py 按字面块解析），勿改动缩进。 */
/* eslint-disable no-var */
var FIELD_GROUPS = [
    {
      title: "运行与路径",
      open: true,
      fields: ["stage", "run_name", "data_dir", "runs_dir", "seed", "device", "dtype"],
    },
    {
      title: "模型结构",
      open: true,
      fields: ["n_layer", "n_head", "n_embd", "block_size", "dropout", "bias", "tie_weights"],
    },
    {
      title: "训练超参数",
      open: true,
      fields: [
        "batch_size",
        "grad_accum_steps",
        "max_steps",
        "lr",
        "min_lr",
        "warmup_steps",
        "weight_decay",
        "beta1",
        "beta2",
        "grad_clip",
      ],
    },
    {
      title: "评估、保存与早停",
      open: true,
      fields: ["eval_interval", "eval_iters", "log_interval", "save_interval", "patience", "min_delta"],
    },
    {
      title: "数据准备",
      open: false,
      fields: [
        "dataset",
        "input_path",
        "vocab_extra",
        "download_dir",
        "val_fraction",
        "test_fraction",
        "zh_books",
        "source_url",
        "sft_input",
        "sft_source",
        "max_samples",
        "tokenizer_dir",
      ],
    },
    {
      title: "SFT",
      open: false,
      fields: ["init_from", "pretrain_data_dir", "pretrain_mix_ratio"],
    },
    {
      title: "生成与采样",
      open: false,
      fields: ["sample_prompt", "temperature", "top_k", "top_p", "max_new_tokens"],
    },
    {
      title: "运行模式与扩展",
      open: false,
      fields: [
        "resume",
        "mini_run",
        "gate",
        "backend_type",
        "tokenizer_type",
        "model_type",
        "callbacks",
        "plugins",
        "num_threads",
        "eval_seed",
      ],
    },
    {
      title: "Web 与部署",
      open: false,
      fields: ["web_host", "web_port", "max_jobs", "web_dir", "project_root"],
    },
  ];

  var FIELD_META = {
    stage: {
      label: "阶段",
      type: "select",
      options: ["pretrain", "sft"],
      help: "pretrain=预训练；sft=指令微调（建议同时切换对应预设）",
    },
    run_name: { label: "运行名", help: "留空自动生成；恢复训练时填已有 run 名" },
    data_dir: { label: "数据目录", help: "prepare.py / prepare_sft.py 的输出目录" },
    runs_dir: { label: "运行根目录", help: "所有 run 写到这里" },
    seed: { label: "随机种子", type: "number", step: 1 },
    device: { label: "设备", type: "select", options: ["auto", "cpu", "cuda"], help: "auto 优先 CUDA，否则 CPU" },
    dtype: {
      label: "精度",
      type: "select",
      options: ["auto", "float32", "bfloat16", "float16"],
      help: "CPU 默认 float32；CUDA 默认 bf16/fp16",
    },
    n_layer: { label: "层数", type: "number", step: 1, min: 1 },
    n_head: { label: "注意力头数", type: "number", step: 1, min: 1 },
    n_embd: { label: "隐藏维度", type: "number", step: 1, min: 8, help: "必须能被注意力头数整除" },
    block_size: { label: "上下文长度", type: "number", step: 1, min: 2 },
    dropout: { label: "Dropout", type: "number", step: 0.01, min: 0, max: 0.99 },
    bias: { label: "使用 bias", type: "bool" },
    tie_weights: { label: "权重绑定", type: "bool" },
    batch_size: { label: "Batch size", type: "number", step: 1, min: 1 },
    grad_accum_steps: { label: "梯度累积步数", type: "number", step: 1, min: 1, help: "显存不足时调大" },
    max_steps: { label: "最大步数", type: "number", step: 1, min: 1 },
    lr: { label: "学习率", type: "number", step: 0.00001 },
    min_lr: { label: "最小学习率", type: "number", step: 0.00001 },
    warmup_steps: { label: "Warmup 步数", type: "number", step: 1, min: 0 },
    weight_decay: { label: "Weight decay", type: "number", step: 0.01 },
    beta1: { label: "Adam β1", type: "number", step: 0.01 },
    beta2: { label: "Adam β2", type: "number", step: 0.01 },
    grad_clip: { label: "梯度裁剪", type: "number", step: 0.1, min: 0 },
    eval_interval: { label: "评估间隔", type: "number", step: 1, min: 1 },
    eval_iters: { label: "评估 batch 数", type: "number", step: 1, min: 1 },
    log_interval: { label: "日志间隔", type: "number", step: 1, min: 1 },
    save_interval: { label: "保存间隔", type: "number", step: 1, min: 1 },
    patience: { label: "早停耐心", type: "number", step: 1, min: 0, help: "0=关闭早停；按评估次数计" },
    min_delta: { label: "早停最小改善", type: "number", step: 0.0001 },
    dataset: {
      label: "数据集",
      type: "select",
      options: ["shakespeare_en", "zh_gutenberg", "input"],
      help: "input 时需填输入路径",
    },
    input_path: { label: "本地文本路径", help: "dataset=input 时使用" },
    vocab_extra: { label: "补词表文件", help: "SFT train 文本（.jsonl/.txt），逗号分隔；禁止 val/test" },
    download_dir: { label: "下载缓存目录" },
    val_fraction: { label: "验证集比例", type: "number", step: 0.01, min: 0, max: 0.99 },
    test_fraction: { label: "测试集比例", type: "number", step: 0.01, min: 0, max: 0.99 },
    zh_books: { label: "Gutenberg 书号", help: "逗号分隔，如 23950" },
    source_url: { label: "自定义语料 URL" },
    sft_input: { label: "SFT 输入文件", help: "prepare_sft.py 的 --input" },
    sft_source: { label: "SFT 来源", help: "synthetic_en / synthetic_zh / alpaca_zh / jsonl" },
    max_samples: { label: "最大采样条数", type: "number", step: 1, min: 1 },
    tokenizer_dir: { label: "Tokenizer 目录", help: "SFT 数据使用的预训练 tokenizer 目录" },
    init_from: { label: "预训练 checkpoint", help: "SFT 从该 best.pt 初始化" },
    pretrain_data_dir: { label: "预训练混入目录", help: "SFT 15% pretrain replay 的数据目录" },
    pretrain_mix_ratio: { label: "预训练混合比例", type: "number", step: 0.01, min: 0, max: 1 },
    sample_prompt: { label: "采样提示词", help: "预训练评估时生成样本的 prompt" },
    temperature: { label: "Temperature", type: "number", step: 0.1, min: 0 },
    top_k: { label: "Top-k", type: "number", step: 1, min: 0 },
    top_p: { label: "Top-p", type: "number", step: 0.01, min: 0, max: 1 },
    max_new_tokens: { label: "最大生成 token", type: "number", step: 1, min: 0 },
    resume: { label: "恢复训练", type: "bool", help: "需要已有 run_name 和 last.pt" },
    mini_run: { label: "仅跑 mini-run", type: "bool" },
    gate: { label: "先跑门禁再训练", type: "bool" },
    backend_type: { label: "训练后端", type: "select", options: ["torch", "pure"], help: "pure 为 Stage 2 预留" },
    tokenizer_type: { label: "Tokenizer 类型", type: "select", options: ["char"] },
    model_type: { label: "模型类型", type: "select", options: ["gpt"] },
    callbacks: { label: "回调", help: "逗号分隔，如 telemetry,sample" },
    plugins: { label: "插件路径", help: "逗号分隔的本地 Python 插件" },
    num_threads: { label: "CPU 线程数", type: "number", step: 1, min: 0, help: "0=自动" },
    eval_seed: { label: "评估种子", type: "number", step: 1 },
    web_host: { label: "Web host" },
    web_port: { label: "Web port", type: "number", step: 1, min: 1 },
    max_jobs: { label: "最大活动任务", type: "number", step: 1, min: 1 },
    web_dir: { label: "Web 静态目录" },
    project_root: { label: "项目根目录", help: "留空自动探测" },
  };
/* eslint-enable no-var */
// FIELD_META 块结束（上方两空格缩进的 `};` 为静态测试解析标记）。

/* ------------------------------------------------------------------ 小工具 */
const fmtNumber = fmtSmart;
const isNum = (value) => typeof value === "number" && Number.isFinite(value);

/**
 * 格式化 step/max 文本。
 */
function fmtStep(step, maxSteps) {
  if (step === undefined || step === null || step === "") return "—";
  if (maxSteps === undefined || maxSteps === null || maxSteps === "") return String(step);
  return `${step} / ${maxSteps}`;
}

/**
 * 把长路径压缩为末两段，避免卡片溢出。
 */
function shortenPath(value) {
  if (!value) return "—";
  const parts = String(value).split(/[\\/]+/).filter(Boolean);
  if (parts.length <= 2) return String(value);
  return `…/${parts.slice(-2).join("/")}`;
}

/**
 * 把参数量格式化为 0.81M 这样的短标签。
 */
function formatParams(value) {
  if (!isNum(value)) return "—";
  if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)}K`;
  return String(value);
}

/**
 * 按运行状态生成徽标。
 */
function statusBadge(status) {
  let key = String(status || "unknown");
  if (!STATUS_LABEL[key]) key = "unknown";
  return el("span", `st st-${key}`, STATUS_LABEL[key]);
}

/**
 * 在指定区域显示成功/错误反馈。
 */
function showFeedback(id, message, isError) {
  const node = byId(id);
  if (!node) return;
  node.textContent = message || "";
  node.className = `feedback${isError ? " error" : " ok"}`;
  if (message) showToast(isError ? "error" : "ok", message, "", { duration: isError ? 8000 : 3500 });
}

/**
 * 清除指定区域的反馈。
 */
function clearFeedback(id) {
  const node = byId(id);
  if (node) {
    node.textContent = "";
    node.className = "feedback";
  }
}

/** 请求序号：同一 key 只认最新一次；旧响应（含晚到）一律丢弃。 */
const requestSeq = new Map();
/**
 * 为某个请求类别分配递增序号（竞态守卫）。
 */
function nextSeq(key) {
  const seq = (requestSeq.get(key) || 0) + 1;
  requestSeq.set(key, seq);
  return seq;
}
/**
 * 判断响应是否仍属于最新一次请求。
 */
function isLatestSeq(key, seq) {
  return requestSeq.get(key) === seq;
}

/* ------------------------------------------------------------------ 子模块 */
const charts = createTrainCharts();

const configForm = createConfigForm({
  fieldGroups: FIELD_GROUPS,
  fieldMeta: FIELD_META,
  get editorConfig() { return state.editorConfig; },
  set editorConfig(value) { state.editorConfig = value; },
  get presets() { return state.presets; },
  get runs() { return state.runs; },
  get stage() { return state.stage; },
  get initAlignRun() { return state.initAlignRun; },
  set initAlignRun(value) { state.initAlignRun = value; },
  editorGeneration: () => state.editorGeneration,
  bumpEditorGeneration: () => { state.editorGeneration += 1; },
  requestInitAlign: (runId) => getJson(endpoints.runSummary(runId)),
  feedback: showFeedback,
});

/* 编辑器入口包装：实现在 train/config-form.js，这里保持工作区级别的调用名。 */
function setEditorMode(mode) { configForm.setEditorMode(mode); }
function renderConfigForm() { configForm.renderConfigForm(); }
function onFormChanged(event) { configForm.onFormChanged(event); }

/* ------------------------------------------------------------------ 失败入口 */
function runErrorText(summary) {
  if (!summary) return "";
  const status = summary.status || {};
  const job = summary.job || {};
  const report = summary.report || {};
  return String(status.error || job.error || report.error || "");
}

/**
 * 取多行文本的第一行。
 */
function firstLine(text) {
  const line = String(text || "").split(/\r?\n/)[0];
  return line.length > 120 ? `${line.slice(0, 120)}…` : line;
}

// 运行列表更新时检测“新失败”的状态跳变，弹一次带入口的常驻 toast。
function updateRunsList(runs) {
  const previous = {};
  (state.runs || []).forEach((run) => { previous[run.run_id] = run; });
  const seen = new Set();
  (runs || []).forEach((run) => {
    seen.add(run.run_id);
    const status = run.display_status || run.status || "";
    const prev = previous[run.run_id];
    const prevStatus = prev ? prev.display_status || prev.status || "" : "";
    if (status === "failed") {
      if (prev && prevStatus !== "failed" && !state.failNotified[run.run_id]) {
        state.failNotified[run.run_id] = true;
        showToast("error", `运行失败：${run.run_id}`, firstLine(run.error) || "任务进程异常退出。", {
          actionLabel: "查看失败原因",
          duration: 0,
          onAction: () => openFailureDialog(run.run_id),
        });
      }
    } else {
      delete state.failNotified[run.run_id];
    }
  });
  // 已删除运行的通知记录一并清理。
  Object.keys(state.failNotified).forEach((runId) => {
    if (!seen.has(runId)) delete state.failNotified[runId];
  });
  state.runs = runs || [];
  fillRunSelect();
  renderGlobalFailureEntry();
}

/**
 * 判断是否为内部/临时 run（不展示给用户）。
 */
function isInternalRun(run) {
  return String(run.run_id || "").charAt(0) === "_";
}

/**
 * 筛出失败或中断的 run。
 */
function failedRuns() {
  return (state.runs || []).filter((run) => !isInternalRun(run) && (run.display_status || run.status || "") === "failed");
}

/**
 * 返回最近一次失败的 run。
 */
function latestFailedRun() {
  const failed = failedRuns();
  failed.sort((a, b) => String(b.updated_at || "").localeCompare(String(a.updated_at || "")));
  return failed[0] || null;
}

let globalFailSig = "";
/**
 * 在全局位置渲染失败入口。
 */
function renderGlobalFailureEntry() {
  const button = byId("global-failure-btn");
  const count = byId("global-failure-count");
  if (!button) return;
  const failed = failedRuns();
  const latest = latestFailedRun();
  const sig = `${failed.length}|${latest ? latest.run_id : ""}`;
  if (sig === globalFailSig) return;
  globalFailSig = sig;
  button.hidden = failed.length === 0;
  if (count) count.textContent = String(failed.length);
  button.title = latest
    ? `最近失败：${latest.run_id} · ${firstLine(latest.error) || "查看失败原因"}`
    : "查看最近一次运行失败的原因";
}

/**
 * 渲染失败横幅（含打开详情按钮）。
 */
function renderFailureBanner() {
  const banner = byId("failure-banner");
  if (!banner) return;
  const summary = state.lastSummary;
  const display = summary ? summary.display_status || (summary.status || {}).status || "" : "";
  if (!summary || display !== "failed" || state.failureDismissedFor === summary.run_id) {
    banner.hidden = true;
    return;
  }
  const text = byId("failure-banner-text");
  if (text) {
    const err = runErrorText(summary);
    text.textContent = `${summary.run_id} 失败：${firstLine(err) || "进程异常退出，未记录错误文本"}`;
    text.title = err || "";
  }
  banner.hidden = false;
}

let failureDialogSeq = 0;
/**
 * 打开失败详情对话框（日志/错误/重试）。
 */
function openFailureDialog(runId) {
  const dialog = byId("failure-dialog");
  const target = runId || state.currentRun;
  if (!dialog || !target) return;
  const seq = ++failureDialogSeq;
  function fill(summary) {
    // 连开两个运行时，旧响应不得覆盖新详情。
    if (seq !== failureDialogSeq) return;
    const meta = byId("failure-dialog-meta");
    const errorNode = byId("failure-dialog-error");
    const consoleNode = byId("failure-dialog-console");
    const status = (summary && summary.status) || {};
    const job = (summary && summary.job) || {};
    if (meta) {
      meta.textContent = [
        `run: ${target}`,
        `阶段: ${status.stage || ((summary && summary.config) || {}).stage || "?"}`,
        `退出码: ${job.returncode !== undefined && job.returncode !== null ? job.returncode : "—"}`,
        `结束: ${fmtTime(job.ended_at || status.updated_at)}`,
      ].join("  ·  ");
    }
    const err = runErrorText(summary);
    if (errorNode) {
      errorNode.textContent = err || `（后端未记录错误文本；请查看控制台尾部或 runs/${target}/ 下的日志）`;
    }
    if (consoleNode) consoleNode.textContent = (summary && summary.console_log) || "（暂无输出）";
    dialog.dataset.run = target;
    if (!dialog.open) dialog.showModal();
  }
  Promise.all([
    getJson(endpoints.runSummary(target)),
    getJson(endpoints.runConsole(target, 200)).catch(() => null),
  ])
    .then((results) => {
      const summary = results[0] || {};
      if (results[1] && results[1].console_log) summary.console_log = results[1].console_log;
      fill(summary);
    })
    .catch((error) => {
      if (seq === failureDialogSeq) showToast("error", "无法读取失败详情", error.message);
    });
}

/* ------------------------------------------------------------------ 阶段过滤与下拉 */
function presetsForStage() {
  return state.presetNames.filter((name) => {
    const preset = state.presets[name] || {};
    return String(preset.stage || "pretrain").toLowerCase() === state.stage;
  });
}

/**
 * 返回当前阶段可展示的 run。
 */
function runsForStage() {
  return state.runs.filter((run) => {
    if (isInternalRun(run)) return false;
    const runStage = String(run.stage || "").toLowerCase();
    // 没有 status.json 的旧运行默认归到预训练，避免在 SFT 页误列。
    if (!runStage) return state.stage === "pretrain";
    return runStage === state.stage;
  });
}

/**
 * 按当前阶段刷新页面元素。
 */
function updateStageUi() {
  const tag = byId("stage-tag");
  if (tag) tag.textContent = STAGE_LABEL[state.stage] || state.stage;
  const runSelect = byId("run-select");
  if (runSelect) {
    runSelect.setAttribute("title", `${STAGE_LABEL[state.stage] || state.stage} 阶段运行：下拉切换查看该阶段的 run`);
  }
}

/**
 * 填充预设下拉框。
 */
function fillPresetSelect() {
  const select = byId("preset-select");
  if (!select) return;
  let names = presetsForStage();
  if (!names.length) names = state.presetNames.slice();
  let preferred = STAGE_PRESET_DEFAULT[state.stage] || names[0] || "";
  if (names.indexOf(preferred) < 0) preferred = names[0] || "";
  const items = names.map((name) => ({ value: name, label: name }));
  syncSelectOptions(select, items, `preset|${state.stage}|${names.join(",")}`, preferred);
}

/**
 * 填充配置下拉框。
 */
function fillConfigSelect() {
  const select = byId("config-select");
  if (!select) return;
  let items;
  let signature;
  let preferred;
  if (!state.configs.length) {
    items = [{ value: "", label: "（configs/ 为空）", disabled: true }];
    signature = "config|empty";
    preferred = "";
  } else {
    items = state.configs.map((item) => ({ value: item.name, label: `${item.name}  ·  ${fmtTime(item.updated_at)}` }));
    signature = `config|${state.configs.map((item) => `${item.name}@${item.updated_at || ""}`).join(",")}`;
    preferred = state.configs[0].name;
  }
  syncSelectOptions(select, items, signature, preferred);
}

/**
 * 填充 run 下拉框。
 */
function fillRunSelect() {
  const select = byId("run-select");
  if (!select) return;
  const runs = runsForStage().slice(0, MAX_RUN_OPTIONS);
  let items;
  let signature;
  if (!runs.length) {
    items = [{ value: "", label: `（${STAGE_LABEL[state.stage] || state.stage} 阶段暂无运行）`, disabled: true }];
    signature = `run|${state.stage}|empty`;
  } else {
    items = runs.map((run) => {
      const status = run.display_status || run.status || "unknown";
      let label = `${run.run_id} · ${STATUS_LABEL[status] || status}`;
      if (run.step !== undefined && run.step !== null) label += ` · ${fmtStep(run.step, run.max_steps)}`;
      if (run.project_id) label += ` · ${run.project_id}`;
      return { value: run.run_id, label };
    });
    signature = `run|${state.stage}|${runs.map((run) => `${run.run_id}:${run.display_status || run.status || ""}:${run.step === undefined || run.step === null ? "" : run.step}`).join(",")}`;
  }
  syncSelectOptions(select, items, signature, state.currentRun || "");
}

/* ------------------------------------------------------------------ 项目归属 */
let projectSig = "";
/**
 * 刷新项目下拉框。
 */
function refreshProjectSelect() {
  const select = byId("project-select");
  if (!select) return;
  getJson(endpoints.projects())
    .then((payload) => {
      const items = payload.projects || [];
      const sig = items.map((item) => `${item.project_id}:${item.name}`).join("|");
      if (sig === projectSig) return;
      projectSig = sig;
      const previous = select.value;
      select.replaceChildren();
      const none = el("option", null, "（不归属项目）");
      none.value = "";
      select.appendChild(none);
      items.forEach((item) => {
        const option = el("option", null, item.name || item.project_id);
        option.value = item.project_id;
        select.appendChild(option);
      });
      const create = el("option", null, "＋ 新建项目…");
      create.value = "__new__";
      select.appendChild(create);
      if ([...select.options].some((option) => option.value === previous)) select.value = previous;
    })
    .catch(() => {});
}

/**
 * 绑定项目切换事件。
 */
function bindProjectSelect() {
  const select = byId("project-select");
  if (!select) return;
  select.addEventListener("change", () => {
    if (select.value !== "__new__") return;
    select.value = "";
    const name = window.prompt("新项目名称（例如：英文莎士比亚流水线）");
    if (!name || !name.trim()) return;
    postJson(endpoints.projects(), { name: name.trim() })
      .then((data) => {
        projectSig = "";
        refreshProjectSelect();
        const project = data.project || {};
        if (project.project_id) {
          window.setTimeout(() => { select.value = project.project_id; }, 300);
        }
        showToast("ok", "项目已创建", data.message || "");
      })
      .catch((error) => showToast("error", "创建项目失败", error.message));
  });
}

/* ------------------------------------------------------------------ 运行切换 */
function currentActiveJob() {
  const active = state.jobs.filter((job) => job.active);
  const matching = active.filter((job) => job.run_id === state.currentRun);
  return matching.length ? matching[0] : active.length ? active[0] : null;
}

/**
 * 从 URL 查询参数读取 run id。
 */
function queryRunId() {
  try {
    return new URLSearchParams(window.location.search).get("run");
  } catch (error) {
    return null;
  }
}

/**
 * 切换到用户选择的 run。
 */
function chooseRun() {
  const runs = runsForStage();
  const requested = queryRunId();
  if (requested && runs.some((run) => run.run_id === requested)) {
    setCurrentRun(requested);
    return;
  }
  if (runs.some((run) => run.run_id === state.currentRun)) return;
  const activeJob = state.jobs.filter((job) => job.active)[0];
  const preferred = activeJob ? runs.filter((run) => run.run_id === activeJob.run_id)[0] : null;
  const target = preferred || runs[0] || null;
  setCurrentRun(target ? target.run_id : null);
}

/**
 * 清空与 run 相关的展示区域。
 */
function resetRunViews() {
  charts.reset();
  state.lastSummary = null;
  state.lastMetrics = null;
  state.lastMetricsRow = null;
  renderMetricStrip(null);
  const failBanner = byId("failure-banner");
  if (failBanner) failBanner.hidden = true;
  updateRunCard();
  renderStatusPanel(null);
  renderGatePanel(null);
  renderDataPanel(null);
  state.events = [];
  state.eventIndex = new Set();
  state.eventFollow = true;
  renderEvents();
  renderSamples(null);
  const consoleNode = byId("console-tail");
  if (consoleNode) consoleNode.textContent = "（暂无输出）";
  charts.setAllOverlays("loading", "加载曲线…");
  updateDeleteRunButton();
}

/**
 * 设置当前 run 并同步 URL/状态。
 */
function setCurrentRun(runId) {
  if (state.currentRun === runId) return;
  state.currentRun = runId;
  state.lastEventAfter = 0;
  state.failureDismissedFor = null;
  nextSeq("run"); // 作废所有在途的运行详情响应
  try {
    const nextUrl = new URL(window.location.href);
    if (runId) nextUrl.searchParams.set("run", runId);
    else nextUrl.searchParams.delete("run");
    window.history.replaceState(null, "", nextUrl.pathname + nextUrl.search + nextUrl.hash);
  } catch (error) {
    /* file:// 或受限浏览器：深链只是便利，不是必需 */
  }
  resetRunViews();
  const select = byId("run-select");
  if (select) {
    for (let i = 0; i < select.options.length; i += 1) {
      if (select.options[i].value === runId) {
        select.selectedIndex = i;
        break;
      }
    }
  }
  if (runId) loadRun();
}

/* ------------------------------------------------------------------ 数据加载 */
/** 运行详情：四个请求各自独立降级；序号守卫保证切换后旧响应不落地。 */
function loadRun() {
  const runId = state.currentRun;
  if (!runId) {
    resetRunViews();
    return Promise.resolve();
  }
  const seq = nextSeq("run");
  const summaryP = getJson(endpoints.runSummary(runId)).catch(() => null);
  const metricsP = getJson(endpoints.runMetrics(runId)).catch(() => null);
  const eventsP = getJson(endpoints.runEvents(runId, state.lastEventAfter, 500)).catch(() => null);
  const samplesP = getJson(endpoints.runSamples(runId, 6)).catch(() => null);
  return Promise.all([summaryP, metricsP, eventsP, samplesP]).then((results) => {
    if (!isLatestSeq("run", seq) || state.currentRun !== runId) return;
    // 每个渲染步骤独立隔离：任何一个面板抛错都不影响其余面板。
    function safeRender(fn) {
      try {
        fn();
      } catch (err) {
        if (window.console && console.error) console.error("面板渲染异常:", err);
      }
    }
    if (results[0]) {
      state.lastSummary = results[0];
      safeRender(() => applySummary(results[0]));
    }
    if (results[2]) safeRender(() => applyEvents(results[2]));
    if (results[1]) {
      state.lastMetrics = results[1];
      const metricRows = results[1].rows || [];
      state.lastMetricsRow = metricRows.length ? metricRows[metricRows.length - 1] : null;
    }
    safeRender(() => renderMetricStrip(results[0] || state.lastSummary));
    const chartSource = results[1] || state.lastMetrics;
    const summaryForCharts = results[0] || state.lastSummary;
    safeRender(() => {
      if (chartSource || (summaryForCharts && summaryForCharts.metrics_count > 0)) {
        charts.update(chartSource, summaryForCharts, state.events);
      } else if (!results[0] && !results[1]) {
        charts.setAllOverlays("error", "曲线数据加载失败");
      } else {
        charts.update(null, summaryForCharts, state.events);
      }
    });
    if (results[3]) safeRender(() => renderSamples(results[3]));
    updateStaleBadge();
  });
}

/**
 * 把 run 摘要应用到各展示区。
 */
function applySummary(summary) {
  renderStatusPanel(summary);
  renderGatePanel(summary);
  renderDataPanel(summary);
  const consoleNode = byId("console-tail");
  if (consoleNode) consoleNode.textContent = (summary && summary.console_log) || "（暂无输出）";
  updateDeleteRunButton();
  updateRunCard();
  renderFailureBanner();
}

/* ------------------------------------------------------------------ 运行卡状态机 */
function activeJobForCurrentRun() {
  if (!state.currentRun) return null;
  return state.jobs.filter((job) => job.active && job.run_id === state.currentRun)[0] || null;
}

/**
 * 返回当前 run 的状态名。
 */
function currentStatusName() {
  const summary = state.lastSummary;
  if (!summary) return "";
  return summary.display_status || (summary.status || {}).status || "";
}

/**
 * 当前 run 是否仍在运行。
 */
function currentRunIsActive() {
  if (!state.currentRun) return false;
  if (activeJobForCurrentRun()) return true;
  return ["running", "starting", "stopping"].indexOf(currentStatusName()) >= 0;
}

// 卡住的任务（status=running 但心跳过期）允许删除，但不允许再次启动，避免双进程写同一目录。
function canDeleteCurrentRun() {
  if (!state.currentRun || activeJobForCurrentRun()) return false;
  const status = currentStatusName();
  if (status === "running" || status === "starting" || status === "stopping") return isRunStale();
  return true;
}

/**
 * 判断当前 run 是否可以启动/续训。
 */
function canStartCurrentRun() {
  if (!state.currentRun || activeJobForCurrentRun()) return false;
  return ["running", "starting", "stopping"].indexOf(currentStatusName()) < 0;
}

/**
 * 判断状态文件是否长时间未更新。
 */
function isRunStale() {
  const summary = state.lastSummary;
  if (!summary) return false;
  const display = summary.display_status || (summary.status || {}).status;
  if (["running", "starting", "stopping"].indexOf(display) < 0) return false;
  const updated = (summary.status || {}).updated_at;
  if (!updated) return false;
  const timestamp = Date.parse(updated);
  if (isNaN(timestamp)) return false;
  return Date.now() - timestamp > RUN_STALE_MS;
}

/**
 * 按状态启用/禁用删除按钮。
 */
function updateDeleteRunButton() {
  const button = byId("delete-run-btn");
  if (!button) return;
  button.disabled = !canDeleteCurrentRun();
  button.title = isRunStale()
    ? "该任务心跳已过期（可能已卡住）；可强制删除 runs/ 下的文件"
    : "删除选中的、已结束的任务（会删除 runs/ 下的文件，不可恢复）";
}

/**
 * 返回当前 run 的阶段（pretrain/sft）。
 */
function currentRunStage() {
  const summary = state.lastSummary;
  const status = (summary && summary.status) || {};
  const config = (summary && summary.config) || {};
  const stage = String(status.stage || config.stage || "").toLowerCase();
  if (stage === "sft" || stage === "pretrain") return stage;
  const run = (state.runs || []).filter((item) => item.run_id === state.currentRun)[0];
  return String((run && run.stage) || state.stage || "pretrain").toLowerCase();
}

// SFT 必须指定预训练 checkpoint（或 resume 续训），否则训练进程必然退出码 1。
function sftConfigProblem(config) {
  if (!config || String(config.stage || "").toLowerCase() !== "sft") return "";
  if (config.resume) return "";
  if (!String(config.init_from || "").trim()) {
    return "SFT 需要预训练 checkpoint：请在配置里选择「从已完成预训练运行选择」或填写 init_from（也可开启 resume 续训）";
  }
  return "";
}

let runCardSig = "";
/**
 * 更新运行卡片（状态、进度、产物）。
 */
function updateRunCard() {
  const statusNode = byId("run-card-status");
  const startBtn = byId("start-btn");
  const stopBtn = byId("stop-btn");
  const nextBtn = byId("next-step-btn");
  const resumeBtn = byId("resume-btn");
  const summary = state.lastSummary;
  const display = summary ? summary.display_status || (summary.status || {}).status || "unknown" : "";

  const mode = (() => {
    if (!summary) return "none";
    if (currentRunIsActive()) return "stop";
    if (display === "completed") return "next";
    if (sftConfigProblem(summary.config)) return "fix";
    if (canStartCurrentRun()) return "start";
    return "none";
  })();
  const resumable = Boolean(summary && !currentRunIsActive()
    && ["completed", "stopped", "failed"].indexOf(display) >= 0
    && summary.has_checkpoint);
  const stale = isRunStale();
  const deletable = canDeleteCurrentRun();
  const sig = `${state.currentRun}|${display}|${mode}|${resumable}|${stale}|${deletable}|${currentRunStage()}|${(summary && summary.project_id) || ""}|${(summary && summary.parent_run_id) || ""}`;
  if (sig === runCardSig) return;
  runCardSig = sig;

  if (statusNode) {
    statusNode.replaceChildren();
    if (summary) statusNode.appendChild(statusBadge(display));
    else statusNode.appendChild(el("span", "muted", state.currentRun ? "加载中…" : "未选择运行"));
  }
  const badgesNode = byId("run-card-badges");
  if (badgesNode) {
    badgesNode.replaceChildren();
    if (summary && summary.project_id) {
      badgesNode.appendChild(badge(`项目 ${summary.project_id}`, "badge-ok"));
    }
    if (summary && summary.parent_run_id) {
      badgesNode.appendChild(badge(`上游 ${summary.parent_run_id}`, "badge-wait"));
    }
  }
  if (startBtn) {
    startBtn.hidden = mode !== "start";
    startBtn.disabled = mode !== "start";
  }
  if (resumeBtn) {
    resumeBtn.hidden = !resumable;
    resumeBtn.disabled = !resumable;
  }
  if (stopBtn) {
    stopBtn.hidden = mode !== "stop";
    stopBtn.disabled = mode !== "stop";
  }
  if (nextBtn) {
    if (mode === "next") {
      const stage = currentRunStage();
      nextBtn.textContent = stage === "pretrain" ? "基于此运行做 SFT" : "去生成可视化";
      nextBtn.title = stage === "pretrain"
        ? "跳到 SFT 配置，并自动填入该运行的 checkpoint 与对齐字段"
        : "跳到生成可视化页，使用该运行的 checkpoint";
      nextBtn.hidden = false;
      nextBtn.disabled = false;
    } else if (mode === "fix") {
      nextBtn.textContent = "去配置 checkpoint";
      nextBtn.title = sftConfigProblem(summary.config);
      nextBtn.hidden = false;
      nextBtn.disabled = false;
    } else {
      nextBtn.hidden = true;
      nextBtn.disabled = true;
    }
  }
  updateDeleteRunButton();
  renderLifecycle();
}

// 生命周期条：创建 → 启动 → 训练 → 完成/失败。签名不变不重建（每秒轮询不夺焦点）。
let lifecycleSig = "";
/**
 * 渲染训练生命周期时间线。
 */
function renderLifecycle() {
  const box = byId("run-lifecycle");
  if (!box) return;
  const summary = state.lastSummary;
  const display = summary ? summary.display_status || (summary.status || {}).status || "unknown" : "";
  const step = Number((summary && summary.status || {}).step);
  const hasProgress = Number.isFinite(step) && step > 0;
  const failed = display === "failed";
  const completed = display === "completed";
  const active = currentRunIsActive();
  const canStart = canStartCurrentRun();
  const sig = `${state.currentRun}|${display}|${hasProgress}|${active}|${canStart}`;
  if (sig === lifecycleSig) return;
  lifecycleSig = sig;
  box.replaceChildren();
  if (!summary) {
    box.appendChild(el("span", "muted", state.currentRun ? "加载中…" : "选择运行后显示生命周期"));
    return;
  }
  const nodes = [
    { key: "created", label: "创建" },
    { key: "start", label: "启动" },
    { key: "train", label: "训练" },
    { key: "end", label: failed ? "失败" : "完成" },
  ];
  function nodeState(key) {
    if (key === "created") return "done";
    if (key === "start") {
      if (active || hasProgress || completed || failed) return "done";
      if (canStart) return "actionable";
      return "current";
    }
    if (key === "train") {
      if (completed || failed) return "done";
      if (active) return "current";
      return "pending";
    }
    if (failed) return "failed";
    if (completed) return "current";
    return "pending";
  }
  nodes.forEach((node, index) => {
    if (index > 0) box.appendChild(el("span", "lifecycle-link"));
    const stateName = nodeState(node.key);
    const isButton = stateName === "actionable" || stateName === "failed";
    const item = el(isButton ? "button" : "span", `lifecycle-node ${stateName}`);
    if (isButton) item.type = "button";
    item.appendChild(el("span", "lifecycle-dot"));
    item.appendChild(el("span", null, node.label));
    if (node.key === "start" && stateName === "actionable") {
      item.title = "启动该任务";
      item.addEventListener("click", () => startSelectedRun());
    } else if (node.key === "end" && stateName === "failed") {
      item.title = "查看失败原因";
      item.addEventListener("click", () => openFailureDialog(state.currentRun));
    } else if (node.key === "end" && stateName === "current") {
      item.title = "任务已完成";
    }
    box.appendChild(item);
  });
}

// 完成后的下一步：预训练 → SFT 配置（自动对齐）；SFT → 生成可视化。
function goNextStep() {
  const runId = state.currentRun;
  if (!runId) return;
  if (state.lastSummary && sftConfigProblem(state.lastSummary.config)) {
    setSubView("config");
    configForm.setEditorFromConfig(state.lastSummary.config);
    showFeedback("config-feedback", sftConfigProblem(state.lastSummary.config), true);
    return;
  }
  const stage = currentRunStage();
  if (stage === "pretrain") {
    state.pendingSftFrom = runId;
    window.location.hash = "#/sft";
  } else {
    window.location.hash = "#/generate";
  }
}

/**
 * 消费「从该 run 发起 SFT」的一次性状态。
 */
function consumePendingSftFrom() {
  const runId = state.pendingSftFrom;
  if (!runId || state.stage !== "sft") return;
  state.pendingSftFrom = null;
  setSubView("config");
  configForm.applyInitFromRun(runId);
}

/**
 * 请求删除当前 run（含确认）。
 */
function deleteRun() {
  const runId = state.currentRun;
  if (!runId) {
    showFeedback("control-feedback", "先选一个运行再删除", true);
    return;
  }
  if (!canDeleteCurrentRun()) {
    showFeedback("control-feedback", "该运行仍在更新心跳；先停止，或等心跳过期后再强制删除", true);
    return;
  }
  const stale = isRunStale();
  const message =
    (stale ? "检测到该任务心跳已过期（可能已卡住），将按强制删除处理。\n" : "") +
    `会删除 runs/${runId}/ 下的全部文件（checkpoint、指标、事件、样本），不可恢复。\n` +
    "输入运行名以确认删除（区分大小写）：";
  promptDialog({
    title: `删除运行 ${runId}`,
    message,
    danger: true,
    okText: "删除",
    placeholder: runId,
    validate: (value) => {
      if (value === null) return "";
      return value === runId ? "" : `名字不匹配，请原样输入：${runId}`;
    },
  }).then((typed) => {
    if (typed !== runId) {
      if (typed !== null) showFeedback("control-feedback", "已取消：输入的名字不匹配", true);
      return null;
    }
    return deleteJson(endpoints.runDelete(runId, stale))
      .then((data) => {
        showFeedback("control-feedback", (data && data.message) || "已删除", false);
        // 走统一切换入口：清 URL 深链、作废在途请求、重置视图。
        setCurrentRun(null);
        return refreshLists();
      })
      .catch((error) => {
        showFeedback("control-feedback", `删除失败: ${error.message}`, true);
      })
      .finally(updateDeleteRunButton);
  });
}

/* ------------------------------------------------------------------ 轮询（单一循环） */
function fetchLists() {
  return Promise.all([
    getJson(endpoints.health()),
    getJson(endpoints.jobs()),
    getJson(endpoints.runs()),
  ]).then((results) => {
    state.lastUpdateOk = Date.now();
    state.lastListFetch = Date.now();
    state.consecutiveFails = 0;
    state.offline = false;
    updateHealth(results[0]);
    updateJobs(results[1]);
    updateRunsList(results[2].runs || []);
    chooseRun();
    refreshProjectSelect();
    const lastUpdate = byId("last-update");
    if (lastUpdate) lastUpdate.textContent = `更新于 ${new Date().toLocaleTimeString()}`;
    // 启动时失败的端点在这里自动重试，不再一败俱败。
    if (!state.systemLoaded) {
      getJson(endpoints.system()).then((system) => {
        state.systemLoaded = true;
        updateSystem(system);
      }).catch(() => {});
    }
    if (!state.presetsLoaded) {
      getJson(endpoints.presets()).then((payload) => {
        state.presets = payload.presets || {};
        state.presetNames = payload.names || Object.keys(state.presets);
        state.presetsLoaded = true;
        fillPresetSelect();
        if (!Object.keys(state.editorConfig).length) {
          const presetSelect = byId("preset-select");
          if (presetSelect && presetSelect.value) configForm.setEditorFromPreset(presetSelect.value);
        }
      }).catch(() => {});
    }
    if (!state.configsLoaded) {
      getJson(endpoints.configs()).then((payload) => {
        state.configs = payload.configs || [];
        state.configsLoaded = true;
        fillConfigSelect();
      }).catch(() => {});
    }
  });
}

let listsInFlight = false;
/**
 * 刷新所有列表数据。
 */
function refreshLists() {
  if (listsInFlight) return Promise.resolve();
  listsInFlight = true;
  return fetchLists()
    .catch(showConnectionError)
    .finally(() => { listsInFlight = false; });
}

let runLoadInFlight = false;
/**
 * 主轮询节拍：拉取状态、事件、指标与样本。
 */
function tick() {
  if (store.paused || document.hidden) {
    updateStaleBadge();
    return;
  }
  // 抽屉在任何页签都可打开，需要保持任务状态新鲜。
  const needLists =
    state.activeView === "train" ||
    state.drawerOpen ||
    Date.now() - state.lastListFetch > LIST_POLL_MS;
  if (needLists && !listsInFlight) {
    listsInFlight = true;
    fetchLists()
      .then(() => {
        if (state.drawerOpen) refreshDrawerConsole();
        // 只有训练工作区可见时才拉取运行详情，其他页签不做重渲染。
        if (state.currentRun && state.activeView === "train" && !runLoadInFlight) {
          runLoadInFlight = true;
          return loadRun().finally(() => { runLoadInFlight = false; });
        }
        return null;
      })
      .catch(showConnectionError)
      .finally(() => {
        listsInFlight = false;
        updateStaleBadge();
      });
    return;
  }
  updateStaleBadge();
}

/**
 * 显示后端连接异常提示。
 */
function showConnectionError(error) {
  state.consecutiveFails += 1;
  const lastUpdate = byId("last-update");
  // 单次请求失败不立刻判离线：连续多次失败才切换健康徽章，避免假离线闪烁。
  if (state.consecutiveFails >= OFFLINE_FAIL_THRESHOLD) {
    state.offline = true;
    const badgeNode = byId("health-badge");
    if (badgeNode) {
      badgeNode.textContent = "离线";
      badgeNode.className = "badge badge-bad";
    }
    if (lastUpdate) lastUpdate.textContent = `刷新失败: ${error && error.message ? error.message : error}`;
  } else if (lastUpdate) {
    lastUpdate.textContent = `刷新重试中（${state.consecutiveFails}）…`;
  }
  updateStaleBadge();
}

/* ------------------------------------------------------------------ 顶栏 */
function updateHealth(health) {
  const badgeNode = byId("health-badge");
  if (!badgeNode) return;
  if (health && health.status === "ok") {
    badgeNode.textContent = `在线 · v${health.version || "?"} · ${health.active_jobs || 0} 任务`;
    badgeNode.className = "badge badge-ok";
  } else {
    badgeNode.textContent = "未知状态";
    badgeNode.className = "badge badge-warn";
  }
}

/**
 * 更新系统信息区。
 */
function updateSystem(system) {
  const node = byId("sys-summary");
  if (!node || !system) return;
  const torch = system.torch_available
    ? `torch ${system.torch_version || "已安装"}`
    : "未安装 torch（看板模式）";
  node.textContent = `Python ${system.python || "?"} · ${torch} · ${system.cpu_count || "?"} 线程`;
}

/**
 * 更新「数据陈旧」标记。
 */
function updateStaleBadge() {
  const badgeNode = byId("stale-badge");
  if (!badgeNode) return;
  if (store.paused) {
    badgeNode.className = "badge badge-warn";
    badgeNode.textContent = "已暂停刷新";
    return;
  }
  const connectionStale =
    state.offline ||
    (state.lastUpdateOk > 0 && Date.now() - state.lastUpdateOk > CONNECTION_STALE_MS);
  if (connectionStale) {
    badgeNode.className = "badge badge-bad";
    badgeNode.textContent = "连接中断";
    return;
  }
  if (isRunStale()) {
    badgeNode.className = "badge badge-warn";
    badgeNode.textContent = "训练心跳过期";
    return;
  }
  badgeNode.className = "badge badge-hidden";
  badgeNode.textContent = "";
}

/* ------------------------------------------------------------------ 任务与抽屉 */
let jobsSig = "";
/**
 * 更新任务列表与抽屉。
 */
function updateJobs(payload) {
  if (!payload || !payload.jobs) return;
  state.jobs = payload.jobs;
  const active = state.jobs.filter((job) => job.active);
  const box = byId("active-jobs");
  const sig = active.map((job) => `${job.run_id}:${job.status}:${job.pid}`).join("|");
  if (box && sig !== jobsSig) {
    jobsSig = sig;
    box.replaceChildren();
    if (!active.length) {
      box.appendChild(el("p", "muted", "没有正在运行的任务"));
    } else {
      active.forEach((job) => {
        const row = el("div", "job-row");
        row.appendChild(statusBadge(job.status));
        row.appendChild(el("span", "job-id", job.run_id));
        row.appendChild(el("span", "muted", `pid ${job.pid || "?"}`));
        const stop = el("button", "btn btn-danger", "停止");
        stop.type = "button";
        stop.setAttribute("data-job-id", job.run_id);
        row.appendChild(stop);
        box.appendChild(row);
      });
    }
  }
  updateRunCard();
  renderTaskDrawer(active);
}

let drawerSig = "";
/**
 * 渲染任务抽屉内容。
 */
function renderTaskDrawer(activeJobs) {
  const active = activeJobs || state.jobs.filter((job) => job.active);
  const count = byId("task-count");
  if (count) {
    count.textContent = String(active.length);
    count.className = `badge ${active.length ? "badge-warn" : "badge-ok"}`;
  }
  const body = byId("task-drawer-body");
  if (!body) return;
  // 抽屉关闭时不重建 DOM；打开时按签名差量更新，避免夺走按钮焦点。
  if (!state.drawerOpen) return;
  const sig = state.jobs.map((job) => `${job.run_id}:${job.status}:${job.pid}:${job.step}:${job.active}`).join("|");
  if (sig === drawerSig) return;
  drawerSig = sig;
  body.replaceChildren();
  if (!state.jobs.length) {
    body.appendChild(el("p", "muted", "本服务还没有启动过任务。"));
    return;
  }
  state.jobs.forEach((job) => {
    const row = el("div", `task-drawer-row${job.active ? " active" : ""}`);
    const head = el("div", "task-drawer-row-head");
    head.appendChild(statusBadge(job.status));
    head.appendChild(el("span", "task-drawer-run", job.run_id));
    const stage = String(job.stage || "");
    if (stage) head.appendChild(el("span", "badge badge-wait", STAGE_LABEL[stage] || stage));
    row.appendChild(head);
    const meta = [`pid ${job.pid || "?"}`, `elapsed ${fmtNumber(job.elapsed_s, 1)}s`];
    if (job.step !== undefined && job.step !== null) meta.push(`step ${fmtStep(job.step, job.max_steps)}`);
    row.appendChild(el("div", "task-drawer-row-meta", meta.join(" · ")));
    const actions = el("div", "task-drawer-row-actions");
    if (job.active) {
      const stop = el("button", "btn btn-danger btn-small", "停止");
      stop.type = "button";
      stop.setAttribute("data-job-id", job.run_id);
      actions.appendChild(stop);
    }
    const open = el("button", "btn btn-small", "查看运行");
    open.type = "button";
    open.setAttribute("data-open-run", job.run_id);
    open.setAttribute("data-open-stage", stage || state.stage);
    actions.appendChild(open);
    if (job.status === "failed") {
      const reason = el("button", "btn btn-small btn-ghost-danger", "查看原因");
      reason.type = "button";
      reason.setAttribute("data-failure-run", job.run_id);
      actions.appendChild(reason);
    }
    row.appendChild(actions);
    body.appendChild(row);
  });
}

// 抽屉焦点陷阱：Tab 只在抽屉内循环，返回解绑函数。
function trapDrawerFocus(container) {
  const selector =
    'a[href], button:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';
  function onKeydown(event) {
    if (event.key !== "Tab") return;
    const nodes = Array.prototype.filter.call(container.querySelectorAll(selector), (node) => node.offsetParent !== null || node === document.activeElement);
    if (!nodes.length) return;
    const first = nodes[0];
    const last = nodes[nodes.length - 1];
    if (event.shiftKey && document.activeElement === first) {
      event.preventDefault();
      last.focus();
    } else if (!event.shiftKey && document.activeElement === last) {
      event.preventDefault();
      first.focus();
    }
  }
  container.addEventListener("keydown", onKeydown);
  return () => container.removeEventListener("keydown", onKeydown);
}

/**
 * 打开任务抽屉。
 */
function openTaskDrawer() {
  const drawer = byId("task-drawer");
  if (!drawer) return;
  state.drawerOpen = true;
  state.drawerReturnFocus = document.activeElement;
  drawer.hidden = false;
  const scrim = byId("drawer-scrim");
  if (scrim) scrim.hidden = false;
  state.releaseDrawerTrap = trapDrawerFocus(drawer);
  drawerSig = "";
  renderTaskDrawer();
  refreshDrawerConsole();
  const closeButton = byId("task-drawer-close");
  if (closeButton) closeButton.focus();
}

/**
 * 关闭任务抽屉。
 */
function closeTaskDrawer() {
  const drawer = byId("task-drawer");
  if (!drawer || !state.drawerOpen) return;
  state.drawerOpen = false;
  drawer.hidden = true;
  const scrim = byId("drawer-scrim");
  if (scrim) scrim.hidden = true;
  if (state.releaseDrawerTrap) {
    state.releaseDrawerTrap();
    state.releaseDrawerTrap = null;
  }
  const back = state.drawerReturnFocus;
  state.drawerReturnFocus = null;
  if (back && typeof back.focus === "function" && document.contains(back)) back.focus();
}

/**
 * 切换任务抽屉开合。
 */
function toggleTaskDrawer() {
  if (state.drawerOpen) closeTaskDrawer();
  else openTaskDrawer();
}

let drawerConsoleSeq = 0;
/**
 * 刷新抽屉里的控制台尾部日志。
 */
function refreshDrawerConsole() {
  const box = byId("task-drawer-console");
  const job = currentActiveJob() || state.jobs[0];
  const runId = job ? job.run_id : state.currentRun;
  if (!box) return;
  if (!runId) {
    box.textContent = "（没有任务运行）";
    return;
  }
  const seq = ++drawerConsoleSeq;
  getJson(endpoints.runConsole(runId, 40))
    .then((payload) => {
      // 抽屉可能已关闭或任务已切换：过期响应不写入。
      if (seq !== drawerConsoleSeq || !state.drawerOpen) return;
      box.textContent = (payload && payload.console_log) || "（暂无输出）";
      box.setAttribute("data-run", runId);
    })
    .catch(() => {
      if (seq === drawerConsoleSeq && state.drawerOpen) box.textContent = "（控制台读取失败）";
    });
}

/* ------------------------------------------------------------------ 详情面板（签名差量） */
function renderKv(container, rows) {
  const list = el("div", "kv");
  let count = 0;
  rows.forEach(([key, value]) => {
    if (value === undefined || value === null || value === "") return;
    count += 1;
    const line = el("div", "kv-row");
    line.appendChild(el("span", "kv-key", key));
    const valueWrap = el("span", "kv-value");
    if (value instanceof Node) valueWrap.appendChild(value);
    else valueWrap.textContent = String(value);
    line.appendChild(valueWrap);
    list.appendChild(line);
  });
  if (!count) list.appendChild(el("p", "muted", "（无数据）"));
  container.appendChild(list);
}

/**
 * 渲染运行状态面板。
 */
function renderStatusPanel(summary) {
  const container = byId("status-panel");
  if (!container) return;
  if (!summary) {
    container.dataset.sig = "";
    container.replaceChildren(el("p", "muted", "（选择一个运行）"));
    return;
  }
  const status = summary.status || {};
  const job = summary.job || {};
  const display = summary.display_status || status.status || "unknown";
  const sig = JSON.stringify([display, summary.run_id, status.step, status.best_val, status.pid, status.updated_at, status.error || job.error || ""]);
  renderSigned(container, sig, () => {
    const rows = [];
    rows.push(["状态", statusBadge(display)]);
    rows.push(["运行", summary.run_id]);
    rows.push(["阶段", status.stage || (summary.config || {}).stage]);
    rows.push(["步数", fmtStep(status.step, status.max_steps || (summary.config || {}).max_steps)]);
    rows.push(["最佳 val", fmtNumber(status.best_val !== undefined && status.best_val !== null ? status.best_val : (summary.report || {}).best_val_loss, 4)]);
    rows.push(["PID", status.pid || job.pid]);
    rows.push(["开始", fmtTime(status.started_at || job.started_at)]);
    rows.push(["更新", fmtTime(status.updated_at || summary.updated_at)]);
    rows.push(["checkpoint", status.checkpoint || (summary.report || {}).checkpoint]);
    const errorText = status.error || job.error || (summary.report || {}).error;
    if (errorText) rows.push(["错误", el("span", "error-text", errorText)]);
    const box = el("div");
    renderKv(box, rows);
    return Array.from(box.childNodes);
  });
}

/**
 * 把事件 payload 摘要成一行文本。
 */
function summarisePayload(payload) {
  if (!payload || typeof payload !== "object") return "";
  const keys = [
    "train_loss", "val_loss", "val_ppl", "best_val", "lr", "grad_norm", "tokens_per_sec",
    "initial_loss", "loss_drop", "passed", "status", "kind", "path", "message", "prompt", "text",
  ];
  const parts = [];
  keys.forEach((key) => {
    const value = payload[key];
    if (value === undefined || value === null || value === "") return;
    if (typeof value === "number") parts.push(`${key}=${fmtNumber(value, 4)}`);
    else if (typeof value === "boolean") parts.push(`${key}=${value ? "true" : "false"}`);
    else if (typeof value === "string" && value.length <= 70) parts.push(`${key}=${value.replace(/\s+/g, " ")}`);
  });
  if (payload.payload && typeof payload.payload === "object") {
    const nested = summarisePayload(payload.payload);
    if (nested) parts.push(nested);
  }
  return parts.join("  ");
}

/**
 * 渲染 mini-run/gate 门槛结果面板。
 */
function renderGatePanel(summary) {
  const container = byId("gate-panel");
  if (!container) return;
  const sig = summary
    ? JSON.stringify([summary.minirun_report || null, (summary.events || []).filter((e) => e.event === "minirun").slice(-4), summary.early_stop || [], summary.report && summary.report.status])
    : "empty";
  renderSigned(container, sig, () => {
    const box = el("div");
    if (!summary) {
      box.appendChild(el("p", "muted", "（本运行未产生 mini-run 门禁或早停事件）"));
      return Array.from(box.childNodes);
    }
    const report = summary.minirun_report;
    if (report) {
      const header = el("div", "chips");
      header.appendChild(el("span", `chip ${report.passed ? "chip-ok" : "chip-bad"}`, report.passed ? "mini-run PASS" : "mini-run FAIL"));
      header.appendChild(el("span", "chip", `run: ${String(summary.minirun_run_id || "?")}`));
      box.appendChild(header);
      const rows = [
        ["初始 loss", fmtNumber(report.initial_loss, 4)],
        ["期望初始 loss", fmtNumber(report.expected_initial_loss, 4)],
        ["loss 偏差", fmtNumber(report.initial_loss_abs_diff, 4)],
        ["梯度正常", report.gradient_ok === undefined ? "—" : report.gradient_ok ? "是" : "否"],
        ["loss 下降", fmtNumber(report.loss_drop, 4)],
        ["要求下降", fmtNumber(report.min_drop, 4)],
      ];
      const kvBox = el("div", "kv");
      kvBox.style.marginTop = "9px";
      renderKv(kvBox, rows);
      box.appendChild(kvBox);
      return Array.from(box.childNodes);
    }
    const miniEvents = (summary.events || []).filter((event) => event.event === "minirun");
    const earlyStop = summary.early_stop || [];
    if (miniEvents.length) {
      miniEvents.slice(-4).forEach((event) => {
        const payload = event.payload || {};
        const node = el("div", "job-row");
        node.appendChild(el("span", `chip ${payload.passed ? "chip-ok" : "chip-bad"}`, payload.passed ? "PASS" : "FAIL"));
        node.appendChild(el("span", "job-id", `step ${event.step !== undefined ? event.step : "?"}  ${summarisePayload(payload)}`));
        box.appendChild(node);
      });
      return Array.from(box.childNodes);
    }
    if (earlyStop.length) {
      earlyStop.forEach((event) => {
        const payload = event.payload || {};
        const node = el("div", "job-row");
        node.appendChild(el("span", "chip chip-bad", "早停"));
        node.appendChild(el("span", "job-id", `step ${event.step !== undefined ? event.step : "?"}  ${summarisePayload(payload)}`));
        box.appendChild(node);
      });
      return Array.from(box.childNodes);
    }
    const reportNode = summary.report;
    if (reportNode && reportNode.status) {
      const summaryRow = el("div", "kv");
      renderKv(summaryRow, [
        ["训练报告", reportNode.status],
        ["早停", reportNode.stopped_early === undefined ? "—" : reportNode.stopped_early ? "是" : "否"],
        ["patience", reportNode.patience],
      ]);
      box.appendChild(summaryRow);
      return Array.from(box.childNodes);
    }
    box.appendChild(el("p", "muted", "（本运行未产生 mini-run 门禁或早停事件）"));
    return Array.from(box.childNodes);
  });
}

/**
 * 渲染数据详情面板。
 */
function renderDataPanel(summary) {
  const container = byId("data-panel");
  if (!container) return;
  const sig = summary ? JSON.stringify([summary.config || {}, summary.environment || {}, summary.files || {}]) : "empty";
  renderSigned(container, sig, () => {
    const box = el("div");
    if (!summary) {
      box.appendChild(el("p", "muted", "（无数据）"));
      return Array.from(box.childNodes);
    }
    const config = summary.config || {};
    const environment = summary.environment || {};
    const rows = [
      ["数据集", config.dataset],
      ["数据目录", config.data_dir ? shortenPath(config.data_dir) : null],
      ["模型", config.n_layer && config.n_head && config.n_embd ? `${config.n_layer} 层 / ${config.n_head} 头 / ${config.n_embd} 维` : null],
      ["block / batch", config.block_size ? `${config.block_size} / ${config.batch_size || "?"} × ${config.grad_accum_steps || 1}` : null],
      ["步数上限", config.max_steps],
      ["学习率", config.lr === undefined ? null : fmtNumber(config.lr, 6)],
      ["设备 / dtype", [config.device, config.dtype].filter(Boolean).join(" / ")],
      ["后端", config.backend_type],
      ["随机种子", config.seed],
      ["Python", environment.python_version || environment.python],
      ["PyTorch", environment.torch_version || environment.torch],
      ["参数数量", formatParams(environment.parameters)],
      ["平台", environment.platform],
      ["训练解释器", shortenPath(environment.python_executable || environment.executable)],
    ];
    renderKv(box, rows);
    const files = summary.files || {};
    const chips = el("div", "chips");
    Object.keys(files).forEach((name) => {
      chips.appendChild(el("span", `chip ${files[name] ? "chip-on" : "chip-off"}`, name));
    });
    box.appendChild(chips);
    return Array.from(box.childNodes);
  });
}

/* ------------------------------------------------------------------ 事件 / 样本 */
/** 事件合并：按 index 去重（并发/重放安全），游标只前进不回退。 */
function applyEvents(payload) {
  if (!payload) return;
  if (typeof payload.next_after === "number") {
    state.lastEventAfter = Math.max(state.lastEventAfter, payload.next_after);
  }
  const fresh = (payload.events || []).filter((event) => {
    const key = event.index !== undefined ? event.index : `${event.ts}|${event.event}|${event.step}`;
    if (state.eventIndex.has(key)) return false;
    state.eventIndex.add(key);
    return true;
  });
  if (fresh.length) {
    state.events = state.events.concat(fresh).slice(-EVENT_KEEP);
    // 索引集合同步裁剪，防止无限增长。
    if (state.eventIndex.size > EVENT_KEEP * 2) {
      state.eventIndex = new Set(state.events.map((event) => event.index !== undefined ? event.index : `${event.ts}|${event.event}|${event.step}`));
    }
    renderEvents();
  }
}

/**
 * 渲染事件列表。
 */
function renderEvents() {
  const container = byId("event-panel");
  if (!container) return;
  // 只有用户本来就停留在底部时才自动跟随；向上翻阅历史时不再被强行拉回。
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 48;
  const follow = state.eventFollow && nearBottom;
  container.replaceChildren();
  const events = state.events.slice(-MAX_EVENT_ROWS);
  if (!events.length) {
    container.appendChild(el("p", "muted", "（暂无事件）"));
    updateEventJumpButton();
    return;
  }
  const frag = document.createDocumentFragment();
  events.forEach((event) => {
    const row = el("div", "event-row");
    row.appendChild(el("span", "event-step", `step ${event.step !== undefined ? event.step : "?"}`));
    const name = String(event.event || "unknown").replace(/[^a-zA-Z_]/g, "");
    row.appendChild(el("span", `event-name event-${name}`, event.event || "?"));
    row.appendChild(el("span", "event-summary", summarisePayload(event.payload)));
    frag.appendChild(row);
  });
  container.appendChild(frag);
  if (follow) container.scrollTop = container.scrollHeight;
  updateEventJumpButton();
}

/**
 * 按滚动位置更新「跳到最新」按钮。
 */
function updateEventJumpButton() {
  const container = byId("event-panel");
  const button = byId("event-jump-btn");
  if (!container || !button) return;
  const hasEvents = state.events.length > 0;
  const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 48;
  button.hidden = !hasEvents || nearBottom;
}

/**
 * 滚动事件列表到底部。
 */
function jumpEventsToBottom() {
  const container = byId("event-panel");
  if (!container) return;
  state.eventFollow = true;
  container.scrollTop = container.scrollHeight;
  updateEventJumpButton();
}

let samplesSig = "";
/**
 * 渲染训练样本卡片。
 */
function renderSamples(payload) {
  const container = byId("samples");
  if (!container) return;
  const samples = (payload && payload.samples) || [];
  const sig = samples.map((sample) => `${sample.step}|${sample.ts}`).join("|");
  if (sig === samplesSig) return;
  samplesSig = sig;
  container.replaceChildren();
  if (!samples.length) {
    container.appendChild(el("p", "muted", "（等待训练写入 samples.jsonl…）"));
    return;
  }
  samples.slice().reverse().forEach((sample) => {
    const card = el("div", "sample");
    const head = el("div", "sample-head");
    head.appendChild(el("span", "sample-step", `step ${sample.step !== undefined ? sample.step : "?"}`));
    head.appendChild(el("span", null, fmtTime(sample.ts)));
    card.appendChild(head);
    if (sample.prompt) card.appendChild(el("div", "sample-prompt", `Q: ${sample.prompt}`));
    card.appendChild(el("pre", "sample-text",
      sample.text !== undefined ? sample.text
        : sample.sample !== undefined ? sample.sample
        : JSON.stringify(sample, null, 2)));
    container.appendChild(card);
  });
}

/* ------------------------------------------------------------------ 动作 */
function createRun() {
  clearFeedback("control-feedback");
  const payload = { preset: byId("preset-select").value || "debug_en", stage: state.stage };
  const runNameInput = byId("run-name-input");
  const runName = runNameInput ? runNameInput.value.trim() : "";
  if (runName) payload.run_name = runName;
  try {
    payload.config = configForm.parseEditor();
  } catch (error) {
    showFeedback("control-feedback", error.message, true);
    return;
  }
  const projectSelect = byId("project-select");
  const projectId = projectSelect ? projectSelect.value : "";
  if (projectId && projectId !== "__new__") payload.config.project_id = projectId;
  const problem = sftConfigProblem(payload.config);
  if (problem) {
    showFeedback("config-feedback", problem, true);
    showToast("error", "无法创建 SFT 任务", problem, { duration: 8000 });
    return;
  }
  const button = byId("create-run-btn");
  if (button) button.disabled = true;
  postJson(endpoints.runs(), payload)
    .then((data) => {
      const job = data.job || {};
      showFeedback("control-feedback", data.message || "任务已创建（未启动）", false);
      if (runNameInput) runNameInput.value = "";
      setSubView("run");
      return refreshLists().then(() => {
        if (job.run_id) setCurrentRun(job.run_id);
      });
    })
    .catch((error) => {
      showFeedback("control-feedback", `创建失败: ${error.message}`, true);
    })
    .finally(() => {
      if (button) button.disabled = false;
    });
}

/**
 * 启动当前选中的 run。
 */
function startSelectedRun() {
  clearFeedback("control-feedback");
  const runId = state.currentRun;
  if (!runId) {
    showFeedback("control-feedback", "先选一个任务（或先点“新建任务”）", true);
    return;
  }
  if (currentRunIsActive()) {
    showFeedback("control-feedback", "该任务已在运行", true);
    return;
  }
  const problem = state.lastSummary ? sftConfigProblem(state.lastSummary.config) : "";
  if (problem) {
    showFeedback("control-feedback", problem, true);
    showToast("error", "无法启动 SFT 任务", problem, {
      actionLabel: "去配置 checkpoint",
      duration: 8000,
      onAction: goNextStep,
    });
    return;
  }
  const button = byId("start-btn");
  if (button) button.disabled = true;
  postJson(endpoints.runStart(runId), {})
    .then((data) => {
      showFeedback("control-feedback", data.message || "任务已启动", false);
      return refreshLists();
    })
    .catch((error) => {
      showFeedback("control-feedback", `启动失败: ${error.message}`, true);
    })
    .finally(updateRunCard);
}

/**
 * 从 last checkpoint 续训当前 run。
 */
function resumeSelectedRun() {
  clearFeedback("control-feedback");
  const runId = state.currentRun;
  if (!runId) {
    showFeedback("control-feedback", "先选一个任务", true);
    return;
  }
  if (currentRunIsActive()) {
    showFeedback("control-feedback", "该任务已在运行", true);
    return;
  }
  const button = byId("resume-btn");
  if (button) button.disabled = true;
  postJson(endpoints.runResume(runId), {})
    .then((data) => {
      showFeedback("control-feedback", data.message || "已从检查点续训", false);
      return refreshLists();
    })
    .catch((error) => {
      showFeedback("control-feedback", `续训失败: ${error.message}`, true);
    })
    .finally(updateRunCard);
}

/**
 * 停止指定任务。
 */
function stopJob(jobId) {
  let targetId = jobId;
  if (!targetId) {
    const job = currentActiveJob();
    targetId = job ? job.run_id : null;
  }
  if (!targetId) {
    showFeedback("control-feedback", "没有可停止的活动任务", true);
    return;
  }
  clearFeedback("control-feedback");
  postJson(endpoints.jobStop(targetId), {})
    .then((data) => {
      showFeedback("control-feedback", data.message || "已请求停止", false);
      return refreshLists();
    })
    .catch((error) => {
      showFeedback("control-feedback", `停止失败: ${error.message}`, true);
    });
}

/**
 * 在线校验当前配置。
 */
function validateConfig() {
  clearFeedback("config-feedback");
  let config;
  try {
    config = configForm.parseEditor();
  } catch (error) {
    showFeedback("config-feedback", error.message, true);
    return;
  }
  const button = byId("validate-btn");
  if (button) button.disabled = true;
  postJson(endpoints.configValidate(), { preset: byId("preset-select").value || "debug_en", config })
    .then((data) => {
      if (data.valid) {
        const resolved = data.resolved || {};
        showFeedback("config-feedback", `配置有效 ✓${resolved.runs_dir ? `  runs_dir=${resolved.runs_dir}` : ""}`, false);
      } else {
        showFeedback("config-feedback", `配置无效: ${(data.errors || []).join("; ")}`, true);
      }
    })
    .catch((error) => {
      showFeedback("config-feedback", `校验失败: ${error.message}`, true);
    })
    .finally(() => {
      if (button) button.disabled = false;
    });
}

/**
 * 刷新配置文件列表。
 */
function refreshConfigs() {
  return getJson(endpoints.configs()).then((data) => {
    state.configs = data.configs || [];
    state.configsLoaded = true;
    fillConfigSelect();
  });
}

/**
 * 保存当前配置。
 */
function saveConfig() {
  clearFeedback("config-feedback");
  let config;
  try {
    config = configForm.parseEditor();
  } catch (error) {
    showFeedback("config-feedback", error.message, true);
    return;
  }
  const defaultName = `config-${new Date().toISOString().slice(0, 10)}`;
  promptDialog({
    title: "保存配置",
    message: "配置名称只能包含字母、数字、点、下划线、连字符。",
    defaultValue: defaultName,
    okText: "保存",
    validate: (value) => {
      if (!value || !value.trim()) return "名称不能为空";
      return /^[A-Za-z0-9._-]+$/.test(value.trim()) ? "" : "名称含非法字符";
    },
  }).then((name) => {
    if (!name) return;
    const trimmed = name.trim();
    postJson(endpoints.configs(), { name: trimmed, config })
      .then((data) => {
        showFeedback("config-feedback", `已保存配置: ${data.name || trimmed}`, false);
        return refreshConfigs();
      })
      .catch((error) => {
        showFeedback("config-feedback", `保存失败: ${error.message}`, true);
      });
  });
}

/**
 * 加载选中配置到编辑器。
 */
function loadConfig() {
  clearFeedback("config-feedback");
  const select = byId("config-select");
  const name = select ? select.value : "";
  if (!name) {
    showFeedback("config-feedback", "configs/ 里还没有配置", true);
    return;
  }
  getJson(endpoints.config(name))
    .then((data) => {
      configForm.setEditorFromConfig(data.config || data);
      showFeedback("config-feedback", `已载入配置: ${name}`, false);
    })
    .catch((error) => {
      showFeedback("config-feedback", `载入失败: ${error.message}`, true);
    });
}

/**
 * 删除选中配置。
 */
function deleteConfig() {
  clearFeedback("config-feedback");
  const select = byId("config-select");
  const name = select ? select.value : "";
  if (!name) {
    showFeedback("config-feedback", "没有可删除的配置", true);
    return;
  }
  confirmDialog({
    title: `删除配置 ${name}`,
    message: "只删除 configs/ 里的配置文件，不影响任何已启动的运行。",
    danger: true,
    okText: "删除",
  }).then((confirmed) => {
    if (!confirmed) return;
    deleteJson(endpoints.config(name))
      .then((data) => {
        showFeedback("config-feedback", (data && data.message) || `已删除配置: ${name}`, false);
        return refreshConfigs();
      })
      .catch((error) => {
        showFeedback("config-feedback", `删除失败: ${error.message}`, true);
      });
  });
}

/* ------------------------------------------------------------------ 子页与指标条 */
function setSubView(sub) {
  state.subView = sub === "config" ? "config" : "run";
  const section = byId("view-train");
  if (section) section.setAttribute("data-sub", state.subView);
  const runArea = section ? section.querySelector(".train-run-area") : null;
  const configArea = section ? section.querySelector(".train-config-area") : null;
  if (runArea) runArea.hidden = state.subView !== "run";
  if (configArea) configArea.hidden = state.subView !== "config";
  const runBtn = byId("sub-run-btn");
  const configBtn = byId("sub-config-btn");
  if (runBtn) runBtn.className = `btn btn-small${state.subView === "run" ? " btn-primary" : ""}`;
  if (configBtn) configBtn.className = `btn btn-small${state.subView === "config" ? " btn-primary" : ""}`;
  const hint = byId("train-sub-hint");
  if (hint) {
    hint.textContent = state.subView === "run"
      ? "按运行卡的主操作继续；新建任务去「配置」子页"
      : "编辑配置并创建任务；创建后自动回到「运行」子页";
  }
  if (state.subView === "run") {
    // 切回运行子页时重画 canvas（隐藏期间 clientWidth 为 0）。
    charts.resize();
    updateEventJumpButton();
  }
}

let metricSig = "";
/**
 * 渲染顶部指标条。
 */
function renderMetricStrip(summary) {
  const box = byId("metric-strip");
  if (!box) return;
  if (!summary) {
    metricSig = "";
    box.replaceChildren(el("span", "muted", "选择一个运行后显示关键指标"));
    return;
  }
  const status = summary.status || {};
  const row = state.lastMetricsRow || summary.metrics || {};
  const best = status.best_val !== undefined && status.best_val !== null ? status.best_val : (summary.report || {}).best_val_loss;
  const tps = row.tokens_per_sec !== undefined ? row.tokens_per_sec : status.tokens_per_sec;
  const values = [
    ["步数", fmtStep(status.step, status.max_steps || (summary.config || {}).max_steps)],
    ["best val", fmtNumber(best, 4)],
    ["train loss", fmtNumber(row.train_loss !== undefined ? row.train_loss : status.train_loss, 4)],
    ["lr", fmtNumber(row.lr !== undefined ? row.lr : status.lr, 6)],
    ["吞吐", isNum(tps) ? `${fmtInt(tps)} tok/s` : "—"],
    ["耗时", fmtElapsed(row.elapsed_s !== undefined ? row.elapsed_s : status.elapsed_s)],
  ];
  const sig = values.map((pair) => pair.join("=")).join("|");
  if (sig === metricSig) return;
  metricSig = sig;
  box.replaceChildren();
  values.forEach(([label, value]) => {
    const item = el("div", "metric");
    item.appendChild(el("span", "metric-label", label));
    item.appendChild(el("span", "metric-value", value));
    box.appendChild(item);
  });
}

const MONITOR_TABS = ["events", "samples", "console", "details"];
/**
 * 切换监控区标签页。
 */
function setMonitorTab(tab) {
  state.monitorTab = MONITOR_TABS.indexOf(tab) >= 0 ? tab : "events";
  document.querySelectorAll(".monitor-tab").forEach((node) => {
    const active = node.getAttribute("data-monitor") === state.monitorTab;
    node.classList.toggle("active", active);
    node.setAttribute("aria-selected", active ? "true" : "false");
    node.tabIndex = active ? 0 : -1;
  });
  MONITOR_TABS.forEach((name) => {
    const pane = byId(`monitor-${name}`);
    if (pane) pane.hidden = name !== state.monitorTab;
  });
  if (state.monitorTab === "events") updateEventJumpButton();
}

/* ------------------------------------------------------------------ 阶段切换 */
function applyStage(stage) {
  const next = stage === "sft" ? "sft" : "pretrain";
  const changed = state.stage !== next;
  state.stage = next;
  updateStageUi();
  if (!state.stageReady) {
    // 首次进入：路由事件可能早于初始加载，等 bootstrap 统一填表。
    state.stageReady = true;
    return;
  }
  if (!changed) return;
  state.editorGeneration += 1;
  fillPresetSelect();
  const presetSelect = byId("preset-select");
  if (presetSelect && presetSelect.value) configForm.setEditorFromPreset(presetSelect.value);
  fillRunSelect();
  chooseRun();
  if (state.currentRun) loadRun();
  else resetRunViews();
  consumePendingSftFrom();
}

window.addEventListener("spa:view", (event) => {
  const detail = (event && event.detail) || {};
  state.activeView = detail.view || "train";
  if (detail.view !== "train") return;
  applyStage(detail.stage);
  // 回到训练页签时立即拉一次最新数据。
  state.lastListFetch = 0;
  if (!store.paused) tick();
});

/* ------------------------------------------------------------------ 事件绑定 */
function bindEvents() {
  configForm.bind();
  const presetSelect = byId("preset-select");
  if (presetSelect) {
    presetSelect.addEventListener("change", () => {
      state.editorGeneration += 1;
      configForm.setEditorFromPreset(presetSelect.value);
    });
  }
  const runSelect = byId("run-select");
  if (runSelect) runSelect.addEventListener("change", () => setCurrentRun(runSelect.value || null));
  const createButton = byId("create-run-btn");
  if (createButton) createButton.addEventListener("click", createRun);
  bindProjectSelect();
  refreshProjectSelect();
  const startButton = byId("start-btn");
  if (startButton) startButton.addEventListener("click", startSelectedRun);
  const resumeButton = byId("resume-btn");
  if (resumeButton) resumeButton.addEventListener("click", resumeSelectedRun);
  const stopButton = byId("stop-btn");
  if (stopButton) stopButton.addEventListener("click", () => stopJob(null));
  const refreshButton = byId("refresh-btn");
  if (refreshButton) {
    refreshButton.addEventListener("click", () => {
      if (store.paused) setPaused(false);
      state.lastListFetch = 0;
      tick();
    });
  }
  const validateButton = byId("validate-btn");
  if (validateButton) validateButton.addEventListener("click", validateConfig);
  const saveButton = byId("save-config-btn");
  if (saveButton) saveButton.addEventListener("click", saveConfig);
  const loadButton = byId("load-config-btn");
  if (loadButton) loadButton.addEventListener("click", loadConfig);
  const deleteConfigButton = byId("delete-config-btn");
  if (deleteConfigButton) deleteConfigButton.addEventListener("click", deleteConfig);
  const deleteRunButton = byId("delete-run-btn");
  if (deleteRunButton) deleteRunButton.addEventListener("click", deleteRun);
  const activeJobs = byId("active-jobs");
  if (activeJobs) {
    activeJobs.addEventListener("click", (event) => {
      const button = event.target.closest ? event.target.closest("button[data-job-id]") : null;
      if (button) stopJob(button.getAttribute("data-job-id"));
    });
  }
  const nextStepBtn = byId("next-step-btn");
  if (nextStepBtn) nextStepBtn.addEventListener("click", goNextStep);
  const globalFailureBtn = byId("global-failure-btn");
  if (globalFailureBtn) {
    globalFailureBtn.addEventListener("click", () => {
      const latest = latestFailedRun();
      if (latest) openFailureDialog(latest.run_id);
    });
  }
  const failureDetailBtn = byId("failure-detail-btn");
  if (failureDetailBtn) failureDetailBtn.addEventListener("click", () => openFailureDialog(state.currentRun));
  const failureDismissBtn = byId("failure-dismiss-btn");
  if (failureDismissBtn) {
    failureDismissBtn.addEventListener("click", () => {
      state.failureDismissedFor = state.currentRun;
      renderFailureBanner();
    });
  }
  const failureCloseBtn = byId("failure-close-btn");
  const failureDialog = byId("failure-dialog");
  if (failureCloseBtn && failureDialog) {
    failureCloseBtn.addEventListener("click", () => {
      if (typeof failureDialog.close === "function") failureDialog.close();
    });
  }
  const failureCopyBtn = byId("failure-copy-btn");
  if (failureCopyBtn) {
    failureCopyBtn.addEventListener("click", () => {
      const errorNode = byId("failure-dialog-error");
      const consoleNode = byId("failure-dialog-console");
      const text = `${errorNode ? errorNode.textContent : ""}\n\n${consoleNode ? consoleNode.textContent : ""}`;
      copyText(text).then((ok) => {
        showToast(ok ? "ok" : "error", ok ? "已复制错误信息" : "复制失败，请手动选择文本");
      });
    });
  }
  const failureConsoleTabBtn = byId("failure-console-tab-btn");
  if (failureConsoleTabBtn) {
    failureConsoleTabBtn.addEventListener("click", () => {
      if (failureDialog && typeof failureDialog.close === "function") failureDialog.close();
      setSubView("run");
      setMonitorTab("console");
    });
  }
  const subRunBtn = byId("sub-run-btn");
  if (subRunBtn) subRunBtn.addEventListener("click", () => setSubView("run"));
  const subConfigBtn = byId("sub-config-btn");
  if (subConfigBtn) subConfigBtn.addEventListener("click", () => setSubView("config"));

  // 监控页签：点击 + 方向键（tablist 键盘语义）。
  const monitorTablist = document.querySelector(".monitor-tabs");
  document.querySelectorAll(".monitor-tab").forEach((node) => {
    node.addEventListener("click", () => setMonitorTab(node.getAttribute("data-monitor")));
  });
  if (monitorTablist) {
    monitorTablist.addEventListener("keydown", (event) => {
      if (event.key !== "ArrowRight" && event.key !== "ArrowLeft" && event.key !== "Home" && event.key !== "End") return;
      const tabs = Array.from(monitorTablist.querySelectorAll(".monitor-tab"));
      const current = tabs.findIndex((node) => node.getAttribute("data-monitor") === state.monitorTab);
      let next = current;
      if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
      else if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
      else if (event.key === "Home") next = 0;
      else next = tabs.length - 1;
      event.preventDefault();
      setMonitorTab(tabs[next].getAttribute("data-monitor"));
      tabs[next].focus();
    });
  }

  const drawerButton = byId("task-drawer-btn");
  if (drawerButton) drawerButton.addEventListener("click", toggleTaskDrawer);
  const drawerClose = byId("task-drawer-close");
  if (drawerClose) drawerClose.addEventListener("click", closeTaskDrawer);
  const drawerScrim = byId("drawer-scrim");
  if (drawerScrim) drawerScrim.addEventListener("click", closeTaskDrawer);
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && state.drawerOpen) closeTaskDrawer();
  });
  const eventJumpButton = byId("event-jump-btn");
  if (eventJumpButton) eventJumpButton.addEventListener("click", jumpEventsToBottom);
  const eventPanel = byId("event-panel");
  if (eventPanel) {
    eventPanel.addEventListener("scroll", () => {
      const nearBottom = eventPanel.scrollHeight - eventPanel.scrollTop - eventPanel.clientHeight < 48;
      state.eventFollow = nearBottom;
      updateEventJumpButton();
    });
  }
  // 下拉框失焦后补上轮询期间暂缓的选项更新。
  bindSelectFlush(["run-select", "preset-select", "config-select"]);
  // 页面从隐藏恢复可见时立即刷新一次，不用等下一个轮询周期。
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && !store.paused) {
      state.lastListFetch = 0;
      tick();
    }
  });
  const drawerBody = byId("task-drawer-body");
  if (drawerBody) {
    drawerBody.addEventListener("click", (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      const stop = target.closest("button[data-job-id]");
      if (stop) {
        stopJob(stop.getAttribute("data-job-id"));
        return;
      }
      const failure = target.closest("button[data-failure-run]");
      if (failure) {
        openFailureDialog(failure.getAttribute("data-failure-run"));
        return;
      }
      const open = target.closest("button[data-open-run]");
      if (!open) return;
      const runId = open.getAttribute("data-open-run");
      const stage = open.getAttribute("data-open-stage");
      if (stage && stage !== state.stage && (stage === "pretrain" || stage === "sft")) {
        window.location.hash = `#/${stage}`;
      }
      setCurrentRun(runId);
      closeTaskDrawer();
    });
  }
  // 暂停状态由全局 store 驱动（顶栏开关在 store.js 绑定）。
  on("paused", (paused) => {
    if (!paused) {
      state.lastUpdateOk = Date.now();
      tick();
    }
    updateStaleBadge();
  });
}

/* ------------------------------------------------------------------ 启动 */
function bootstrap() {
  state.activeView = store.view || document.body.getAttribute("data-active-view") || "train";
  if (store.stage) state.stage = store.stage;
  charts.init();
  bindEvents();
  updateStageUi();
  setSubView("run");
  setMonitorTab("events");
  renderMetricStrip(null);
  // 各端点独立降级：任何一个失败只影响对应功能，轮询会自动重试。
  getJson(endpoints.health()).then(updateHealth).catch(() => {});
  getJson(endpoints.system()).then((system) => {
    state.systemLoaded = true;
    updateSystem(system);
  }).catch(() => {});
  getJson(endpoints.presets()).then((payload) => {
    state.presets = payload.presets || {};
    state.presetNames = payload.names || Object.keys(state.presets);
    state.presetsLoaded = true;
    fillPresetSelect();
    const presetSelect = byId("preset-select");
    if (presetSelect && presetSelect.value) configForm.setEditorFromPreset(presetSelect.value);
    configForm.setEditorMode("form");
  }).catch(() => {});
  getJson(endpoints.configs()).then((payload) => {
    state.configs = payload.configs || [];
    state.configsLoaded = true;
    fillConfigSelect();
  }).catch(() => {});
  Promise.all([
    getJson(endpoints.runs()).catch(() => ({ runs: [] })),
    getJson(endpoints.jobs()).catch(() => ({})),
  ]).then((results) => {
    state.lastUpdateOk = Date.now();
    updateRunsList((results[0] && results[0].runs) || []);
    updateJobs(results[1] || {});
    state.stageReady = true;
    chooseRun();
    if (state.currentRun) return loadRun();
    return null;
  }).finally(() => {
    updateStaleBadge();
    window.setInterval(tick, POLL_MS);
  });
}

bootstrap();