/* 生成页：逐 token 采样调试器（实时流式）。
 *
 * 数据路径
 * --------
 * 浏览器用 fetch 读取 ``POST /api/v1/inference/stream`` 的 NDJSON 流：
 * ``status -> meta -> step xN -> final -> done``（出错时 ``error``）。
 * 推理实时完成并入缓冲；播放引擎按速度逐 token 揭示（可暂停/单步/重播）。
 *
 * 生命周期约束
 * ------------
 *  - 每个分支持有自己的 AbortController：停止/删除/切换都不留孤儿流；
 *  - 播放状态机显式化：单步到末步同样收尾；停止后不再被残留定时器覆写；
 *  - 揭示走增量 DOM（追加 chip / 输出 token），不随 token 数全量重绘；
 *  - 参数对比可中止，结果绑定请求标识，切换 checkpoint 不会串结果。
 *
 * 只用浏览器原生 JS（fetch / DOM / Canvas），无构建链、无外部网络。
 */

import { initAttentionDemo } from "./attention.js";
import { endpoints, getJson, postJson } from "./api.js";
import { byId, el, showToast, canvasFont, fontTokens, invalidateFontTokens } from "./ui.js";
import { store, on } from "./store.js";

/** 同时保留的分支上限（多了面板放不下，也容易看晕）。 */
const MAX_BRANCHES = 8;
/** 每步向前裁剪的上下文长度，与后端 max_context 默认值一致。 */
const MAX_CONTEXT = 32;
/** 1× 时每个 token 的揭示间隔 */
const BASE_REVEAL_MS = 220;

const state = {
  runs: [],
  branches: [],
  activeId: 0,
  selected: null,
  follow: true,
  running: false,
  layer: 0,
  head: 0,
  seq: 0,
  attentionDemo: null,
  visible: false,
  // 播放引擎：推理实时完成并缓存，UI 按速度逐 token 揭示。
  play: {
    speed: 0.5,
    paused: false,
    revealed: 0,
    timer: null,
    lastReplay: null,
    finished: false,
  },
  // 增量渲染游标
  stripRendered: 0,
  outputRendered: 0,
  outputBranchId: 0,
  stripBranchId: 0,
  pipelineKey: "",
};

const STATUS_LABEL = {
  idle: "未开始",
  running: "生成中",
  done: "完成",
  stopped: "已停止",
  error: "出错",
};

/* ------------------------------------------------------------------ 小工具 */
const element = el;

/**
 * 生成弱化提示文本节点。
 */
function muted(text) {
  return element("p", "muted", text);
}

/**
 * 格式化小数，非法值显示占位符。
 */
function fmt(value, digits = 4) {
  const number = Number(value);
  return Number.isFinite(number) ? number.toFixed(digits) : "?";
}

/** 百分比显示：大的值给 1 位小数，小的值给更多小数，避免全是 0.0%。 */
function pct(value) {
  const number = Number(value);
  if (!Number.isFinite(number) || number <= 0) return "0";
  if (number >= 0.1) return `${(number * 100).toFixed(1)}%`;
  return `${(number * 100).toFixed(number >= 0.01 ? 2 : 3)}%`;
}

/** 把 token 的可读表示还原成真实字符（后端用 Python 风格转义）。 */
function tokenText(raw) {
  const text = raw === undefined || raw === null ? "" : String(raw);
  if (/^\\u[0-9a-fA-F]{4}$/.test(text)) {
    return String.fromCharCode(parseInt(text.slice(2), 16));
  }
  const escapes = { "\\n": "\n", "\\r": "\r", "\\t": "\t", "\\\\": "\\" };
  return Object.prototype.hasOwnProperty.call(escapes, text) ? escapes[text] : text;
}

/** 短标签：空白/控制字符换成可见符号，长的截断。 */
function tokenLabel(raw) {
  const text = tokenText(raw);
  if (text === "") return "∅";
  if (text === " ") return "␠";
  if (text === "\n") return "↵";
  if (text === "\r") return "␍";
  if (text === "\t") return "⇥";
  if (text.startsWith("<") && text.endsWith(">")) return text;
  return text.length > 4 ? `${text.slice(0, 4)}…` : text;
}

/** 概率色带从 CSS 变量取端点色，亮暗主题自动跟随。 */
let probStops = null;
/**
 * 按相对概率生成颜色。
 */
function probColor(relative) {
  if (!probStops) {
    const style = getComputedStyle(document.documentElement);
    const parse = (name, fallback) => {
      const raw = style.getPropertyValue(name).trim() || fallback;
      const m = raw.match(/^#([0-9a-f]{6})$/i);
      if (!m) return fallback;
      const hex = m[1];
      return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
    };
    probStops = [parse("--prob-low", [217, 122, 114]), parse("--prob-mid", [211, 164, 78]), parse("--prob-high", [95, 184, 143])];
  }
  const value = Math.max(0, Math.min(1, Number(relative) || 0));
  const x = value * 2;
  const i = Math.min(1, Math.floor(x));
  const f = x - i;
  const a = probStops[i];
  const b = probStops[i + 1];
  const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}

/** 选中概率相对该步最高概率的比值：越接近 1 说明越接近“最优候选”。 */
function stepRelative(step) {
  const top1 =
    step.sampling && Number.isFinite(Number(step.sampling.top1_prob))
      ? Number(step.sampling.top1_prob)
      : Number(step.prob);
  if (!(top1 > 0)) return 1;
  return Math.max(0, Math.min(1, Number(step.prob) / top1));
}

function clampIndex(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

/* ------------------------------------------------------------------ 状态 */
function activeBranch() {
  return state.branches.find((branch) => branch.id === state.activeId) || null;
}

/** 当前分支已揭示（可见）的步数：播放进度与已缓冲步数取小。 */
function visibleCount(branch) {
  const b = branch || activeBranch();
  if (!b) return 0;
  return Math.min(state.play.revealed, b.steps.length);
}

/**
 * 返回当前选中的生成步骤。
 */
function selectedStep() {
  const branch = activeBranch();
  const visible = visibleCount(branch);
  if (!branch || !visible) return null;
  const wanted = state.selected === null ? visible : Math.min(state.selected, visible);
  const pool = branch.steps.slice(0, visible);
  return pool.find((step) => step.step === wanted) || pool[pool.length - 1];
}

/* ------------------------------------------------------------------ 播放引擎 */
function revealDelay() {
  return BASE_REVEAL_MS / Math.max(0.25, state.play.speed);
}

/**
 * 清除播放定时器。
 */
function clearPlayTimer() {
  if (state.play.timer !== null) {
    window.clearTimeout(state.play.timer);
    state.play.timer = null;
  }
}

/** 按当前播放进度更新所有「可见步」相关视图（增量优先）。 */
function renderRevealed() {
  const branch = activeBranch();
  syncStrip();
  syncOutput();
  renderOutputMeta();
  renderPipeline();
  renderAttention();
  syncBranchProgress();
  syncPlayControls();
  const visible = visibleCount(branch);
  const total = branch ? branch.params.max_new_tokens : 0;
  setProgress(`第 ${visible} / ${total} 步`);
}

/** 调度下一个 token 的揭示；暂停或追平缓冲时停止。 */
function scheduleReveal() {
  clearPlayTimer();
  const branch = activeBranch();
  if (!branch) return;
  if (state.play.paused) {
    syncPlayControls();
    return;
  }
  if (state.play.revealed >= branch.steps.length) {
    // 已追平缓冲：推理完成则收尾，否则等下一步到达后再调度。
    if (branch.final) finishPlayback(branch);
    syncPlayControls();
    return;
  }
  const branchId = branch.id;
  state.play.timer = window.setTimeout(() => {
    state.play.timer = null;
    const b = activeBranch();
    // 定时器触发时分支可能已切换/删除：过期回调直接丢弃。
    if (!b || b.id !== branchId || state.play.paused) return;
    if (state.play.revealed < b.steps.length) {
      revealOne(b);
    }
    scheduleReveal();
  }, revealDelay());
}

/** 揭示一个缓冲中的 token（增量渲染 + 状态文案）。 */
function revealOne(branch) {
  state.play.revealed += 1;
  if (state.follow) state.selected = state.play.revealed;
  renderRevealed();
  const step = branch.steps[state.play.revealed - 1];
  if (branch.status === "running" || !branch.final) {
    setStatus(`生成中：${state.play.revealed} / ${branch.params.max_new_tokens}${step && step.elapsed_ms ? ` · 本步 ${fmt(step.elapsed_ms, 1)} ms` : ""}`);
  }
}

/**
 * 同步播放/暂停按钮状态。
 */
function syncPlayControls() {
  const branch = activeBranch();
  const pauseBtn = byId("gen-play-pause");
  const stepBtn = byId("gen-play-step");
  const replayBtn = byId("gen-play-replay");
  const progress = byId("gen-play-progress");
  const bar = document.querySelector(".gen-playbar");
  const hasSteps = Boolean(branch && branch.steps.length);
  const canReplay = Boolean(state.play.lastReplay);
  if (pauseBtn) {
    pauseBtn.disabled = !hasSteps;
    pauseBtn.textContent = state.play.paused ? "继续" : "暂停";
    pauseBtn.title = state.play.paused ? "继续逐 token 揭示" : "暂停逐 token 揭示";
  }
  if (stepBtn) stepBtn.disabled = !hasSteps || !state.play.paused || state.play.revealed >= (branch ? branch.steps.length : 0);
  if (replayBtn) replayBtn.disabled = !canReplay || state.running;
  if (progress) {
    const visible = visibleCount(branch);
    const buffered = branch ? branch.steps.length : 0;
    progress.textContent = `已揭示 ${visible} / ${buffered}`;
  }
  if (bar) bar.classList.toggle("playing", hasSteps && !state.play.paused && state.play.revealed < (branch ? branch.steps.length : 0));
}

/**
 * 切换播放/暂停。
 */
function togglePlayPause() {
  const branch = activeBranch();
  if (!branch || !branch.steps.length) return;
  state.play.paused = !state.play.paused;
  if (!state.play.paused) {
    scheduleReveal();
  } else {
    clearPlayTimer();
  }
  syncPlayControls();
}

/**
 * 手动前进一个 token。
 */
function playStep() {
  const branch = activeBranch();
  if (!branch || !state.play.paused) return;
  if (state.play.revealed < branch.steps.length) {
    revealOne(branch);
    // 单步到末步同样收尾：矩阵按钮、完成态与自动播放一致。
    if (branch.final && state.play.revealed >= branch.steps.length) finishPlayback(branch);
  }
  syncPlayControls();
}

/**
 * 设置播放速度。
 */
function setPlaySpeed(value) {
  state.play.speed = Math.max(0.25, Math.min(4, Number(value) || 0.5));
  const out = byId("gen-play-speed-out");
  if (out) out.textContent = `${state.play.speed}×`;
  // 速度变化立即生效：重排下一次揭示。
  if (!state.play.paused) scheduleReveal();
}

/**
 * 重置播放进度。
 */
function resetPlay() {
  clearPlayTimer();
  state.play.paused = false;
  state.play.revealed = 0;
  state.play.finished = false;
  state.stripRendered = 0;
  state.outputRendered = 0;
}

/** 揭示追平且推理已完成时收尾：显示完成态、完整矩阵与统计。 */
function finishPlayback(branch) {
  const b = branch || activeBranch();
  if (!b || !b.final) return;
  if (state.play.revealed < b.steps.length) return;
  clearPlayTimer();
  state.play.finished = true;
  if (state.follow) state.selected = b.steps.length;
  setupHeatmap(b);
  renderRevealed();
  const reason = b.final.reason === "eos" ? "遇到 <eos>" : "达到步数上限";
  setStatus(`完成：${b.steps.length} 个 token（${reason}）。`);
  const matrixBtn = byId("gen-attn-matrix-btn");
  if (matrixBtn) matrixBtn.disabled = false;
}

/**
 * 创建一个生成分支（参数、前缀、随机种子）。
 */
function createBranch({ name, params, prefixIds, seed }) {
  state.seq += 1;
  return {
    id: state.seq,
    name,
    params: { ...params },
    prefixIds: [...(prefixIds || [])],
    seed,
    steps: [],
    meta: null,
    final: null,
    status: "idle",
    error: null,
    controller: null,
  };
}

/**
 * 启动指定分支的流式生成。
 */
function startBranch(branch) {
  state.branches.push(branch);
  state.activeId = branch.id;
  state.selected = null;
  state.follow = true;
  resetPlay();
  const toggle = byId("gen-follow-toggle");
  if (toggle) toggle.checked = true;
  rebuildStrip();
  renderAll();
  runBranch(branch);
}

/**
 * 按运行状态启用/禁用按钮。
 */
function syncButtons() {
  const running = state.running;
  const start = byId("gen-start-btn");
  const stop = byId("gen-stop-btn");
  const addBranch = byId("gen-branch-new-btn");
  const random = byId("gen-random-seed-btn");
  if (start) start.disabled = running;
  if (stop) stop.disabled = !running;
  if (addBranch) addBranch.disabled = running;
  if (random) random.disabled = running;
}

/**
 * 显示生成状态提示。
 */
function setStatus(text, isError = false) {
  const node = byId("gen-status");
  if (!node) return;
  node.textContent = text || "";
  node.classList.toggle("error", Boolean(isError));
}

/**
 * 更新生成进度文本。
 */
function setProgress(text) {
  const node = byId("gen-strip-progress");
  if (node) node.textContent = text;
}

/* ------------------------------------------------------------------ 生成 */
function collectParams() {
  const number = (id, fallback) => {
    const node = byId(id);
    const value = Number(node ? node.value : fallback);
    return Number.isFinite(value) ? value : fallback;
  };
  return {
    run_id: byId("gen-run-select") ? byId("gen-run-select").value : "",
    prompt: byId("gen-prompt") ? byId("gen-prompt").value : "",
    template: byId("gen-template") ? byId("gen-template").value : "auto",
    temperature: number("gen-temperature", 0.8),
    top_k: Math.round(number("gen-top-k", 40)),
    top_p: number("gen-top-p", 1),
    max_new_tokens: Math.round(number("gen-max-new", 24)),
    trace_candidates: Math.round(number("gen-trace-k", 16)),
    seed: Math.round(number("gen-seed", 1337)),
  };
}

/**
 * 构造推理请求体。
 */
function buildRequestBody(branch) {
  return {
    run_id: branch.params.run_id,
    prompt: branch.params.prompt,
    template: branch.params.template,
    temperature: branch.params.temperature,
    top_k: branch.params.top_k,
    top_p: branch.params.top_p,
    max_new_tokens: branch.params.max_new_tokens,
    trace_candidates: branch.params.trace_candidates,
    max_context: MAX_CONTEXT,
    seed: branch.seed,
    prefix_token_ids: branch.prefixIds,
    device: "auto",
  };
}

/** 分支列表轻量刷新：签名不变不重建 DOM（每个 step 事件都会调用）。 */
let branchesSig = "";
/**
 * 渲染分支标签栏。
 */
function renderBranches(force) {
  const root = byId("gen-branches");
  if (!root) return;
  const sig = state.branches.map((b) => `${b.id}:${b.status}:${b.steps.length}:${b.error || ""}`).join("|") + `#${state.activeId}`;
  if (!force && sig === branchesSig) {
    syncBranchProgress();
    return;
  }
  branchesSig = sig;
  root.replaceChildren();
  if (!state.branches.length) {
    root.appendChild(muted("还没有分支。选中某一步的候选后点“续写”，会在这里开一个新分支。"));
    return;
  }
  state.branches.forEach((branch) => {
    const item = element("div", "gen-branch-item" + (branch.id === state.activeId ? " active" : ""));
    item.dataset.branchId = String(branch.id);
    item.setAttribute("role", "button");
    item.tabIndex = 0;
    item.setAttribute("aria-pressed", branch.id === state.activeId ? "true" : "false");
    const head = element("div", "gen-branch-head");
    head.appendChild(element("span", "gen-branch-name", branch.name));
    head.appendChild(element("span", `gen-branch-status ${branch.status}`, STATUS_LABEL[branch.status] || branch.status));
    item.appendChild(head);
    const prefixNote = branch.prefixIds.length ? ` · 固定前缀 ${branch.prefixIds.length}` : "";
    const shownCount = branch.id === state.activeId ? visibleCount(branch) : branch.steps.length;
    item.appendChild(
      element("span", "gen-branch-sub", `seed ${branch.seed} · ${shownCount}/${branch.params.max_new_tokens} token${prefixNote}`),
    );
    const preview = branch.steps.slice(0, shownCount).map((step) => tokenText(step.token)).join("").replace(/\s+/g, " ").trim();
    item.appendChild(element("span", "gen-branch-preview", preview || "（还没生成 token）"));
    if (branch.error) item.appendChild(element("span", "gen-branch-preview error", branch.error));
    const del = element("button", "gen-branch-del", "删除");
    del.type = "button";
    del.dataset.deleteBranch = String(branch.id);
    item.appendChild(del);
    root.appendChild(item);
  });
}

/** 只更新活动分支的进度/预览文本（揭示每个 token 时调用，不重建列表）。 */
function syncBranchProgress() {
  const root = byId("gen-branches");
  const branch = activeBranch();
  if (!root || !branch) return;
  const item = root.querySelector(`.gen-branch-item[data-branch-id="${branch.id}"]`);
  if (!item) return;
  const shownCount = visibleCount(branch);
  const prefixNote = branch.prefixIds.length ? ` · 固定前缀 ${branch.prefixIds.length}` : "";
  const sub = item.querySelector(".gen-branch-sub");
  if (sub) sub.textContent = `seed ${branch.seed} · ${shownCount}/${branch.params.max_new_tokens} token${prefixNote}`;
  const preview = item.querySelector(".gen-branch-preview:not(.error)");
  if (preview) {
    const text = branch.steps.slice(0, shownCount).map((step) => tokenText(step.token)).join("").replace(/\s+/g, " ").trim();
    preview.textContent = text || "（还没生成 token）";
  }
}

/**
 * 处理 worker 推来的单个事件。
 */
function handleEvent(branch, event) {
  const type = event.type;
  if (type === "status") {
    if (branch.id === state.activeId) {
      setStatus(event.phase === "loading" ? "加载 checkpoint / 启动 PyTorch 子进程…" : "模型已加载，开始逐 token 生成…");
    }
    return;
  }
  if (type === "meta") {
    branch.meta = event;
    if (branch.id === state.activeId) {
      renderMetaLine();
      syncOutput();
      setStatus(`生成中：0 / ${branch.params.max_new_tokens}`);
    }
    renderBranches();
    return;
  }
  if (type === "step") {
    branch.steps.push(event.data);
    if (branch.id === state.activeId) {
      // 推理实时入缓冲；揭示节奏由播放引擎控制。
      syncPlayControls();
      if (!state.play.paused && state.play.timer === null) scheduleReveal();
    }
    renderBranches();
    return;
  }
  if (type === "final") {
    branch.final = event.data;
    branch.status = "done";
    if (branch.id === state.activeId) {
      setupHeatmap(branch);
      syncPlayControls();
      // 若已追平缓冲则立即收尾，否则等播放揭示完成后再收尾。
      if (state.play.revealed >= branch.steps.length) finishPlayback(branch);
    }
    renderBranches();
    return;
  }
  if (type === "error") {
    branch.status = "error";
    branch.error = event.message || "未知错误";
    if (branch.id === state.activeId) {
      clearPlayTimer();
      setStatus(`生成失败：${branch.error}`, true);
    }
    renderBranches();
    return;
  }
  if (type === "done") {
    if (branch.status === "running") branch.status = "done";
    renderBranches();
  }
}

/**
 * 以流式方式执行一次生成。
 */
async function runBranch(branch) {
  state.running = true;
  const controller = new AbortController();
  branch.controller = controller;
  syncButtons();
  branch.status = "running";
  branch.error = null;
  setStatus("连接推理子进程…");
  try {
    const response = await fetch(endpoints.inferenceStream(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildRequestBody(branch)),
      signal: controller.signal,
    });
    if (!response.ok) {
      const text = await response.text();
      let message = `HTTP ${response.status}`;
      try {
        const payload = JSON.parse(text);
        message = (payload && payload.error) || message;
      } catch (error) {
        message = `${message}: ${text.slice(0, 200)}`;
      }
      throw new Error(message);
    }
    if (!response.body) throw new Error("浏览器不支持流式响应（需要 fetch + ReadableStream）");
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";
    let terminal = false;
    let parseWarned = false;
    const consumeLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed) return false;
      let event;
      try {
        event = JSON.parse(trimmed);
      } catch (error) {
        // 单行损坏不拖垮整条流：提示一次并跳过该行。
        if (!parseWarned) {
          parseWarned = true;
          showToast("warn", "推理流出现一行无法解析的数据", "已跳过该行，继续接收后续 token。");
        }
        return false;
      }
      handleEvent(branch, event);
      return event.type === "done" || event.type === "error";
    };
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";
      for (const line of lines) {
        if (consumeLine(line)) {
          terminal = true;
          break;
        }
      }
      if (terminal) break;
    }
    if (!terminal && buffer.trim()) {
      terminal = consumeLine(buffer);
    }
    if (terminal) {
      // 收到终止事件就主动关掉连接，别让“停止”按钮看上去还在工作。
      try {
        await reader.cancel();
      } catch (error) {
        /* 已经结束，忽略 */
      }
    }
    if (branch.status === "running") {
      branch.status = branch.final ? "done" : "stopped";
    }
  } catch (error) {
    if (error && error.name === "AbortError") {
      if (branch.final) {
        // 已经拿到 final，说明生成其实完成了，这次中断只是关连接。
        branch.status = "done";
      } else {
        branch.status = "stopped";
        if (branch.id === state.activeId) {
          setStatus(branch.steps.length ? `已停止（保留已收到的 ${branch.steps.length} 个 token）。` : "已停止。");
        }
      }
    } else {
      branch.status = "error";
      branch.error = error && error.message ? error.message : String(error);
      if (branch.id === state.activeId) {
        setStatus(`生成失败：${branch.error}`, true);
      }
    }
  } finally {
    branch.controller = null;
    // 只有本分支仍是唯一在跑的流时才复位全局 running。
    if (!state.branches.some((b) => b.controller)) {
      state.running = false;
    }
    syncButtons();
    renderBranches(true);
    if (branch.id === state.activeId) {
      renderOutputMeta();
      syncOutput();
      if (branch.status === "stopped" || branch.status === "error") {
        // 中断/出错：立即呈现全部已缓冲 token，不再残留“生成中”。
        clearPlayTimer();
        state.play.revealed = branch.steps.length;
        renderRevealed();
        if (branch.status === "stopped") {
          setStatus(branch.steps.length ? `已停止（保留已收到的 ${branch.steps.length} 个 token）。` : "已停止。");
        }
      } else if (branch.final) {
        // 正常完成：保持播放节奏，揭示追平缓冲后才收尾。
        if (state.play.revealed >= branch.steps.length) finishPlayback(branch);
        else if (!state.play.paused) scheduleReveal();
      }
      syncPlayControls();
    }
  }
}

/* ------------------------------------------------------------------ ① 时间轴 */
function chipFor(step) {
  const chip = element("button", "gen-chip");
  chip.type = "button";
  chip.dataset.step = String(step.step);
  if (step.finished) chip.classList.add("eos");
  const color = probColor(stepRelative(step));
  chip.title = `#${step.step} 选中 ${tokenLabel(step.token)} · p=${fmt(step.prob)}${step.elapsed_ms ? ` · ${fmt(step.elapsed_ms, 1)} ms` : ""}`;
  chip.appendChild(element("span", "gen-chip-index", `#${step.step}`));
  const token = element("span", "gen-chip-token", tokenLabel(step.token));
  token.style.color = color;
  chip.appendChild(token);
  const bar = element("span", "gen-chip-bar");
  const fill = element("span", "gen-chip-bar-fill");
  fill.style.width = `${Math.max(2, Math.round(stepRelative(step) * 100))}%`;
  fill.style.background = color;
  bar.appendChild(fill);
  chip.appendChild(bar);
  chip.appendChild(element("span", "gen-chip-prob", `p ${pct(step.prob)}`));
  return chip;
}

/**
 * 重建 token 芯片条。
 */
function rebuildStrip() {
  const root = byId("gen-strip");
  if (!root) return;
  root.replaceChildren();
  state.stripRendered = 0;
  state.stripBranchId = state.activeId;
  const branch = activeBranch();
  if (!branch || (!branch.steps.length && !(branch.meta && branch.meta.forced_tokens))) {
    root.appendChild(muted("生成后每个 token 会出现在这里；颜色越绿表示该步越接近最高概率候选。"));
    if (branch) setProgress(`第 0 / ${branch.params.max_new_tokens} 步`);
    return;
  }
  appendPinnedChips(root, branch);
  syncStrip();
}

/** 增量同步：只追加新揭示的 chip；分支切换或回退时全量重建。 */
function syncStrip() {
  const root = byId("gen-strip");
  const branch = activeBranch();
  if (!root || !branch) return;
  if (state.stripBranchId !== branch.id) {
    rebuildStrip();
    return;
  }
  const visible = visibleCount(branch);
  if (visible < state.stripRendered) {
    rebuildStrip();
    return;
  }
  if (visible === state.stripRendered) {
    markStripActive();
    return;
  }
  const placeholder = root.querySelector("p.muted");
  if (placeholder) placeholder.remove();
  if (!root.querySelector(".gen-pin-group")) appendPinnedChips(root, branch);
  for (let i = state.stripRendered; i < visible; i += 1) {
    root.appendChild(chipFor(branch.steps[i]));
  }
  state.stripRendered = visible;
  markStripActive();
  if (state.follow) requestAnimationFrame(() => { root.scrollLeft = root.scrollWidth; });
}

/** 分支固定前缀（改选/重抽时锁定的 token）以只读 chip 显示在最左边。 */
function appendPinnedChips(root, branch) {
  const forced = (branch.meta && branch.meta.forced_tokens) || [];
  if (!forced.length) return;
  const group = element("span", "gen-pin-group");
  group.appendChild(element("span", "gen-pin-label", `固定前缀 ${forced.length}`));
  const chips = element("span", "gen-pin-chips");
  forced.forEach((token, index) => {
    const chip = element("span", "gen-chip pinned", "");
    chip.title = `分支创建时固定的第 ${index + 1} 个 token：${tokenLabel(token)}`;
    chip.appendChild(element("span", "gen-chip-index", `固定${index + 1}`));
    chip.appendChild(element("span", "gen-chip-token", tokenLabel(token)));
    chips.appendChild(chip);
  });
  group.appendChild(chips);
  root.appendChild(group);
}

/**
 * 标记当前激活芯片。
 */
function markStripActive() {
  const root = byId("gen-strip");
  if (!root) return;
  const selected = selectedStep();
  root.querySelectorAll(".gen-chip").forEach((chip) => {
    chip.classList.toggle("active", Boolean(selected) && chip.dataset.step === String(selected.step));
  });
  const followBtn = byId("gen-follow-btn");
  if (followBtn) followBtn.hidden = state.follow;
}

/* --------------------------------------------------------------- ② 采样流水线 */
function renderPipeline() {
  const root = byId("gen-pipeline");
  const title = byId("gen-step-title");
  if (!root) return;
  const step = selectedStep();
  const branch = activeBranch();
  const key = step && step.sampling ? `${branch.id}:${step.step}` : "";
  if (key === state.pipelineKey) return;
  state.pipelineKey = key;
  if (!step || !step.sampling) {
    root.replaceChildren(muted("生成后这里显示：候选概率 → 温度 → top-k → top-p → 随机落点 → 选中 token。"));
    if (title) title.textContent = "等待生成";
    return;
  }
  const sampling = step.sampling;
  if (title) {
    title.textContent = `第 ${step.step} 步 · 选中 ${tokenLabel(step.token)} · p=${fmt(step.prob)} · ${fmt(step.elapsed_ms, 1)} ms`;
  }
  root.replaceChildren();
  root.appendChild(pipelineSummary(step, sampling));
  root.appendChild(pipelineTable(step, sampling));
  root.appendChild(pipelineAxis(sampling));
}

/**
 * 汇总一步的采样过程（温度/top-k/top-p）。
 */
function pipelineSummary(step, sampling) {
  const box = element("div", "gen-sum");
  const sum = (label, value, extraClass) => {
    const span = element("span", extraClass || "", "");
    span.appendChild(element("b", "", label));
    span.appendChild(document.createTextNode(value));
    return span;
  };
  box.appendChild(sum("模式", sampling.mode === "greedy" ? "贪心（T=0，直接取最大）" : "采样（按概率抽签）"));
  box.appendChild(sum("temperature", fmt(sampling.temperature, 2)));
  if (sampling.filters_applied === false) {
    box.appendChild(sum("截断", "未使用（贪心直接取最大，不看 top-k/top-p）"));
  } else {
    box.appendChild(sum("top_k", String(sampling.top_k)));
    box.appendChild(sum("top_p", fmt(sampling.top_p, 2)));
    box.appendChild(
      sum("核内候选", `${sampling.nucleus_count} 个 / 词表 ${sampling.vocab_size}，质量 ${fmt(sampling.nucleus_mass, 3)}`),
    );
  }
  box.appendChild(sum("熵", `${fmt(sampling.entropy_bits, 3)} bit`));
  box.appendChild(sum("top1", pct(sampling.top1_prob)));
  box.appendChild(sum("选中概率", pct(step.prob), "gen-sum-chosen"));
  if (sampling.top_k_threshold_logit !== null && sampling.top_k_threshold_logit !== undefined) {
    box.appendChild(sum("top-k 门槛 logit", fmt(sampling.top_k_threshold_logit, 3)));
  }
  const reroll = element("button", "gen-mini-btn", "重抽这一步");
  reroll.type = "button";
  reroll.dataset.reroll = String(step.step);
  box.appendChild(reroll);
  return box;
}

/**
 * 把采样步骤渲染成表格。
 */
function pipelineTable(step, sampling) {
  const box = element("div");
  const legend = element("div", "gen-stage-legend");
  ["候选 token", "T=1", "温度后", "top-k 后", "top-p 后", "最终概率", "操作"].forEach((text) => {
    legend.appendChild(element("span", "", text));
  });
  box.appendChild(legend);

  const candidates = sampling.candidates || [];
  const maxRaw = Math.max(1e-9, ...candidates.map((candidate) => candidate.prob_raw));
  const list = element("div", "gen-cand-list");
  candidates.forEach((candidate) => {
    const row = element("div", "gen-cand-row");
    if (candidate.chosen) row.classList.add("chosen");
    if (!candidate.in_top_k) row.classList.add("out-k");
    else if (!candidate.in_top_p) row.classList.add("out-p");
    row.title = `token_id=${candidate.token_id} · logit=${fmt(candidate.logit, 3)} · 区间 [${fmt(candidate.cum_low, 3)}, ${fmt(candidate.cum_high, 3)})`;
    row.appendChild(element("span", "gen-cand-token", tokenLabel(candidate.token)));
    row.appendChild(cell(candidate.prob_t1, "T=1"));
    row.appendChild(cell(candidate.prob_raw, "温度后"));
    row.appendChild(cell(candidate.prob_top_k, "top-k"));
    row.appendChild(cell(candidate.prob_final, "最终"));
    const bar = element("span", "gen-cand-bar");
    const fill = element("span", "gen-cand-bar-fill" + (candidate.chosen ? " chosen" : ""));
    fill.style.width = `${Math.max(1, Math.round((candidate.prob_final / Math.max(1e-9, sampling.top1_prob)) * 100))}%`;
    bar.appendChild(fill);
    bar.title = `原始概率条相对最大值（${pct(maxRaw)} 表示该列最大）`;
    row.appendChild(bar);
    const action = element("span", "gen-cand-action");
    const use = element("button", "gen-mini-btn", "续写");
    use.type = "button";
    use.dataset.branchToken = String(candidate.token_id);
    use.dataset.step = String(step.step);
    use.title = `固定这一步为 ${tokenLabel(candidate.token)}，作为新分支继续生成`;
    action.appendChild(use);
    row.appendChild(action);
    list.appendChild(row);
  });
  box.appendChild(list);
  return box;
}

/**
 * 渲染流水线表格单元格。
 */
function cell(value, label) {
  const node = element("span", "gen-cand-cell", pct(value));
  if (label) node.dataset.label = label;
  if (!(Number(value) > 0)) node.classList.add("zero");
  return node;
}

/**
 * 绘制采样阈值轴与候选分布。
 */
function pipelineAxis(sampling) {
  const box = element("div", "gen-axis");
  const head = element("div", "gen-axis-head");
  head.appendChild(element("span", "", "累积分布：把最终概率按 token id 累加，随机数 u 落在哪个区间就选哪个 token"));
  const u = sampling.u;
  head.appendChild(
    element(
      "span",
      "",
      u === null || u === undefined
        ? "贪心模式：没有随机数"
        : `u = ${fmt(u, 4)} → 选中 ${tokenLabel(sampling.candidates.find((c) => c.chosen)?.token)} 的区间 [${fmt(sampling.chosen_cum_low, 3)}, ${fmt(sampling.chosen_cum_high, 3)})`,
    ),
  );
  box.appendChild(head);

  const track = element("div", "gen-axis-track");
  const segments = sampling.nucleus_segments || [];
  if (segments.length) {
    segments.forEach((segment) => {
      const node = element("span", "gen-axis-seg" + (segment.chosen ? " chosen" : ""));
      node.style.left = `${segment.cum_low * 100}%`;
      node.style.width = `${Math.max(0.25, (segment.cum_high - segment.cum_low) * 100)}%`;
      node.style.background = probColor(segment.prob / Math.max(1e-9, sampling.top1_prob));
      node.title = `token_id=${segment.token_id} ${tokenLabel(segment.token)} p=${pct(segment.prob)} 区间 [${fmt(segment.cum_low, 3)}, ${fmt(segment.cum_high, 3)})`;
      track.appendChild(node);
    });
  } else {
    const node = element("span", "gen-axis-seg chosen");
    node.style.left = `${sampling.chosen_cum_low * 100}%`;
    node.style.width = `${Math.max(0.5, (sampling.chosen_cum_high - sampling.chosen_cum_low) * 100)}%`;
    node.style.background = probColor(1);
    track.appendChild(node);
  }
  if (u !== null && u !== undefined) {
    const marker = element("span", "gen-axis-u");
    marker.style.left = `${u * 100}%`;
    marker.title = `u = ${fmt(u, 4)}`;
    track.appendChild(marker);
  }
  box.appendChild(track);

  const scale = element("div", "gen-axis-scale");
  scale.appendChild(element("span", "", "0"));
  scale.appendChild(
    element(
      "span",
      "",
      segments.length ? `核内 ${segments.length} 段（按 token id 顺序）` : `核太大（${sampling.nucleus_count} 个），只画出选中区间`,
    ),
  );
  scale.appendChild(element("span", "", "1"));
  box.appendChild(scale);
  return box;
}

/* ----------------------------------------------------------------- ③ 注意力 */
/** 只把当前 Query 的 attn_last 一行传给固定尺寸重点视图。
 *  attn_last[layer][head] 与 context 按下标对齐；不再构造整段 T×T 矩阵。 */
function syncAttentionDemo() {
  if (!state.attentionDemo) return;
  const branch = activeBranch();
  const step = selectedStep();
  if (!step || !step.attn_last || !step.attn_last.length) {
    state.attentionDemo.setData(null);
    syncAttentionQuerySelect();
    return;
  }
  const visible = visibleCount(branch);
  const steps = branch ? branch.steps.slice(0, visible) : [];
  const idx = steps.findIndex((s) => s.step === step.step);
  const isFinalStep = branch && branch.final && step.step === branch.final.step;
  const fullyRevealed = isFinalStep && visible >= branch.steps.length && (state.selected === null || state.selected === step.step);
  const context = step.context || [];
  // 累积已揭示步的注意力行，供局部矩阵在播放中按绝对位置对齐（尾窗滑动安全）。
  const firstStep = steps[0];
  const promptLen = firstStep && firstStep.context
    ? firstStep.context.length - (firstStep.step - 1)
    : context.length;
  const history = steps.slice(-16).map((s) => {
    const abs = promptLen + s.step - 1;
    const ctx = s.context || [];
    return { abs, start: abs + 1 - ctx.length, context: ctx, rows: s.attn_last };
  });
  state.attentionDemo.setData({
    context,
    rows: step.attn_last,
    history,
    promptLen,
    queryToken: context.length ? context[context.length - 1] : "",
    predictedToken: step.token,
    step: step.step,
    matrix: fullyRevealed ? branch.final.attention : null,
    matrixContext: fullyRevealed ? branch.final.context : null,
    live: branch && branch.status === "running",
    label: "",
    seriesId: `${branch.id}:${step.step}`,
    nav: {
      canPrev: idx > 0,
      canNext: idx >= 0 && idx < steps.length - 1,
      atLatest: idx === steps.length - 1,
    },
  });
  syncAttentionQuerySelect();
}

/**
 * 同步注意力 Q 位置下拉框。
 */
function syncAttentionQuerySelect() {
  const select = byId("attn-query-select");
  if (!select) return;
  const branch = activeBranch();
  const visible = visibleCount(branch);
  const steps = branch ? branch.steps.slice(0, visible) : [];
  const signature = `${branch ? branch.id : 0}:${visible}:${state.selected}`;
  if (select.dataset.sig === signature) return;
  select.dataset.sig = signature;
  select.replaceChildren();
  if (!steps.length) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "示例 / 等待生成";
    select.appendChild(option);
    select.disabled = true;
    return;
  }
  select.disabled = false;
  const current = selectedStep();
  const recent = steps.slice(-64);
  const list = current && !recent.some((s) => s.step === current.step) ? [current, ...recent] : recent;
  if (steps.length > recent.length) {
    const note = document.createElement("option");
    note.value = "";
    note.disabled = true;
    note.textContent = `仅列最近 ${recent.length} 个位置（共 ${steps.length}）`;
    select.appendChild(note);
  }
  list.forEach((step) => {
    const option = document.createElement("option");
    option.value = String(step.step);
    option.textContent = `#${step.step} · 预测 ${tokenLabel(step.token, 8)}`;
    select.appendChild(option);
  });
  select.value = current ? String(current.step) : "";
}

/**
 * 按增量切换注意力查询位置。
 */
function navigateAttentionQuery(delta) {
  const branch = activeBranch();
  const visible = visibleCount(branch);
  if (!branch || !visible) return;
  const current = selectedStep();
  const pool = branch.steps.slice(0, visible);
  const idx = current ? pool.findIndex((s) => s.step === current.step) : pool.length - 1;
  const next = pool[clampIndex(idx + delta, 0, pool.length - 1)];
  if (!next) return;
  state.follow = next.step === pool[pool.length - 1].step;
  state.selected = next.step;
  const toggle = byId("gen-follow-toggle");
  if (toggle) toggle.checked = state.follow;
  if (state.attentionDemo) state.attentionDemo.setFollow(state.follow);
  markStripActive();
  renderPipeline();
  renderAttention();
}

/**
 * 跳转到最新位置。
 */
function jumpAttentionLatest() {
  const branch = activeBranch();
  const visible = visibleCount(branch);
  if (!branch || !visible) return;
  state.follow = true;
  state.selected = visible;
  const toggle = byId("gen-follow-toggle");
  if (toggle) toggle.checked = true;
  if (state.attentionDemo) state.attentionDemo.setFollow(true);
  markStripActive();
  renderPipeline();
  renderAttention();
}

/**
 * 渲染注意力热图与控制项。
 */
function renderAttention() {
  syncAttentionDemo();
  const canvas = byId("gen-attn-strip");
  const note = byId("gen-attn-note");
  const legend = byId("gen-attn-legend");
  if (!canvas) return;
  const step = selectedStep();
  if (!step || !step.attn_last || !step.attn_last.length) {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(240, canvas.clientWidth || 320);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(160 * dpr);
    canvas.style.height = "160px";
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, 160);
    const style = getComputedStyle(document.documentElement);
    ctx.fillStyle = style.getPropertyValue("--muted").trim() || "rgba(154,157,166,0.7)";
    ctx.font = canvasFont("ui", fontTokens().size.sm);
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText("等待生成 · 当前 Query 的多头注意力概览", width / 2, 80);
    if (note) note.textContent = "等待生成；概览只保留当前 Query 的最近上下文窗口。";
    if (legend) legend.hidden = true;
    return;
  }
  drawAttentionStrip(canvas, step);
  if (legend) legend.hidden = false;
  if (note) {
    const layers = step.attn_last.length;
    const heads = step.attn_last[0] ? step.attn_last[0].length : 0;
    const keep = Math.min(32, step.context.length);
    note.textContent = `预测「${tokenLabel(step.token)}」时，当前 Query 对最近 ${keep} 个 Key 的注意力（${layers} 层 × ${heads} 头；颜色为真实权重）。`;
  }
}

/**
 * 绘制单行注意力色条。
 */
function drawAttentionStrip(canvas, step) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssWidth = Math.max(240, canvas.clientWidth || 360);
  const cssHeight = 160;
  const layers = step.attn_last;
  const nHead = layers[0] ? layers[0].length : 0;
  const rows = layers.length * nHead;
  const context = step.context || [];
  const keep = Math.min(32, context.length);
  const offset = context.length - keep;
  const labelH = 30;
  const labelW = 42;
  const cell = Math.max(2, (cssWidth - labelW - 8) / Math.max(1, keep));
  const rowH = Math.max(4, Math.min(16, (cssHeight - labelH - 8) / Math.max(1, rows)));
  canvas.width = Math.round(cssWidth * dpr);
  canvas.height = Math.round(cssHeight * dpr);
  canvas.style.height = `${cssHeight}px`;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);
  ctx.font = canvasFont("data", fontTokens().size.xs);
  ctx.textBaseline = "middle";
  const style = getComputedStyle(document.documentElement);
  const labelColor = style.getPropertyValue("--muted").trim() || "#9a9da6";
  const heat = style.getPropertyValue("--accent").trim() || "#7c93f0";
  const heatRgb = (() => {
    const m = heat.match(/^#([0-9a-f]{6})$/i);
    if (!m) return [124, 147, 240];
    return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
  })();

  // 行高不足以容纳文字时抽稀行标签，避免多层多头文字重叠。
  const labelEvery = Math.max(1, Math.ceil(12 / Math.max(1, rowH)));
  for (let layer = 0; layer < layers.length; layer += 1) {
    for (let head = 0; head < nHead; head += 1) {
      const rowIndex = layer * nHead + head;
      const values = (layers[layer][head] || []).slice(offset);
      const y = labelH + rowIndex * rowH;
      if (rowIndex % labelEvery === 0 || rowIndex === rows - 1) {
        ctx.fillStyle = labelColor;
        ctx.textAlign = "right";
        ctx.fillText(`L${layer}H${head}`, labelW - 4, y + rowH / 2);
      }
      values.forEach((value, index) => {
        const alpha = clampIndex(Number(value) || 0, 0, 1);
        ctx.fillStyle = `rgba(${heatRgb[0]}, ${heatRgb[1]}, ${heatRgb[2]}, ${alpha.toFixed(3)})`;
        ctx.fillRect(labelW + index * cell, y, Math.max(1, cell - 0.6), Math.max(1, rowH - 0.6));
      });
    }
  }

  ctx.fillStyle = labelColor;
  ctx.textBaseline = "top";
  const every = Math.max(1, Math.ceil(keep / 12));
  for (let index = 0; index < keep; index += every) {
    ctx.save();
    ctx.translate(labelW + index * cell + cell / 2, labelH - 4);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = "left";
    ctx.fillText(tokenLabel(context[offset + index], 4), 0, 0);
    ctx.restore();
  }
}

/* ------------------------------------------------------------------- ④ 分支 */
function switchBranch(id) {
  const branch = state.branches.find((item) => item.id === id);
  if (!branch || id === state.activeId) return;
  clearPlayTimer();
  state.activeId = id;
  state.play.finished = false;
  // 历史分支直接显示全部已缓冲步骤；正在跑的分支从当前缓冲末尾继续揭示。
  state.play.revealed = branch.steps.length;
  state.play.paused = false;
  state.selected = branch.steps.length ? branch.steps[branch.steps.length - 1].step : null;
  state.follow = true;
  state.pipelineKey = "";
  const toggle = byId("gen-follow-toggle");
  if (toggle) toggle.checked = true;
  renderAll();
  if (branch.final) finishPlayback(branch);
  else if (branch.status === "running") scheduleReveal();
  const matrixBtn = byId("gen-attn-matrix-btn");
  if (matrixBtn) matrixBtn.disabled = !branch.final;
  if (branch.final) setupHeatmap(branch);
  if (branch.status === "running") {
    setStatus(`正在生成「${branch.name}」；推理实时入缓冲，按播放速度揭示。`);
  } else {
    setStatus(branch.error ? `生成失败：${branch.error}` : `已切换到「${branch.name}」。`, Boolean(branch.error));
  }
}

/* ------------------------------------------------------------------- ⑤ 输出 */
function syncOutput() {
  const root = byId("gen-output");
  const branch = activeBranch();
  if (!root) return;
  if (!branch || !branch.meta) {
    root.replaceChildren(muted("（空）"));
    state.outputRendered = 0;
    state.outputBranchId = 0;
    return;
  }
  const visible = visibleCount(branch);
  if (state.outputBranchId !== branch.id || visible < state.outputRendered) {
    root.replaceChildren();
    root.appendChild(element("span", "gen-out-prompt", branch.meta.prompt));
    state.outputRendered = 0;
    state.outputBranchId = branch.id;
  }
  for (let i = state.outputRendered; i < visible; i += 1) {
    const step = branch.steps[i];
    const span = element("span", "gen-out-token", tokenText(step.token));
    span.dataset.step = String(step.step);
    span.title = `#${step.step} p=${fmt(step.prob)} · 点击查看这一步`;
    span.style.color = probColor(stepRelative(step));
    root.appendChild(span);
    if (step.finished) root.appendChild(element("span", "gen-out-eos", " ⟨eos⟩"));
  }
  state.outputRendered = visible;
  // 结尾省略号：缓冲还有未揭示内容时显示。
  const tail = root.querySelector(".gen-out-tail");
  const needTail = visible < branch.steps.length || (branch.status === "running" && !visible);
  if (needTail && !tail) {
    root.appendChild(element("span", "muted gen-out-tail", "…"));
  } else if (!needTail && tail) {
    tail.remove();
  }
}

/**
 * 渲染生成结果。
 */
function renderOutput() {
  syncOutput();
}

/**
 * 渲染结果元信息。
 */
function renderOutputMeta() {
  const node = byId("gen-output-meta");
  if (!node) return;
  const branch = activeBranch();
  const visible = visibleCount(branch);
  if (!branch || !visible) {
    node.textContent = "";
    return;
  }
  const shown = branch.steps.slice(0, visible);
  const totalMs = shown.reduce((sum, step) => sum + (Number(step.elapsed_ms) || 0), 0);
  const speed = totalMs > 0 ? `${(shown.length * 1000 / totalMs).toFixed(1)} tok/s` : "速度未知";
  const done = branch.final && visible >= branch.steps.length;
  const reason = done ? (branch.final.reason === "eos" ? "遇到 <eos>" : "达到步数上限") : branch.status === "running" ? "生成中" : STATUS_LABEL[branch.status] || "";
  node.textContent = `${visible} token · ${speed} · 前向合计 ${fmt(totalMs / 1000, 2)} s · seed ${branch.seed} · ${reason}`;
}

/**
 * 渲染参数/耗时摘要行。
 */
function renderMetaLine() {
  const node = byId("gen-meta-line");
  if (!node) return;
  const branch = activeBranch();
  const meta = branch && branch.meta;
  if (!meta) {
    node.textContent = "";
    return;
  }
  const model = meta.model || {};
  node.textContent = `${meta.stage} · ${meta.device} · ${Number(meta.parameters).toLocaleString()} 参数 · ${model.n_layer}L/${model.n_head}H/${model.n_embd}D · prompt ${meta.prompt_tokens.length} token · seed ${meta.seed ?? branch.seed}`;
}

/* ------------------------------------------------------------ 完整矩阵浮层 */
function setupHeatmap(branch) {
  const data = branch && branch.final;
  const layerSelect = byId("gen-layer-select");
  const headSelect = byId("gen-head-select");
  if (!data || !data.attention || !data.attention.length || !layerSelect || !headSelect) return;
  layerSelect.replaceChildren();
  headSelect.replaceChildren();
  data.attention.forEach((heads, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `layer ${index}`;
    layerSelect.appendChild(option);
  });
  (data.attention[0] || []).forEach((_, index) => {
    const option = document.createElement("option");
    option.value = String(index);
    option.textContent = `head ${index}`;
    headSelect.appendChild(option);
  });
  state.layer = 0;
  state.head = 0;
  layerSelect.value = "0";
  headSelect.value = "0";
  const button = byId("gen-attn-matrix-btn");
  if (button) button.disabled = false;
}

/** 矩阵热图：DPR 感知，位图尺寸 = CSS 尺寸 × dpr，高分屏不模糊。 */
function drawMatrix() {
  const canvas = byId("gen-heatmap");
  const note = byId("gen-final-note");
  const legend = byId("gen-legend");
  const branch = activeBranch();
  if (!canvas || !branch || !branch.final) return;
  const attention = branch.final.attention || [];
  if (!attention.length) return;
  const layer = Math.max(0, Math.min(state.layer, attention.length - 1));
  const heads = attention[layer] || [];
  const head = Math.max(0, Math.min(state.head, heads.length - 1));
  const matrix = heads[head] || [];
  const context = branch.final.context || [];
  const n = matrix.length;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const margin = 70;
  const size = 620;
  const cssSize = margin + size + 18;
  const cell = size / Math.max(1, n);
  canvas.width = Math.round(cssSize * dpr);
  canvas.height = Math.round(cssSize * dpr);
  canvas.style.height = "auto";
  canvas.hidden = false;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssSize, cssSize);
  const style = getComputedStyle(document.documentElement);
  const labelColor = style.getPropertyValue("--muted").trim() || "#9a9da6";
  const borderColor = style.getPropertyValue("--border").trim() || "#2a2d33";
  const heat = style.getPropertyValue("--accent").trim() || "#7c93f0";
  const heatRgb = (() => {
    const m = heat.match(/^#([0-9a-f]{6})$/i);
    if (!m) return [124, 147, 240];
    return [parseInt(m[1].slice(0, 2), 16), parseInt(m[1].slice(2, 4), 16), parseInt(m[1].slice(4, 6), 16)];
  })();
  ctx.font = canvasFont("data", fontTokens().size.xs);
  ctx.fillStyle = labelColor;
  const every = Math.max(1, Math.ceil(n / 16));
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i < n; i += 1) {
    if (i % every === 0) ctx.fillText(tokenLabel(context[i]), margin - 6, margin + i * cell + cell / 2);
  }
  ctx.textAlign = "left";
  ctx.textBaseline = "bottom";
  for (let j = 0; j < n; j += 1) {
    if (j % every !== 0) continue;
    ctx.save();
    ctx.translate(margin + j * cell + cell / 2, margin - 4);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(tokenLabel(context[j]), 0, 0);
    ctx.restore();
  }
  for (let i = 0; i < n; i += 1) {
    for (let j = 0; j < n; j += 1) {
      const value = Number((matrix[i] || [])[j] || 0);
      ctx.fillStyle = `rgba(${heatRgb[0]},${heatRgb[1]},${heatRgb[2]},${Math.min(1, Math.pow(value, 0.6) * 1.7).toFixed(3)})`;
      ctx.fillRect(margin + j * cell, margin + i * cell, cell - 0.5, cell - 0.5);
    }
  }
  ctx.strokeStyle = borderColor;
  ctx.strokeRect(margin, margin, size, size);
  if (note) {
    note.hidden = false;
    note.textContent = `最终上下文窗口 ${n} 个 token（非全部生成历史）；layer ${layer} / head ${head}；行=query，列=key，因果上三角为 0。`;
  }
  if (legend) legend.hidden = false;
}

/* ------------------------------------------------------------------ 渲染总入口 */
function renderAll() {
  state.pipelineKey = "";
  rebuildStrip();
  renderPipeline();
  renderAttention();
  renderBranches(true);
  state.outputBranchId = 0;
  syncOutput();
  renderOutputMeta();
  renderMetaLine();
  syncButtons();
  syncPlayControls();
  const branch = activeBranch();
  const matrixBtn = byId("gen-attn-matrix-btn");
  if (matrixBtn) matrixBtn.disabled = !(branch && branch.final && visibleCount(branch) >= branch.steps.length);
}

/* ------------------------------------------------------------------ 分支操作 */
function randomSeed() {
  return Math.floor(Math.random() * 100000);
}

/**
 * 开始一次新的生成。
 */
function startGeneration() {
  if (state.running) {
    setStatus("正在生成，先点“停止”再开始新的生成。", true);
    return;
  }
  const params = collectParams();
  if (!params.run_id) {
    setStatus("没有可用 checkpoint；先训练一次，或检查 runs/ 下是否有 best.pt。", true);
    return;
  }
  state.branches = [];
  state.branches.push(createBranch({ name: "主路径", params, prefixIds: [], seed: params.seed }));
  state.activeId = state.branches[state.branches.length - 1].id;
  state.selected = null;
  state.follow = true;
  state.play.lastReplay = { params, seed: params.seed };
  resetPlay();
  const toggle = byId("gen-follow-toggle");
  if (toggle) toggle.checked = true;
  renderAll();
  runBranch(state.branches[0]);
}

/** 用同一 seed 与参数从头重播本次生成。 */
function replayGeneration() {
  const replay = state.play.lastReplay;
  if (!replay || state.running) return;
  const params = replay.params;
  state.branches = [];
  state.branches.push(createBranch({ name: "重播", params, prefixIds: [], seed: replay.seed }));
  state.activeId = state.branches[state.branches.length - 1].id;
  state.selected = null;
  state.follow = true;
  resetPlay();
  const toggle = byId("gen-follow-toggle");
  if (toggle) toggle.checked = true;
  renderAll();
  runBranch(state.branches[0]);
}

/**
 * 以新种子新增分支。
 */
function addBranchWithNewSeed() {
  if (state.running) {
    setStatus("正在生成，先停止再开新分支。", true);
    return;
  }
  if (!state.branches.length) {
    startGeneration();
    return;
  }
  if (state.branches.length >= MAX_BRANCHES) {
    setStatus(`分支最多 ${MAX_BRANCHES} 个，先删除一个。`, true);
    return;
  }
  const params = collectParams();
  const seed = randomSeed();
  const branch = createBranch({ name: `新分支 seed ${seed}`, params, prefixIds: [], seed });
  startBranch(branch);
}

/**
 * 从某个候选 token 分叉出新分支。
 */
function branchFromCandidate(stepNumber, tokenId) {
  if (state.running) {
    setStatus("正在生成，先停止再改选。", true);
    return;
  }
  const branch = activeBranch();
  if (!branch) return;
  if (state.branches.length >= MAX_BRANCHES) {
    setStatus(`分支最多 ${MAX_BRANCHES} 个，先删除一个。`, true);
    return;
  }
  const params = collectParams();
  if (params.run_id !== branch.params.run_id) {
    setStatus("当前选择的 checkpoint 与该分支不同，不能跨模型改选。", true);
    return;
  }
  if (stepNumber < 1 || stepNumber > branch.steps.length) return;
  const step = branch.steps[stepNumber - 1];
  const candidate = (step.sampling && step.sampling.candidates ? step.sampling.candidates : []).find(
    (item) => item.token_id === tokenId,
  );
  const label = candidate ? tokenLabel(candidate.token) : `id ${tokenId}`;
  const prefix = branch.steps.slice(0, stepNumber - 1).map((item) => item.token_id);
  prefix.push(tokenId);
  const remaining = Math.max(1, params.max_new_tokens - stepNumber);
  const newBranch = createBranch({
    name: `#${stepNumber} 改选 ${label}`,
    params: { ...params, max_new_tokens: remaining },
    prefixIds: prefix,
    seed: params.seed,
  });
  startBranch(newBranch);
}

/**
 * 对指定步骤重新采样（固定前缀）。
 */
function rerollStep(stepNumber) {
  if (state.running) {
    setStatus("正在生成，先停止再重抽。", true);
    return;
  }
  const branch = activeBranch();
  if (!branch) return;
  if (state.branches.length >= MAX_BRANCHES) {
    setStatus(`分支最多 ${MAX_BRANCHES} 个，先删除一个。`, true);
    return;
  }
  const params = collectParams();
  if (params.run_id !== branch.params.run_id) {
    setStatus("当前选择的 checkpoint 与该分支不同，不能跨模型重抽。", true);
    return;
  }
  if (stepNumber < 1 || stepNumber > branch.steps.length) return;
  const prefix = branch.steps.slice(0, stepNumber - 1).map((item) => item.token_id);
  const remaining = Math.max(1, params.max_new_tokens - (stepNumber - 1));
  const seed = (branch.seed + 1 + randomSeed()) % 100000;
  const newBranch = createBranch({
    name: `#${stepNumber} 重抽 seed ${seed}`,
    params: { ...params, max_new_tokens: remaining },
    prefixIds: prefix,
    seed,
  });
  startBranch(newBranch);
}

/**
 * 删除分支。
 */
function deleteBranch(id) {
  const branch = state.branches.find((item) => item.id === id);
  if (!branch) return;
  // 无论是否活动分支，运行中的流都要中止，不留孤儿连接。
  if (branch.controller) branch.controller.abort();
  state.branches = state.branches.filter((item) => item.id !== id);
  if (state.activeId === id) {
    clearPlayTimer();
    const next = state.branches[state.branches.length - 1];
    state.activeId = next ? next.id : 0;
    state.play.revealed = next ? next.steps.length : 0;
    state.play.paused = false;
    state.play.finished = false;
    state.selected = next && next.steps.length ? next.steps[next.steps.length - 1].step : null;
    state.pipelineKey = "";
    renderAll();
    if (next && next.final) finishPlayback(next);
    else if (next && next.status === "running") scheduleReveal();
    else syncPlayControls();
    if (!next) setStatus("尚未生成。");
  } else {
    renderBranches(true);
  }
}

/* ------------------------------------------------------------------ N09 参数对比 */
const MAX_COMPARE_VARIANTS = 4;
const COMPARE_FIELDS = [
  { key: "temperature", label: "temperature", value: 0.8, step: 0.05, min: 0, max: 5 },
  { key: "top_k", label: "top_k", value: 40, step: 1, min: 0, max: 1000 },
  { key: "top_p", label: "top_p", value: 1, step: 0.01, min: 0, max: 1 },
  { key: "repetition_penalty", label: "重复惩罚", value: 1.0, step: 0.05, min: 0.1, max: 5 },
  { key: "no_repeat_ngram_size", label: "禁重复n-gram", value: 0, step: 1, min: 0, max: 8 },
];
const compareState = { variants: [], running: false, controller: null, requestKey: "" };

/**
 * 显示对比模式状态。
 */
function compareStatus(text, isError = false) {
  const node = byId("gen-compare-status");
  if (!node) return;
  node.textContent = text || "";
  node.classList.toggle("error", Boolean(isError));
}

/**
 * 新增一个对比变体。
 */
function addCompareVariant(preset) {
  if (compareState.variants.length >= MAX_COMPARE_VARIANTS) {
    compareStatus(`最多 ${MAX_COMPARE_VARIANTS} 组参数。`, true);
    return;
  }
  const root = byId("gen-compare-variants");
  if (!root) return;
  const index = compareState.variants.length;
  const row = element("div", "gen-compare-variant");
  row.dataset.variantIndex = String(index);
  const variant = { inputs: {} };
  COMPARE_FIELDS.forEach((field) => {
    const label = element("label", "field");
    label.appendChild(element("span", null, field.label));
    const input = document.createElement("input");
    input.type = "number";
    input.min = String(field.min);
    input.max = String(field.max);
    input.step = String(field.step);
    const value = preset && preset[field.key] !== undefined ? preset[field.key] : field.value;
    input.value = String(value);
    input.dataset.param = field.key;
    label.appendChild(input);
    row.appendChild(label);
    variant.inputs[field.key] = input;
  });
  const remove = element("button", "btn btn-small btn-ghost-danger gen-compare-remove", "移除");
  remove.type = "button";
  remove.addEventListener("click", () => {
    compareState.variants = compareState.variants.filter((item) => item !== variant);
    row.remove();
    renumberCompareVariants();
  });
  row.appendChild(remove);
  root.appendChild(row);
  compareState.variants.push(variant);
}

/**
 * 重新编号对比变体。
 */
function renumberCompareVariants() {
  const root = byId("gen-compare-variants");
  if (!root) return;
  root.querySelectorAll(".gen-compare-variant").forEach((row, index) => {
    row.dataset.variantIndex = String(index);
  });
}

/**
 * 收集待对比的参数组合。
 */
function collectCompareVariants() {
  return compareState.variants.map((variant) => {
    const params = {};
    COMPARE_FIELDS.forEach((field) => {
      const raw = Number(variant.inputs[field.key].value);
      params[field.key] = Number.isFinite(raw) ? raw : field.value;
    });
    params.top_k = Math.round(params.top_k);
    params.no_repeat_ngram_size = Math.round(params.no_repeat_ngram_size);
    return params;
  });
}

/**
 * 把参数组合格式化为短标签。
 */
function paramsLabel(params) {
  return `T=${params.temperature} · k=${params.top_k} · p=${params.top_p} · rep=${params.repetition_penalty} · ngram=${params.no_repeat_ngram_size}`;
}

/**
 * 渲染对比结果。
 */
function renderCompareResults(payload, requestKey) {
  const root = byId("gen-compare-results");
  if (!root) return;
  // 结果绑定请求标识：checkpoint/prompt 已变化时不渲染过期结果。
  if (requestKey && compareState.requestKey && requestKey !== compareState.requestKey) return;
  root.replaceChildren();
  const results = (payload && payload.results) || [];
  if (!results.length) {
    root.appendChild(muted("没有结果。"));
    return;
  }
  results.forEach((entry) => {
    const card = element("div", "gen-compare-card" + (entry.ok ? "" : " error"));
    const head = element("div", "gen-compare-card-head");
    head.appendChild(element("b", null, `组 ${entry.index + 1}`));
    head.appendChild(element("span", "badge " + (entry.ok ? "badge-ok" : "badge-bad"), entry.ok ? "完成" : "失败"));
    card.appendChild(head);
    card.appendChild(element("div", "gen-compare-card-params", paramsLabel(entry.params || {})));
    if (entry.ok) {
      card.appendChild(element("pre", "gen-compare-card-text", entry.text || "（空输出）"));
      card.appendChild(element("div", "gen-compare-card-meta", `耗时 ${fmt(Number(entry.elapsed_ms) / 1000, 2)} s`));
    } else {
      card.appendChild(element("pre", "gen-compare-card-text", entry.error || "未知错误"));
    }
    root.appendChild(card);
  });
}

/**
 * 执行参数对比请求。
 */
async function runCompare() {
  // 运行中再点 = 取消当前对比。
  if (compareState.running) {
    if (compareState.controller) compareState.controller.abort();
    return;
  }
  if (state.running) {
    compareStatus("主流式生成进行中；先停止再发起参数对比（推理并发上限 2）。", true);
    return;
  }
  const runSelect = byId("gen-run-select");
  const promptInput = byId("gen-prompt");
  const runId = runSelect ? runSelect.value : "";
  if (!runId) {
    compareStatus("没有可用 checkpoint；先训练一次。", true);
    return;
  }
  if (!compareState.variants.length) addCompareVariant();
  const variants = collectCompareVariants();
  const body = {
    run_id: runId,
    prompt: promptInput ? promptInput.value : "",
    template: byId("gen-template") ? byId("gen-template").value : "auto",
    max_new_tokens: Math.round(Number(byId("gen-max-new") ? byId("gen-max-new").value : 24) || 24),
    seed: Math.round(Number(byId("gen-seed") ? byId("gen-seed").value : 1337) || 1337),
    variants,
  };
  const requestKey = `${body.run_id}|${body.prompt}|${body.seed}|${Date.now()}`;
  compareState.requestKey = requestKey;
  compareState.running = true;
  const controller = new AbortController();
  compareState.controller = controller;
  const button = byId("gen-compare-run-btn");
  if (button) button.textContent = "停止对比";
  compareStatus(`正在逐组生成（${variants.length} 组，串行）…再点一次可停止`);
  try {
    const payload = await postJson(endpoints.inferenceCompare(), body, { signal: controller.signal, timeoutMs: 0 });
    renderCompareResults(payload, requestKey);
    const failed = (payload.results || []).filter((entry) => !entry.ok).length;
    compareStatus(
      failed ? `对比完成：${payload.count - failed}/${payload.count} 组成功` : `对比完成：${payload.count} 组`,
      Boolean(failed),
    );
  } catch (error) {
    if (controller.signal.aborted) {
      compareStatus("对比已停止。");
    } else {
      compareStatus(`参数对比失败：${error.message}`, true);
    }
  } finally {
    compareState.running = false;
    compareState.controller = null;
    if (button) {
      button.textContent = "开始对比";
      button.disabled = false;
    }
  }
}

/**
 * 切换对比模式。
 */
function setCompareMode(enabled) {
  const panel = byId("gen-compare-panel");
  if (panel) panel.hidden = !enabled;
  if (enabled && !compareState.variants.length) {
    // 预置两组常用对照：贪心 vs 采样。
    addCompareVariant({ temperature: 0, top_k: 0, top_p: 1, repetition_penalty: 1, no_repeat_ngram_size: 0 });
    addCompareVariant({ temperature: 0.8, top_k: 40, top_p: 0.95, repetition_penalty: 1.1, no_repeat_ngram_size: 2 });
  }
}

/**
 * 初始化对比模式。
 */
function initCompareMode() {
  const toggle = byId("gen-compare-toggle");
  if (!toggle) return;
  toggle.addEventListener("change", () => setCompareMode(toggle.checked));
  const add = byId("gen-compare-add");
  if (add) add.addEventListener("click", () => addCompareVariant());
  const run = byId("gen-compare-run-btn");
  if (run) run.addEventListener("click", runCompare);
}

/* ------------------------------------------------------------------ 运行列表 */
async function loadRuns() {
  const select = byId("gen-run-select");
  if (!select) return;
  try {
    const payload = await getJson(endpoints.runs());
    const runs = (payload.runs || []).filter((run) => run.has_best);
    state.runs = runs;
    select.replaceChildren();
    if (!runs.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "没有带 best.pt 的运行";
      select.appendChild(option);
      setStatus("没有可生成的 checkpoint。先完成一次训练，或检查 runs/ 下是否有 best.pt。", true);
      return;
    }
    runs.forEach((run) => {
      const option = document.createElement("option");
      option.value = run.run_id;
      const best = run.best_val === null || run.best_val === undefined ? "" : ` · best=${fmt(run.best_val, 4)}`;
      option.textContent = `${run.run_id} · ${run.stage || "?"}${best}`;
      select.appendChild(option);
    });
    let requested = null;
    try {
      requested = new URLSearchParams(window.location.search).get("run");
    } catch (error) {
      requested = null;
    }
    // 默认选最近更新的可用 checkpoint（列表按更新时间倒序），不再硬编码演示运行。
    const preferred = runs.find((run) => run.run_id === requested) || runs[0];
    select.value = preferred.run_id;
  } catch (error) {
    setStatus(`运行列表加载失败：${error.message}`, true);
  }
}

/**
 * 把 run id 写入 URL 便于分享。
 */
function rememberRunInUrl(runId) {
  try {
    const nextUrl = new URL(window.location.href);
    if (runId) nextUrl.searchParams.set("run", runId);
    else nextUrl.searchParams.delete("run");
    window.history.replaceState({}, "", nextUrl);
  } catch (error) {
    /* 忽略：URL 不可写时只是少了深链。 */
  }
}

/* ------------------------------------------------------------------ 事件绑定 */
function bindEvents() {
  const start = byId("gen-start-btn");
  if (start) start.addEventListener("click", startGeneration);

  const stop = byId("gen-stop-btn");
  if (stop) {
    stop.addEventListener("click", () => {
      const branch = activeBranch();
      const controller = (branch && branch.controller) || state.branches.map((b) => b.controller).find(Boolean);
      if (controller) {
        setStatus("正在停止…");
        controller.abort();
      }
    });
  }

  const random = byId("gen-random-seed-btn");
  if (random) {
    random.addEventListener("click", () => {
      const seedInput = byId("gen-seed");
      if (seedInput) seedInput.value = String(randomSeed());
    });
  }

  const playPause = byId("gen-play-pause");
  if (playPause) playPause.addEventListener("click", togglePlayPause);
  const playStepBtn = byId("gen-play-step");
  if (playStepBtn) playStepBtn.addEventListener("click", playStep);
  const playReplay = byId("gen-play-replay");
  if (playReplay) playReplay.addEventListener("click", replayGeneration);
  const playSpeed = byId("gen-play-speed");
  if (playSpeed) {
    playSpeed.addEventListener("input", () => setPlaySpeed(playSpeed.value));
    setPlaySpeed(playSpeed.value);
  }

  const attentionQuery = byId("attn-query-select");
  if (attentionQuery) {
    attentionQuery.addEventListener("change", () => {
      const stepNumber = Number(attentionQuery.value);
      if (!Number.isFinite(stepNumber)) return;
      const branch = activeBranch();
      const pool = branch ? branch.steps.slice(0, visibleCount(branch)) : [];
      state.follow = Boolean(pool.length && stepNumber === pool[pool.length - 1].step);
      state.selected = stepNumber;
      const toggle = byId("gen-follow-toggle");
      if (toggle) toggle.checked = state.follow;
      if (state.attentionDemo) state.attentionDemo.setFollow(state.follow);
      markStripActive();
      renderPipeline();
      renderAttention();
    });
  }

  const prompt = byId("gen-prompt");
  if (prompt) {
    prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        startGeneration();
      }
    });
  }

  const runSelect = byId("gen-run-select");
  if (runSelect) {
    runSelect.addEventListener("change", () => {
      rememberRunInUrl(runSelect.value);
      // checkpoint 变化后旧对比结果不再可信。
      compareState.requestKey = "";
    });
  }

  const followToggle = byId("gen-follow-toggle");
  if (followToggle) {
    followToggle.addEventListener("change", () => {
      state.follow = followToggle.checked;
      const branch = activeBranch();
      const latest = branch && branch.steps.length ? branch.steps[branch.steps.length - 1] : null;
      if (state.follow || state.selected === null) {
        // 恢复跟随时跳回最新一步；取消跟随时冻结在当前选中的那一步。
        state.selected = latest ? latest.step : null;
      }
      if (state.attentionDemo) state.attentionDemo.setFollow(state.follow);
      markStripActive();
      renderPipeline();
      renderAttention();
    });
  }

  const followBtn = byId("gen-follow-btn");
  if (followBtn) {
    followBtn.addEventListener("click", () => {
      state.follow = true;
      if (followToggle) followToggle.checked = true;
      const branch = activeBranch();
      state.selected = branch && branch.steps.length ? branch.steps[branch.steps.length - 1].step : null;
      if (state.attentionDemo) state.attentionDemo.setFollow(true);
      markStripActive();
      renderPipeline();
      renderAttention();
    });
  }

  const strip = byId("gen-strip");
  if (strip) {
    strip.addEventListener("click", (event) => {
      const chip = event.target.closest ? event.target.closest(".gen-chip") : null;
      if (!chip || chip.classList.contains("pinned")) return;
      const stepNumber = Number(chip.dataset.step);
      if (!Number.isFinite(stepNumber)) return;
      state.follow = false;
      if (followToggle) followToggle.checked = false;
      if (state.attentionDemo) state.attentionDemo.setFollow(false);
      state.selected = stepNumber;
      markStripActive();
      renderPipeline();
      renderAttention();
    });
  }

  const pipeline = byId("gen-pipeline");
  if (pipeline) {
    pipeline.addEventListener("click", (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      const branchButton = target.closest("[data-branch-token]");
      if (branchButton) {
        branchFromCandidate(Number(branchButton.dataset.step), Number(branchButton.dataset.branchToken));
        return;
      }
      const reroll = target.closest("[data-reroll]");
      if (reroll) rerollStep(Number(reroll.dataset.reroll));
    });
  }

  const branches = byId("gen-branches");
  if (branches) {
    branches.addEventListener("click", (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      const del = target.closest("[data-delete-branch]");
      if (del) {
        event.stopPropagation();
        deleteBranch(Number(del.dataset.deleteBranch));
        return;
      }
      const item = target.closest(".gen-branch-item");
      if (item) switchBranch(Number(item.dataset.branchId));
    });
    branches.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const target = event.target;
      if (!target || !target.closest) return;
      const item = target.closest(".gen-branch-item");
      if (!item || target.closest("[data-delete-branch]")) return;
      event.preventDefault();
      switchBranch(Number(item.dataset.branchId));
    });
  }

  const newBranch = byId("gen-branch-new-btn");
  if (newBranch) newBranch.addEventListener("click", addBranchWithNewSeed);

  const output = byId("gen-output");
  if (output) {
    output.addEventListener("click", (event) => {
      const target = event.target;
      if (!target || !target.closest) return;
      const token = target.closest("[data-step]");
      if (!token) return;
      const stepNumber = Number(token.dataset.step);
      if (!Number.isFinite(stepNumber)) return;
      state.follow = false;
      if (followToggle) followToggle.checked = false;
      if (state.attentionDemo) state.attentionDemo.setFollow(false);
      state.selected = stepNumber;
      markStripActive();
      renderPipeline();
      renderAttention();
    });
  }

  const matrixBtn = byId("gen-attn-matrix-btn");
  const dialog = byId("gen-matrix-dialog");
  if (matrixBtn && dialog) {
    matrixBtn.addEventListener("click", () => {
      const branch = activeBranch();
      if (!branch || !branch.final) return;
      setupHeatmap(branch);
      drawMatrix();
      dialog.showModal();
    });
  }
  const closeBtn = byId("gen-matrix-close");
  if (closeBtn && dialog) {
    closeBtn.addEventListener("click", () => dialog.close());
  }

  const layerSelect = byId("gen-layer-select");
  if (layerSelect) {
    layerSelect.addEventListener("change", () => {
      state.layer = Number(layerSelect.value) || 0;
      drawMatrix();
    });
  }
  const headSelect = byId("gen-head-select");
  if (headSelect) {
    headSelect.addEventListener("change", () => {
      state.head = Number(headSelect.value) || 0;
      drawMatrix();
    });
  }

  // resize 防抖：一帧内只重画一次。
  let resizeRaf = 0;
  window.addEventListener("resize", () => {
    if (resizeRaf) return;
    resizeRaf = requestAnimationFrame(() => {
      resizeRaf = 0;
      const branch = activeBranch();
      if (branch && branch.steps.length) renderAttention();
      const matrixDialog = byId("gen-matrix-dialog");
      if (matrixDialog && matrixDialog.open) drawMatrix();
    });
  });

  // 视图从隐藏切到显示时 clientWidth 才有效，需要重画 canvas。
  window.addEventListener("spa:view", (event) => {
    const visible = Boolean(event.detail && event.detail.route === "generate");
    state.visible = visible;
    if (visible) {
      renderAll();
      if (state.attentionDemo) state.attentionDemo.refresh();
    }
  });

  // 主题切换：概率色带、字体令牌与画布配色重取。
  on("theme", () => {
    probStops = null;
    invalidateFontTokens();
    state.pipelineKey = "";
    if (state.visible) renderAll();
  });
}

/**
 * 初始化生成页（加载 run、恢复 URL 状态）。
 */
export function initGenerationView() {
  const start = byId("gen-start-btn");
  if (!start) return;
  state.attentionDemo = initAttentionDemo({
    onNavigate: navigateAttentionQuery,
    onLatest: jumpAttentionLatest,
    onFollowChange: (value) => {
      if (value) jumpAttentionLatest();
      else {
        state.follow = false;
        const toggle = byId("gen-follow-toggle");
        if (toggle) toggle.checked = false;
      }
    },
  });
  bindEvents();
  initCompareMode();
  loadRuns();
  syncButtons();
}