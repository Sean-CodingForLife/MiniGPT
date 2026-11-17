/* 集中式 API 层：路径前缀、端点构造与统一请求管道只此一份。
 *
 * 请求层职责：
 *  - GET/POST/PUT/DELETE 统一超时（AbortController），支持外部 signal 级联取消；
 *  - 响应体读取也在超时保护内；HTTP 错误统一抛出后端 message；
 *  - ApiError 携带 status/type，调用方可按语义（409/429/504…）分支处理；
 *  - latest() 提供“只认最新一次”的竞态守卫：旧请求被主动 abort，
 *    即使响应晚到也会因序号过期而被丢弃。
 * 全部原生 ESM，无外部依赖；各视图模块不再散落 "/api/v1/..." 字面量。
 */

export const API_BASE = "/api/v1";

/** 拼接 API 路径：path("runs", id, "summary") -> "/api/v1/runs/<id>/summary"。
 * 各段自动 encodeURIComponent，调用方无需手动编码。 */
export function path(...segments) {
  const tail = segments
    .filter((segment) => segment !== undefined && segment !== null && segment !== "")
    .map((segment) => encodeURIComponent(String(segment)))
    .join("/");
  return tail ? `${API_BASE}/${tail}` : API_BASE;
}

/** 常用端点（带查询参数的用函数返回完整路径）。 */
export const endpoints = {
  health: () => path("health"),
  system: () => path("system"),
  presets: () => path("presets"),
  jobs: () => path("jobs"),
  jobStop: (id) => path("jobs", id, "stop"),
  runs: () => path("runs"),
  run: (id) => path("runs", id),
  runSummary: (id) => path("runs", id, "summary"),
  runStart: (id) => path("runs", id, "start"),
  runResume: (id) => path("runs", id, "resume"),
  runConsole: (id, limit) => `${path("runs", id, "console")}?limit=${limit}`,
  runMetrics: (id) => path("runs", id, "metrics"),
  runEvents: (id, after, limit) => `${path("runs", id, "events")}?after=${after}&limit=${limit}`,
  runSamples: (id, limit) => `${path("runs", id, "samples")}${limit ? `?limit=${limit}` : ""}`,
  runDelete: (id, force) => `${path("runs", id)}${force ? "?force=1" : ""}`,
  configs: () => path("configs"),
  config: (name) => path("configs", name),
  configValidate: () => path("configs", "validate"),
  datasets: () => path("datasets"),
  dataset: (name, limit) => `${path("datasets", name)}?limit=${limit}`,
  evaluations: () => path("evaluations"),
  evaluation: (runId, query) => `${path("evaluations", runId)}${query || ""}`,
  evaluate: () => path("evaluate"),
  evaluateStatus: (runId) => `${path("evaluate", "status")}?run_id=${encodeURIComponent(runId)}`,
  exports: () => path("exports"),
  export: (name) => path("exports", name),
  compare: (runs) => `${path("compare")}?runs=${encodeURIComponent(runs)}`,
  projects: () => path("projects"),
  project: (id) => path("projects", id),
  projectRuns: (id) => path("projects", id, "runs"),
  inferenceStream: () => path("inference", "stream"),
  inferenceCompare: () => path("inference", "compare"),
};

/* ------------------------------------------------------------------ 请求管道 */

export class ApiError extends Error {
  constructor(message, status, type) {
    super(message);
    this.name = "ApiError";
    this.status = status || 0;
    this.type = type || "";
  }
}

/** 把多个 AbortSignal 合成一个：任一触发即中止。返回 { signal, dispose }。 */
export function joinSignals(external) {
  const controller = new AbortController();
  if (!external) return { signal: controller.signal, dispose: () => {} };
  const onAbort = () => controller.abort();
  if (external.aborted) controller.abort();
  else external.addEventListener("abort", onAbort, { once: true });
  return {
    signal: controller.signal,
    dispose: () => external.removeEventListener("abort", onAbort),
  };
}

/** 统一 JSON 请求。
 * options: { method, body, timeoutMs=15000, signal }
 * 超时/取消/网络错误抛 ApiError；HTTP 非 2xx 抛后端 error 文案。 */
export async function requestJson(url, options = {}) {
  const timeoutMs = options.timeoutMs === undefined ? 15000 : options.timeoutMs;
  const controller = new AbortController();
  const timerHandle = timeoutMs > 0 ? window.setTimeout(() => controller.abort(), timeoutMs) : null;
  const external = options.signal;
  const forward = () => controller.abort();
  if (external) {
    if (external.aborted) controller.abort();
    else external.addEventListener("abort", forward, { once: true });
  }

  const init = {
    method: options.method || "GET",
    headers: { Accept: "application/json" },
    signal: controller.signal,
  };
  if (options.body !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.body);
  }
  try {
    let response;
    try {
      response = await fetch(url, init);
    } catch (error) {
      if (options.signal && options.signal.aborted) {
        throw new ApiError("请求已取消", 0, "cancelled");
      }
      if (controller.signal.aborted) {
        throw new ApiError(`请求超时（${Math.round(timeoutMs / 1000)}s）：${url}`, 0, "timeout");
      }
      throw new ApiError(`网络错误：${error && error.message ? error.message : error}`, 0, "network");
    }
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : null;
    } catch (error) {
      throw new ApiError(`HTTP ${response.status}: ${text.slice(0, 200)}`, response.status, "parse");
    }
    if (!response.ok) {
      const message = (payload && (payload.error || payload.message)) || `HTTP ${response.status}`;
      throw new ApiError(message, response.status, (payload && payload.type) || "");
    }
    return payload === null ? {} : payload;
  } finally {
    if (timerHandle !== null) window.clearTimeout(timerHandle);
    if (external) external.removeEventListener("abort", forward);
  }
}

export const getJson = (url, options) => requestJson(url, { ...options, method: "GET" });
export const postJson = (url, body, options) => requestJson(url, { ...options, method: "POST", body });
export const putJson = (url, body, options) => requestJson(url, { ...options, method: "PUT", body });
export const deleteJson = (url, options) => requestJson(url, { ...options, method: "DELETE" });

/** 竞态守卫：同一 key 只保留最新一次调用。
 * 用法：const latest = createLatest(); latest("run", (signal) => getJson(url, { signal }))
 * 旧调用被 abort；若旧响应仍晚到，结果以 { stale:true } 标记，调用方应忽略。 */
export function createLatest() {
  const counters = new Map();
  const controllers = new Map();
  return async function latest(key, task) {
    const seq = (counters.get(key) || 0) + 1;
    counters.set(key, seq);
    const previous = controllers.get(key);
    if (previous) previous.abort();
    const controller = new AbortController();
    controllers.set(key, controller);
    try {
      const result = await task(controller.signal);
      if (counters.get(key) !== seq) return { stale: true, value: null };
      return { stale: false, value: result };
    } finally {
      if (controllers.get(key) === controller) controllers.delete(key);
    }
  };
}