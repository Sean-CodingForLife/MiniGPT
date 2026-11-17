/* 训练配置编辑器：字段元数据 + 分组表单 + 表单/JSON 双模式。
 *
 * 从 dashboard.js 拆出的纯编辑层：不碰运行监控，只维护 ctx.editorConfig。
 * FIELD_GROUPS / FIELD_META 的书写格式是测试契约（tests/test_form_ui.py
 * 按字面块解析），保持 `var FIELD_GROUPS = [` / `var FIELD_META = {` 形式。
 * ctx 由编排层注入：{ editorConfig, runs, stage, initAlignRun,
 *                    requestInitAlign(runId), onEditorChanged() }
 */

import { byId, el } from "../ui.js";

// FIELD_GROUPS / FIELD_META 定义在 dashboard.js（测试按字面块解析），
// 通过 ctx.fieldGroups / ctx.fieldMeta 注入，保持单一数据源。

// 表单按阶段收敛：部署/服务器字段不属于训练配置，只在高级 JSON 里保留。
const DEPLOY_FIELDS = ["web_host", "web_port", "max_jobs", "web_dir", "project_root"];
const PRETRAIN_ONLY_FIELDS = ["dataset", "input_path", "vocab_extra", "download_dir", "zh_books", "source_url", "val_fraction", "test_fraction"];
const SFT_ONLY_FIELDS = ["init_from", "pretrain_data_dir", "pretrain_mix_ratio", "sft_input", "sft_source", "tokenizer_dir", "max_samples"];

const SPAN2_FIELDS = ["data_dir", "runs_dir", "download_dir", "input_path", "vocab_extra", "tokenizer_dir", "init_from", "pretrain_data_dir", "sample_prompt", "plugins", "project_root", "web_dir"];

/**
 * 深拷贝配置对象，避免编辑时污染源数据。
 */
function clone(value) {
  return JSON.parse(JSON.stringify(value));
}

/**
 * 创建配置表单控制器：字段分组渲染、校验与表单/JSON 双模式。
 */
export function createConfigForm(ctx) {
  const FIELD_GROUPS = ctx.fieldGroups;
  const FIELD_META = ctx.fieldMeta;
  let editorMode = "form";

  function syncJsonFromConfig() {
    const editor = byId("config-editor");
    if (editor) editor.value = JSON.stringify(ctx.editorConfig || {}, null, 2);
  }

  function setEditorConfig(config) {
    ctx.editorConfig = clone(config || {});
    renderConfigForm();
    syncJsonFromConfig();
  }

  function setEditorFromPreset(name) {
    const preset = ctx.presets[name];
    if (!preset) return;
    setEditorConfig(preset);
  }

  function setEditorFromConfig(config) {
    if (!config) return;
    // 合并进当前选中的预设，保证表单字段完整。
    const presetSelect = byId("preset-select");
    const base = presetSelect && ctx.presets[presetSelect.value] ? ctx.presets[presetSelect.value] : {};
    setEditorConfig(Object.assign({}, clone(base), clone(config)));
  }

  function fieldType(name, value, meta) {
    if (meta && meta.type) return meta.type;
    if (typeof value === "boolean") return "bool";
    if (typeof value === "number") return "number";
    return "text";
  }

  function renderFormField(name) {
    const meta = FIELD_META[name] || { label: name };
    const value = ctx.editorConfig[name];
    const type = fieldType(name, value, meta);
    const wrapper = el("label", "form-field");
    const label = el("span", "form-label", meta.label || name);
    label.title = meta.help || name;
    wrapper.appendChild(label);

    let input;
    if (type === "bool") {
      wrapper.className += " form-check";
      input = el("input");
      input.type = "checkbox";
      input.checked = Boolean(value);
    } else if (type === "select") {
      input = el("select");
      const options = (meta.options || []).slice();
      if (value !== undefined && options.indexOf(value) < 0) options.push(value);
      options.forEach((optionValue) => {
        const option = el("option", null, optionValue);
        option.value = optionValue;
        input.appendChild(option);
      });
      input.value = value === undefined ? "" : value;
    } else if (type === "number") {
      input = el("input");
      input.type = "number";
      input.step = meta.step === undefined ? "any" : String(meta.step);
      if (meta.min !== undefined) input.min = String(meta.min);
      if (meta.max !== undefined) input.max = String(meta.max);
      input.value = value === undefined || value === null ? "" : String(value);
    } else {
      input = el("input");
      input.type = "text";
      input.value = value === undefined || value === null ? "" : String(value);
    }
    input.dataset.field = name;
    input.autocomplete = "off";
    wrapper.appendChild(input);
    if (meta.help) wrapper.appendChild(el("small", "field-help", meta.help));
    if (SPAN2_FIELDS.indexOf(name) >= 0) wrapper.className += " span-2";
    return wrapper;
  }

  function renderConfigForm() {
    const container = byId("config-form");
    if (!container) return;
    container.replaceChildren();
    const config = ctx.editorConfig || {};
    const known = {};
    FIELD_GROUPS.forEach((group) => group.fields.forEach((name) => { known[name] = true; }));
    const stage = String(config.stage || "pretrain").toLowerCase() === "sft" ? "sft" : "pretrain";
    function visibleField(name) {
      if (DEPLOY_FIELDS.indexOf(name) >= 0) return false;
      if (stage === "pretrain" && SFT_ONLY_FIELDS.indexOf(name) >= 0) return false;
      if (stage === "sft" && PRETRAIN_ONLY_FIELDS.indexOf(name) >= 0) return false;
      return true;
    }
    let groups = FIELD_GROUPS.map((group) => {
      // 部署字段不属于训练配置：分组保留（完整性校验），表单不渲染。
      if (group.title === "Web 与部署") return null;
      let open = group.open;
      let title = group.title;
      if (group.title === "SFT") {
        title = "SFT 初始化与数据";
        open = stage === "sft";
      }
      if (group.title === "数据准备") {
        title = stage === "sft" ? "SFT 数据准备" : "预训练语料准备";
      }
      const fields = group.fields.filter(visibleField);
      if (!fields.length) return null;
      return { title, open, fields };
    }).filter(Boolean);
    // SFT 页把初始化组提到最前，先看“从哪个 checkpoint 继续”。
    if (stage === "sft") {
      groups = groups.slice().sort((a, b) => {
        const aSft = a.title === "SFT 初始化与数据" ? 0 : 1;
        const bSft = b.title === "SFT 初始化与数据" ? 0 : 1;
        return aSft - bSft;
      });
    }
    const rest = Object.keys(FIELD_META).filter((name) => !known[name] && visibleField(name));
    Object.keys(config).forEach((name) => {
      if (!known[name] && visibleField(name) && rest.indexOf(name) < 0) rest.push(name);
    });
    if (rest.length) groups.push({ title: "其他", open: false, fields: rest });
    const scopeNote = byId("config-scope-note");
    if (scopeNote) {
      scopeNote.textContent = stage === "sft"
        ? "当前为 SFT 配置：只显示 SFT 相关字段；预训练语料与部署字段不在此列（高级 JSON 可改全部字段）。"
        : "当前为预训练配置：只显示预训练相关字段；SFT 与部署字段不在此列（高级 JSON 可改全部字段）。";
    }
    groups.forEach((group) => {
      const details = el("details", "form-group");
      details.open = Boolean(group.open);
      const summary = document.createElement("summary");
      summary.appendChild(el("span", null, group.title));
      summary.appendChild(el("span", "group-count", String(group.fields.length)));
      details.appendChild(summary);
      const grid = el("div", "form-grid");
      if (stage === "sft" && group.title === "SFT 初始化与数据") {
        grid.appendChild(renderInitAlignBox());
      }
      group.fields.forEach((name) => {
        if (!(name in config) && !(name in FIELD_META)) return;
        grid.appendChild(renderFormField(name));
      });
      details.appendChild(grid);
      container.appendChild(details);
    });
  }

  // init_from 对齐控件：从已完成预训练运行选择 checkpoint，填路径并自动对齐相关字段。
  function completedPretrainRuns() {
    return (ctx.runs || []).filter((run) => {
      if (String(run.run_id || "").charAt(0) === "_") return false;
      if (String(run.stage || "pretrain").toLowerCase() !== "pretrain") return false;
      const status = run.display_status || run.status || "";
      return status === "completed" || run.has_best;
    });
  }

  function renderInitAlignBox() {
    const box = el("div", "init-align-box");
    const field = el("label", "field");
    field.appendChild(el("span", null, "从已完成预训练运行选择"));
    const select = el("select");
    select.id = "init-from-run-select";
    const placeholder = el("option", null, "（手动填写 init_from）");
    placeholder.value = "";
    select.appendChild(placeholder);
    completedPretrainRuns().forEach((run) => {
      const label = run.run_id + (run.step !== undefined && run.step !== null ? ` · ${run.step}${run.max_steps ? ` / ${run.max_steps}` : ""}` : "");
      const option = el("option", null, label);
      option.value = run.run_id;
      select.appendChild(option);
    });
    field.appendChild(select);
    box.appendChild(field);
    box.appendChild(el("span", "init-align-note", "选中后填入 best.pt，并按该运行配置对齐模型结构、tokenizer_dir、pretrain_data_dir；仍可手改。"));
    if (ctx.initAlignRun) select.value = ctx.initAlignRun;
    return box;
  }

  /** 用某个预训练运行的 checkpoint 与配置对齐 SFT 初始化。
   *  带编辑代号守卫：响应晚到时若编辑器已被切换，不再覆盖。 */
  function applyInitFromRun(runId) {
    ctx.initAlignRun = runId || "";
    if (!runId) return;
    const run = (ctx.runs || []).find((item) => item.run_id === runId);
    if (!run) return;
    const config = ctx.editorConfig || {};
    const runDir = run.run_dir || `runs/${runId}`;
    config.init_from = `${runDir}/${run.best_checkpoint || "best.pt"}`;
    const generation = ctx.editorGeneration();
    ctx.requestInitAlign(runId)
      .then((summary) => {
        if (ctx.editorGeneration() !== generation) return;
        const source = (summary && summary.config) || {};
        ["n_layer", "n_head", "n_embd", "block_size", "dropout", "bias", "tie_weights", "backend_type", "tokenizer_type"].forEach((key) => {
          if (source[key] !== undefined) config[key] = source[key];
        });
        if (source.data_dir) {
          config.pretrain_data_dir = source.data_dir;
          config.tokenizer_dir = source.data_dir;
        }
        setEditorConfig(config);
        ctx.feedback("config-feedback", `已用 ${runId} 的 checkpoint 与配置对齐 SFT 初始化`, false);
      })
      .catch(() => {
        if (ctx.editorGeneration() !== generation) return;
        setEditorConfig(config);
        ctx.feedback("config-feedback", `已填入 ${runId} 的 best.pt 路径；无法读取其配置，结构字段请手动核对`, true);
      });
  }

  function onFormChanged(event) {
    const target = event.target;
    if (target && target.id === "init-from-run-select") {
      applyInitFromRun(target.value);
      return;
    }
    const name = target && target.dataset ? target.dataset.field : null;
    if (!name) return;
    let value;
    if (target.type === "checkbox") value = target.checked;
    else if (target.type === "number") value = target.value === "" ? "" : Number(target.value);
    else value = target.value;
    ctx.editorConfig[name] = value;
    ctx.bumpEditorGeneration();
    syncJsonFromConfig();
  }

  function setEditorMode(mode) {
    const form = byId("config-form");
    const editor = byId("config-editor");
    const formButton = byId("editor-form-btn");
    const jsonButton = byId("editor-json-btn");
    const hint = byId("editor-mode-hint");
    if (!form || !editor) return;
    if (mode === "form") {
      // JSON -> 表单必须先解析；无效文本不静默丢弃。
      try {
        const parsed = JSON.parse(editor.value || "{}");
        if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
          throw new Error("配置必须是 JSON 对象");
        }
        ctx.editorConfig = clone(parsed);
        renderConfigForm();
      } catch (error) {
        ctx.feedback("config-feedback", `JSON 解析失败，仍停留在 JSON 模式: ${error.message}`, true);
        return;
      }
      form.hidden = false;
      editor.hidden = true;
      if (formButton) formButton.className = "btn btn-small btn-primary";
      if (jsonButton) jsonButton.className = "btn btn-small";
      if (hint) hint.textContent = "表单已同步（可滚动）；高级字段切到 JSON";
    } else {
      syncJsonFromConfig();
      form.hidden = true;
      editor.hidden = false;
      if (formButton) formButton.className = "btn btn-small";
      if (jsonButton) jsonButton.className = "btn btn-small btn-primary";
      if (hint) hint.textContent = "JSON 直接编辑；校验/保存/启动都会先解析";
    }
    editorMode = mode;
  }

  /** 读取当前编辑器内容（JSON 模式先解析并回写表单）。 */
  function parseEditor() {
    if (editorMode === "json") {
      const editor = byId("config-editor");
      const text = editor ? editor.value.trim() : "";
      if (!text) return {};
      let parsed = null;
      try {
        parsed = JSON.parse(text);
      } catch (error) {
        throw new Error(`配置 JSON 解析失败: ${error.message}`);
      }
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("配置必须是 JSON 对象");
      }
      ctx.editorConfig = clone(parsed);
      renderConfigForm();
      return parsed;
    }
    return clone(ctx.editorConfig || {});
  }

  function bind() {
    const formNode = byId("config-form");
    if (formNode) {
      formNode.addEventListener("input", onFormChanged);
      formNode.addEventListener("change", onFormChanged);
    }
    const formButton = byId("editor-form-btn");
    if (formButton) formButton.addEventListener("click", () => setEditorMode("form"));
    const jsonButton = byId("editor-json-btn");
    if (jsonButton) jsonButton.addEventListener("click", () => setEditorMode("json"));
  }

  return {
    bind,
    setEditorConfig,
    setEditorFromPreset,
    setEditorFromConfig,
    renderConfigForm,
    setEditorMode,
    parseEditor,
    applyInitFromRun,
    onFormChanged,
    get editorMode() { return editorMode; },
  };
}