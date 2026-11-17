"""把一次 MiniGPT 生成过程导出为自包含的 HTML 可视化页面。

页面内容：

* prompt 与生成参数；
* 每个生成 token 一张卡片：本次采样概率、top-k 候选概率，以及最近上下文
  上逐层/逐头的注意力；
* 最终一步的注意力热图（可选 layer / head）；
* 最终隐藏状态的紧凑摘要。

只使用 Python 标准库 + PyTorch，输出单个本地 HTML 文件（CSS/JS 全部内联），
不发起任何网络请求，也不依赖外部前端库。

用法::

    python visualize.py --checkpoint runs/demo-en-sft/best.pt --prompt "Hello"
    python visualize.py --checkpoint runs/demo-zh-pretrain/best.pt --prompt "三国演义：" --max_new_tokens 30
"""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import _bootstrap  # noqa: F401
from minigpt.config import overlay_config
from minigpt.inference import load_bundle
from minigpt.utils import seed_everything


# 追踪采集逻辑已下沉到包内 minigpt.render（恢复包可移植性）；
# 这里保留同名导出，CLI 与既有调用方不受影响。
from minigpt.render import (  # noqa: E402
    _hidden_summary,
    _prompt_for,
    _sampling_with_tokens,
    collect_trace,
    iter_trace,
)


def _esc(value: Any) -> str:
    """HTML 转义任意值，防止样本文本里的标签破坏页面结构。"""
    return html.escape(str(value), quote=True)


def _bar_rows(step: dict[str, Any]) -> str:
    """把一步的 top-k 候选渲染成横向条形图。

    条宽以本步最高概率为基准归一化，选中项额外加 ``chosen`` 类高亮。
    """
    topk = step.get("topk") or []
    if not topk:
        return '<div class="muted">no top-k candidates (trace_top_k=0)</div>'
    maximum = max([step["prob"]] + [candidate["prob"] for candidate in topk] + [1e-12])
    rows = []
    for candidate in topk:
        width = max(0.0, min(100.0, candidate["prob"] / maximum * 100.0))
        chosen = " chosen" if candidate.get("chosen") else ""
        rows.append(
            f'<div class="bar-row{chosen}">'
            f'<span class="bar-token">{_esc(candidate["token"])}</span>'
            f'<span class="bar-track"><span class="bar-fill" style="width:{width:.2f}%"></span></span>'
            f'<span class="bar-prob">{candidate["prob"]:.4f}</span>'
            "</div>"
        )
    return "".join(rows)


def _attention_strip(step: dict[str, Any]) -> str:
    """渲染一步内所有 layer/head 的注意力色条（每格对应一个上下文 token）。"""
    attention = step.get("attn_last") or []
    if not attention:
        return '<div class="muted">no attention trace</div>'
    context = step.get("context") or []
    rows = []
    for layer_index, heads in enumerate(attention):
        for head_index, values in enumerate(heads):
            maximum = max(values + [1e-12])
            cells = []
            for token_index, value in enumerate(values):
                alpha = min(1.0, value / maximum) if maximum else 0.0
                label = context[token_index] if token_index < len(context) else str(token_index)
                cells.append(
                    f'<span class="cell" style="background:rgba(88,166,255,{alpha:.3f})" '
                    f'title="t={token_index} token={_esc(label)} weight={value:.5f}"></span>'
                )
            rows.append(
                f'<div class="strip-row"><span class="strip-label">L{layer_index}H{head_index}</span>'
                f'<span class="strip-cells">{"".join(cells)}</span></div>'
            )
    return "".join(rows)


def _hidden_html(step: dict[str, Any]) -> str:
    """渲染最终隐藏状态的统计量与被选中的若干维度。"""
    hidden = step.get("hidden") or {}
    top = hidden.get("top") or []
    chips = " ".join(
        f'<span class="chip">d{item["dim"]}: {item["value"]:+.4f}</span>'
        for item in top
    )
    return (
        '<div class="hidden">'
        f'<span>hidden norm={hidden.get("norm", 0.0):.3f}</span>'
        f'<span>mean={hidden.get("mean", 0.0):+.4f}</span>'
        f'<span>std={hidden.get("std", 0.0):.4f}</span>'
        f'{chips}'
        "</div>"
    )


def _steps_html(data: dict[str, Any]) -> str:
    """把每个生成步骤拼成卡片序列（token、概率、候选条、注意力、隐藏状态）。"""
    cards = []
    for step in data["steps"]:
        finish = '<span class="finish">EOS</span>' if step.get("finished") else ""
        cards.append(
            '<article class="step">'
            f'<div class="step-head"><span class="idx">#{step["step"]}</span>'
            f'<code class="token">{_esc(step["token"])}</code>'
            f'<span class="prob">p={step["prob"]:.4f}</span>{finish}</div>'
            f'<div class="bars">{_bar_rows(step)}</div>'
            f'<div class="attention">{_attention_strip(step)}</div>'
            f'{_hidden_html(step)}'
            "</article>"
        )
    return "".join(cards)


CSS = """
/* 深色主题；颜色全部走 CSS 变量，页面无外部字体/样式请求。 */
:root {
  --bg: #0d1117; --panel: #161b22; --line: #30363d; --text: #c9d1d9;
  --muted: #8b949e; --blue: #58a6ff; --green: #3fb950; --yellow: #d29922;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--bg); color: var(--text);
  font: 14px/1.5 "Segoe UI", "Microsoft YaHei", system-ui, sans-serif; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 24px; }
h1 { margin: 0 0 8px; font-size: 24px; }
h2 { margin: 26px 0 10px; font-size: 17px; }
.meta { color: var(--muted); display: flex; flex-wrap: wrap; gap: 14px; }
.prompt { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px; white-space: pre-wrap; word-break: break-word; margin: 14px 0; }
.summary { color: var(--muted); margin-bottom: 14px; }
.step { background: var(--panel); border: 1px solid var(--line); border-radius: 8px;
  padding: 12px; margin: 10px 0; }
.step-head { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; }
.idx { color: var(--muted); font-family: ui-monospace, Consolas, monospace; }
.token { color: var(--yellow); background: #21262d; padding: 1px 6px; border-radius: 4px;
  font-family: ui-monospace, Consolas, monospace; }
.prob { color: var(--green); font-family: ui-monospace, Consolas, monospace; }
.finish { color: var(--muted); border: 1px solid var(--line); padding: 0 5px; border-radius: 4px; }
.bars { margin: 8px 0; max-width: 620px; }
.bar-row { display: flex; align-items: center; gap: 8px; margin: 2px 0; font-family: ui-monospace, Consolas, monospace; font-size: 12px; }
.bar-row.chosen .bar-token, .bar-row.chosen .bar-prob { color: var(--green); }
.bar-token { width: 70px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-track { flex: 1; height: 8px; background: #21262d; border-radius: 4px; overflow: hidden; }
.bar-fill { display: block; height: 100%; background: var(--blue); }
.bar-prob { width: 66px; text-align: right; color: var(--muted); }
.attention { margin-top: 10px; }
.strip-row { display: flex; align-items: center; gap: 6px; margin: 2px 0; }
.strip-label { width: 34px; color: var(--muted); font: 10px ui-monospace, Consolas, monospace; }
.strip-cells { display: flex; gap: 1px; }
.cell { width: 12px; height: 12px; border-radius: 2px; background: #21262d; display: inline-block; }
.hidden { margin-top: 8px; color: var(--muted); font: 11px ui-monospace, Consolas, monospace; display: flex; flex-wrap: wrap; gap: 10px; }
.chip { color: #a5d6ff; }
.muted { color: var(--muted); }
.heat-controls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; color: var(--muted); }
.heat-controls select { background: var(--panel); color: var(--text); border: 1px solid var(--line); border-radius: 5px; padding: 3px 6px; }
canvas { width: 100%; max-width: 760px; background: #0b0f14; border: 1px solid var(--line); border-radius: 8px; margin-top: 10px; }
.legend { display: flex; align-items: center; gap: 8px; color: var(--muted); margin-top: 6px; font-size: 12px; }
.legend-bar { width: 160px; height: 10px; border-radius: 5px; background: linear-gradient(90deg, rgba(88,166,255,0), rgba(88,166,255,1)); }
.footer { color: var(--muted); margin-top: 30px; font-size: 12px; }
"""

JS = """
// 追踪数据由 Python 以 <script type="application/json"> 内联注入，避免 XSS 与额外请求。
const DATA = JSON.parse(document.getElementById("trace-data").textContent);
const layerSelect = document.getElementById("layer-select");
const headSelect = document.getElementById("head-select");
const canvas = document.getElementById("heatmap");
const ctx = canvas.getContext("2d");

function addOptions(select, count, prefix) {
  // 按 layer/head 数量动态生成 <option>，避免页面模板里写死。
  for (let i = 0; i < count; i++) {
    const option = document.createElement("option");
    option.value = String(i);
    option.textContent = prefix + i;
    select.appendChild(option);
  }
}

function drawHeatmap() {
  // 取最终一步的注意力矩阵，按当前 layer/head 选择绘制到 canvas。
  const attention = DATA.final.attention || [];
  const context = DATA.final.context || [];
  if (!attention.length) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#8b949e";
    ctx.fillText("no attention trace", 20, 30);
    return;
  }
  const layer = Math.max(0, Math.min(Number(layerSelect.value), attention.length - 1));
  const heads = attention[layer];
  const head = Math.max(0, Math.min(Number(headSelect.value), heads.length - 1));
  const matrix = heads[head];
  const n = matrix.length;
  const margin = 70;
  const size = Math.max(300, Math.min(canvas.width - margin - 18, canvas.height - margin - 18));
  const cell = size / Math.max(1, n);
  canvas.width = margin + size + 18;
  canvas.height = margin + size + 18;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.font = "11px ui-monospace, Consolas, monospace";
  ctx.fillStyle = "#8b949e";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  const labelEvery = Math.max(1, Math.ceil(n / 16));
  function tokenLabel(index) {
    const value = String(context[index] === undefined ? index : context[index]);
    if (value === " ") return "␠";
    if (value === "\\n") return "↵";
    if (value === "'\\t'") return "⇥";
    return value.replace(/^'|'$/g, "").slice(0, 5);
  }
  for (let i = 0; i < n; i++) {
    const y = margin + i * cell + cell / 2;
    if (i % labelEvery === 0) {
      ctx.fillText(tokenLabel(i), margin - 6, y);
    }
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "bottom";
  for (let j = 0; j < n; j++) {
    if (j % labelEvery !== 0) continue;
    const x = margin + j * cell + cell / 2;
    ctx.save();
    ctx.translate(x, margin - 4);
    ctx.rotate(-Math.PI / 4);
    ctx.textAlign = "left";
    ctx.fillText(tokenLabel(j), 0, 0);
    ctx.restore();
  }
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const value = Number(matrix[i][j] || 0);
      const alpha = Math.min(1, Math.pow(value, 0.6) * 1.7);
      ctx.fillStyle = "rgba(88,166,255," + alpha.toFixed(3) + ")";
      ctx.fillRect(margin + j * cell, margin + i * cell, cell - 0.5, cell - 0.5);
    }
  }
  ctx.strokeStyle = "#30363d";
  ctx.strokeRect(margin, margin, size, size);
  ctx.fillStyle = "#8b949e";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText("query (行) / key (列)，颜色越亮注意力越大", margin, 8);
}

function setupControls() {
  // 没有注意力数据时（如 pure 引擎）不建下拉框，页面只展示文本摘要。
  const attention = DATA.final.attention || [];
  if (!attention.length) return;
  addOptions(layerSelect, attention.length, "layer ");
  addOptions(headSelect, attention[0].length, "head ");
  layerSelect.addEventListener("change", drawHeatmap);
  headSelect.addEventListener("change", drawHeatmap);
}

document.getElementById("prompt").textContent = DATA.prompt;
document.getElementById("meta").textContent =
  "checkpoint=" + (DATA.checkpoint || "?") + " · stage=" + DATA.stage +
  " · device=" + DATA.device + " · params=" + DATA.parameters + " · " +
  (DATA.model.n_layer + " layers / " + DATA.model.n_head + " heads / " +
   DATA.model.n_embd + " dim / block " + DATA.model.block_size);
document.getElementById("continuation").textContent = DATA.continuation || "(empty)";
setupControls();
drawHeatmap();
"""


def build_html(data: dict[str, Any], title: str = "MiniGPT 生成过程") -> str:
    """生成完整 HTML 页面。

    追踪数据以 JSON 内联到 ``<script type="application/json">`` 中，并把 ``<``
    转义为 ``\\u003c``，避免样本文本里的 ``</script>`` 提前闭合脚本标签。
    """
    trace_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <div class="meta" id="meta"></div>
  <div class="summary" id="continuation"></div>
  <pre class="prompt" id="prompt"></pre>
  <h2>逐 token 概率</h2>
  <div class="muted">绿色是本次采样选中的 token；条形是 softmax 后的相对概率。</div>
  {_steps_html(data)}
  <h2>最终一步注意力热图</h2>
  <div class="heat-controls">
    <label>layer <select id="layer-select"></select></label>
    <label>head <select id="head-select"></select></label>
    <span>最后 {len(data.get("final", {}).get("context", []))} 个 token</span>
  </div>
  <canvas id="heatmap" width="760" height="760"></canvas>
  <div class="legend"><span>0</span><span class="legend-bar"></span><span>max</span></div>
  <div class="footer">本文件由 visualize.py 生成；内联 CSS/JS，不请求任何外部资源。</div>
</div>
<script id="trace-data" type="application/json">{trace_json}</script>
<script>{JS}</script>
</body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    """构造可视化命令行参数（公共配置项由 overlay_config 补齐）。"""
    parser = argparse.ArgumentParser(description="Visualize one MiniGPT generation as local HTML")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max_new_tokens", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--trace_top_k", type=int, default=5)
    parser.add_argument("--max_context", type=int, default=32)
    parser.add_argument("--template", default="auto", choices=["auto", "qa", "raw"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--out", default=None, help="output HTML (default: <checkpoint dir>/generation.html)")
    return parser


def main(argv: list[str] | None = None) -> int:
    """采样一次生成、采集追踪数据并写出 HTML，返回进程退出码。

    默认输出到 ``<checkpoint 所在目录>/generation.html``。
    """
    args = overlay_config(build_parser(), argv, stage="pretrain", field_map={"prompt": "sample_prompt"})
    if args.max_context < 1:
        print("--max_context must be >= 1", file=sys.stderr)
        return 2
    seed_everything(args.seed)
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    bundle = load_bundle(checkpoint, device=args.device)
    data = collect_trace(
        bundle,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        trace_top_k=args.trace_top_k,
        max_context=args.max_context,
        template=args.template,
    )
    data["checkpoint"] = str(checkpoint.name)
    output = Path(args.out).expanduser().resolve() if args.out else checkpoint.parent / "generation.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(data), encoding="utf-8")
    print(f"[visualize] wrote {output}")
    print(f"[visualize] steps={len(data['steps'])} final_step={data['final']['step']}")
    print(f"[visualize] continuation={data['continuation']!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
