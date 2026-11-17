/* N07 对比纯函数检查：加载真实 web/js/compare-core.js，
 * 验证 metrics → Chart.js dataset 转换与摘要表构建。输出 JSON 供 Python 断言。
 */
import assert from "node:assert/strict";
import {
  buildLossDatasets,
  buildPplDatasets,
  buildSummaryTable,
  colorForIndex,
} from "../web/js/compare-core.js";

const payload = {
  runs: [
    {
      run_id: "run-a",
      stage: "pretrain",
      display_status: "completed",
      metrics_count: 3,
      series: {
        step: [100, 200, 300],
        train_loss: [3.5, 2.5, null],
        val_loss: [3.0, 2.2, 1.9],
        val_ppl: [20.1, 9.0, 6.7],
      },
      report: { best_val_loss: 1.87, best_step: 280, steps: 300, elapsed_s: 12.5, status: "completed" },
    },
    {
      run_id: "run-b",
      stage: "sft",
      display_status: "completed",
      metrics_count: 2,
      series: {
        step: [100, 200],
        train_loss: [4.0, 3.1],
        val_loss: [3.8, 3.0],
        val_ppl: [44.7, 20.1],
      },
      report: null,
    },
  ],
};

const loss = buildLossDatasets(payload);
assert.equal(loss.length, 4, "2 run × (train+val) = 4 datasets");
assert.equal(loss[0].label, "run-a · train_loss");
assert.equal(loss[1].label, "run-a · val_loss");
assert.equal(loss[2].label, "run-b · train_loss");
assert.deepEqual(loss[0].data, [
  { x: 100, y: 3.5 },
  { x: 200, y: 2.5 },
]); // null 值被跳过
assert.deepEqual(loss[1].data, [
  { x: 100, y: 3.0 },
  { x: 200, y: 2.2 },
  { x: 300, y: 1.9 },
]);
assert.ok(Array.isArray(loss[1].borderDash) && loss[1].borderDash.length > 0, "val 用虚线");
assert.equal(colorForIndex(0), colorForIndex(0));
assert.notEqual(loss[0].borderColor, loss[2].borderColor, "不同 run 不同颜色");

const ppl = buildPplDatasets(payload);
assert.equal(ppl.length, 2);
assert.equal(ppl[0].label, "run-a · val_ppl");
assert.deepEqual(ppl[1].data, [
  { x: 100, y: 44.7 },
  { x: 200, y: 20.1 },
]);

const summary = buildSummaryTable(payload);
assert.deepEqual(summary.headers, ["指标", "run-a", "run-b"]);
const byLabel = Object.fromEntries(summary.rows.map((row) => [row[0], row.slice(1)]));
assert.deepEqual(byLabel["阶段"], ["pretrain", "sft"]);
assert.deepEqual(byLabel["状态"], ["completed", "completed"]);
assert.deepEqual(byLabel["best_val_loss"], ["1.8700", "—"]);
assert.deepEqual(byLabel["指标行数"], ["3", "2"]);

// 空输入不崩。
assert.deepEqual(buildLossDatasets({ runs: [] }), []);
assert.deepEqual(buildPplDatasets(null), []);
assert.deepEqual(buildSummaryTable({}).headers, ["指标"]);

console.log(
  JSON.stringify({
    ok: true,
    lossDatasetCount: loss.length,
    pplDatasetCount: ppl.length,
    skippedNull: 3 - loss[0].data.length,
    summaryHeaders: summary.headers,
  }),
);