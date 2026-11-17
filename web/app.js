/* MiniGPT 控制台前端入口（原生 ESM，无构建链、无 CDN、页面不发起任何外部网络请求）。
 *
 * 在 SPA 中的位置：整站唯一入口；index.html 只加载 /static/vendor/chart.umd.min.js 与本文件。
 * 模块分工（每个阶段视图一个模块，进入视图时才按需拉数据）：
 *   store.js      → 全局 UI 状态（主题 / 暂停 / 路由切片）
 *   dashboard.js  → 训练工作区（预训练 / SFT 共用，按阶段过滤）
 *   data.js       → 数据（只读）
 *   eval.js       → 评测（只读 + 面板内评估）
 *   export.js     → 导出（只读）
 *   generation.js → 生成可视化（实时流式 + 注意力重点视图）
 *   compare.js    → 多运行对比
 *   projects.js   → 项目（数据→预训练→SFT→评测→导出的全流程容器）
 *   router.js     → hash 路由（#/projects #/data #/pretrain #/sft #/eval #/generate #/compare #/export）
 *
 * 初始化顺序：全局状态（主题/暂停）→ 训练工作区 → 各视图注册 spa:view
 * 监听 → 启动路由。这样直接打开任意深链时首屏事件不会丢。
 *
 * 关键约束：
 *  - dashboard.js 在模块求值阶段就会执行 bootstrap()，因此必须先导入它、再启动路由；
 *  - 各视图监听的是 window 上的 "spa:view" 自定义事件（router.js 派发），
 *    所以所有 init*View() 都必须早于路由启动，否则首屏深链收不到通知。
 */
import { initStore } from "./js/store.js";
import "./js/dashboard.js";
import { initCompareView } from "./js/compare.js";
import { initDataView } from "./js/data.js";
import { initEvalView } from "./js/eval.js";
import { initExportView } from "./js/export.js";
import { initGenerationView } from "./js/generation.js";
import { initProjectsView } from "./js/projects.js";
import { initRouter } from "./js/router.js";

initStore();
// 训练工作区没有导出 init 钩子：导入 dashboard.js 时已自行 bootstrap 并启动轮询。
initDataView();
initEvalView();
initExportView();
initGenerationView();
initCompareView();
initProjectsView();
// 最后启动路由：applyRoute() 会立刻派发第一个 spa:view，把当前 hash 通知给上面注册好的视图。
initRouter();
