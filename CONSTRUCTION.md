# MiniGPT 施工合同

- 状态：当前实现
- 作用：本项目范围、验收、依赖、数据、接口、目录的唯一依据；代码、测试、README/MANUAL 与本文件保持一致。

---

## 0. 文档结构（防止说明散落）

| 文件 | 定位 | 规则 |
|---|---|---|
| `CONSTRUCTION.md` | 施工合同：范围、验收、决策、契约、风险 | 当前范围与验收的唯一依据 |
| `MANUAL.md` | 唯一长手册：零基础概念、操作、代码导读、实验、故障、术语、附录 | 所有学习/使用说明都放这里 |
| `README.md` | 短入口：安装、5 分钟命令、两个文档链接 | 不超过 120 行 |
| `runs/<run_id>/` | 实验产物：配置、指标、事件、checkpoint、报告 | 不是文档，不参与“文档散落”判断 |

不创建 `docs/*.md` 或其他长篇说明文件；代码内部只写模块/函数 docstring 和必要注释。

---

## 1. 目标与非目标

### 1.1 目标

交付一套可复现、可观察、可恢复、可扩展、可移植的字符级 MiniGPT 方法：

- 数据准备 → 字符级 tokenizer → 预训练 → SFT → 评估 → 对话；
- 训练过程可遥测，Web 控制面板可启动/停止任务、在线编辑并校验配置；
- 中英文双主 demo；
- 合成 SFT 数据 + 公开中文指令数据两条路径都实际跑通；
- 详细但集中的手册、代码注释、测试和实验记录；
- 所有路径/设备/端口/数据源可配置，不假设本机是唯一机器；
- 通过 `ModelBackend` 协议隔离计算后端，当前提供 `TorchBackend` 与 `PureBackend`（纯 Python，见 A18）。

### 1.2 当前非目标

- 纯 Python Tensor/autograd 后端已实现（A18，tiny 训练/推理）；纯后端的大模型训练（0.81M/10M）不实用，不在本机进行。
- 不在本机全量训练 10M 模型。
- 不承诺“可用聊天助手”质量。
- 不做 BPE/WordPiece、多模态、RLHF、分布式训练、集群调度。
- 不做公网鉴权/多租户。
- 不依赖任何预训练权重或外部 API。

---

## 2. 当前范围与验收边界

当前实现覆盖：

- PyTorch 全流程：数据准备、字符级 tokenizer、预训练、SFT、评估、对话、Web 控制面板、checkpoint 恢复、模型包导出；
- 中英文双主 demo；
- 合成 SFT 与公开中文指令数据两条路径；
- 10M 中英文配置可实例化、校验参数、跑 2–3 步冒烟，不在本机全量训练。

当前不包含：纯后端的大模型训练（0.81M/10M 在纯 Python 下不实用）、严格 `transformers` 兼容、BPE/子词 tokenizer、HF Hub 上传。（纯 Python Tensor/autograd 后端本身已实现，见 A18。）

**当前双主 demo：**

- 英文：Tiny Shakespeare，0.8M debug，3000 步预训练 + 300 步合成 SFT；初始 loss ≈ 4.22，最终 val 目标 < 2.2。
- 中文：Project Gutenberg 公版语料，0.8M debug，500–1000 步预训练 + 300 步合成 SFT；初始 loss ≈ 8.4，按相对下降判断，不套用英文阈值。
- 公开 SFT：`shibing624/alpaca-zh` 采样 500–1000 条，150–300 步 SFT smoke，验证导入路径。

---

## 3. 当前验收矩阵

| # | 验收项 | 通过标准 |
|---|---|---|
| A1 | 测试 | `python -m unittest discover -s tests -v` 全通过 |
| A2 | 英文数据 | Tiny Shakespeare 校验、tokenizer、三分、`split_report.json` |
| A3 | 中文数据 | Gutenberg 下载/清洗/tokenizer/三分可跑通 |
| A4 | 英文训练 | 3000 步 debug 预训练 + 300 步 SFT + test 评估 + chat |
| A5 | 中文训练 | 500–1000 步 debug 预训练 + 300 步 SFT + 评估 + chat |
| A6 | SFT 双数据 | 中英文合成各 500 条 + Alpaca-ZH 采样均实际跑通 |
| A7 | 门禁/恢复 | mini-run 三门禁、resume 一致、早停/best、RNG 可复现 |
| A8 | 仪表盘 | 启动/停止真实训练、在线编辑/校验配置、实时曲线/样本/门禁 |
| A9 | 截图 | Edge headless 生成 `screenshots/dashboard.png`、`stage-data.png`；内置浏览器生成 `spa-generate.png`，并人工检查 |
| A10 | 移植 | 任意 cwd、配置优先级、checkpoint 移动、无 torch 看板、绝对路径/依赖扫描 |
| A11 | 文档 | `CONSTRUCTION.md`/`MANUAL.md`/`README.md` 齐备互链；docstring/章节测试通过 |
| A12 | 10M 校验 | 中英文配置参数校验 + 2–3 步冒烟；命令/估算写入文档；本机不全量训练 |
| A13 | 可扩展 | `ModelBackend` 协议、`TorchBackend`、registry、`--plugins`、扩展测试 |
| A14 | 可复现 | 所有 demo 权重/指标/截图可由脚本重新生成，不依赖外部权重/API |
| A15 | 可交换包 | `export.py` 生成 HF 风格自有 bundle（config/权重/tokenizer/模型卡/run.py/serve.py/MANIFEST）；从临时 cwd 运行、`--verify` 通过；不声称 transformers 兼容 |
| A16 | 生成可视化 | `chat.py --stream` 输出逐 token 概率/top-k；`visualize.py` 生成单文件 HTML（概率条、注意力热图、hidden 摘要）；三者共用采样过程 trace（T=1/温度/top-k/top-p/累积轴/`u`）；不请求外部资源 |
| A17 | SPA / 面板 | 原生 ESM + 六阶段 hash SPA（`#/data`、`#/pretrain`、`#/sft`、`#/eval`、`#/generate`、`#/export`，`#/dashboard` 重定向到 `#/pretrain`）；数据/评测/导出为只读扫描接口；生成视图经 `POST /api/v1/inference/stream`（NDJSON）逐 token 渲染；看板后端保持无 torch |
| A18 | 纯后端 | `minigpt/pure/`：自研 Tensor/autograd/AdamW、纯 GPT、`PureBackend`、`.mgckpt`；与 torch 同权重对拍；`--backend_type pure` 训练 tiny 预设，`chat.py`/`evaluate.py` 按后缀分派；bundle `weights.f32` + `run.py --engine pure` 无 torch 运行 |

---

## 4. 依赖政策

### 4.1 当前实现

| 层 | 允许 | 禁止 |
|---|---|---|
| AI/训练/数据 | Python 标准库 + PyTorch | numpy、pytest、transformers、datasets、accelerate、deepspeed、tokenizers、tqdm、matplotlib、pandas |
| Web 后端 | Python 标准库 | Flask、FastAPI、aiohttp、uvicorn、torch |
| 前端 | Chart.js 4.4.7（本地 vendor）+ 原生 JS/CSS | CDN、运行时网络、Node 构建链 |
| 测试 | `unittest` | pytest 及其他测试框架 |

- `requirements.txt` 只有 `torch>=2.1,<3`（唯一运行时依赖；本机测试 2.15.0）。
- 数据二进制读写用 `array`/`struct`，显式小端；不使用 numpy。
- 后端不得在模块顶层导入 torch；看板必须能在无 PyTorch 的机器上运行。
- 新增后端不得引入新依赖。

### 4.2 依赖可替换性

- 定义 `ModelBackend` 协议：构建模型、构建优化器、训练步、评估步、state_dict/load、generate、backend 名称/版本。
- 当前实现 `TorchBackend` 与 `PureBackend`；Trainer/数据/checkpoint/遥测/dashboard 只依赖协议。
- Registry：`TOKENIZERS / MODELS / DATA_SOURCES / CALLBACKS / BACKENDS`；支持 `--plugins path/to/module.py`。
- Checkpoint 记录 backend 名称与 format_version；不做跨后端 checkpoint 互转（torch `.pt` 与 pure `.mgckpt` 不直接互换），但 export 会把 torch 权重导出为 pure 引擎可用的 `weights.f32`。

---

## 5. 硬件、语言、数据与许可

| 项目 | 结论 | 处理 |
|---|---|---|
| 本机 CPU | i5-13400F，10 核 16 线程，32GB | 当前主验证平台 |
| 本机 GPU | Quadro K2200 4GB，驱动 446.14 | 现代 PyTorch 不可用；不引入 EOL 栈；GPU 命令供其他机器 |
| 英文语料 | Tiny Shakespeare，1,115,394 字节，65 字符 | jsDelivr 主源 + 镜像回退；SHA-256 `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed` |
| 中文语料 | Gutenberg 23950/23962/24264/23863/23839/7337 等 | UTF-8、繁体；重试 + START/END 完整性校验；`--input` 兜底 |
| 合成 SFT | 中英文各 500 条 | 脚本生成，提交仓库；仅管道验证 |
| 公开 SFT | `shibing624/alpaca-zh` 的 `alpaca_gpt4_data_zh.json` | 约 35.1MB，48,818 条；字段 instruction/input/output；CC BY-NC 4.0 仅研究；只缓存不提交；采样 500–1000 |
| 语言 | 中英文双主 demo | 共用代码/配置/指标框架；指标独立解释 |

**数据规则：**

- 预训练词表默认只用 train 切分构建；特殊 token `<unk>/<eos>/<pad>`；val/test 未登录字符计入 `<unk>` 并报告 unk rate。
- 允许通过 `vocab_extra` 声明额外的 **SFT train** 文本（UTF-8 文本或 JSONL 的 q/a；绝不能包含 SFT val/test）参与词表构建；目的是解决“繁体预训练语料 + 简体 SFT”的字符不匹配。额外文件与哈希写入 `meta.json`。
- 预训练三分 90/5/5 连续切分；SFT 三分 90/5/5 按 group（缺省归一化问题）分组，禁止同组跨集合。
- SFT 格式：`Q: {q}\nA: {a}<eos>`；question/pad 标签 -100，仅 answer+EOS 计 loss。
- SFT 默认混入 15% 预训练 batch。
- Test 集只在最终 best 上评估一次；训练/调参只使用 val。

---

## 6. 配置、路径与移植

### 6.1 优先级

**默认值 < `config.json` < `MINIGPT_*` 环境变量 < 命令行参数。**

- 支持 `--preset`、`--config`、所有字段 CLI 覆盖。
- `MINIGPT_ROOT` 可覆盖项目根；默认由代码位置自动探测。
- 路径字段：默认相对项目根；绝对路径也接受；`pathlib` 统一处理。
- 设备/dtype/线程/端口/host/run_id/数据源 URL/中文书单/采样数/prompt/seed/模型尺寸/步数/lr 全部可覆盖。
- 不硬编码：用户名、`C:\Users\...`、`/home/...`、固定机器名、固定 GPU、固定端口、固定 run 目录。
- 跨平台：显式 UTF-8、`os.replace` 原子写、无 Windows 专用 API、无 shell 命令。

### 6.2 Checkpoint 移植

- checkpoint 内嵌 tokenizer state（chars/specials），`chat.py` 只需单个 checkpoint。
- `torch.load(..., map_location=...)`；`--device auto`。
- 路径尽量存项目相对路径；数据目录缺失时 `--data_dir` 覆盖。
- `format_version` 校验；不兼容时明确报错。

### 6.3 仪表盘移植

- 后端纯标准库，不导入 torch；训练机写 `runs/`，看板机只读。
- 默认 `127.0.0.1`；`--host 0.0.0.0` 仅供内网，手册警告无鉴权。
- 静态资源相对路径；Chart.js 本地 vendor，无 CDN。
- 支持从任意 cwd 启动；`--runs_dir`/`--web_dir` 可配置。
- 手册给局域网、反向代理、Docker 示例。

---

## 7. 目录与接口契约

### 7.1 目录

```text
MiniGPT/
├─ CONSTRUCTION.md
├─ MANUAL.md
├─ README.md
├─ config.example.json
├─ requirements.txt
├─ .gitignore
├─ _bootstrap.py
├─ minigpt/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ registry.py
│  ├─ backends/{base,torch_backend}.py
│  ├─ tokenization/{base,char}.py
│  ├─ data/{download,sources,formatters,binary}.py
│  ├─ models/gpt.py
│  ├─ training/{trainer,checkpoint,callbacks,early_stop,job_manager}.py
│  ├─ eval/metrics.py
│  ├─ bundle.py                 # MiniGPT Bundle v1 导出/校验
│  ├─ pure/{tensor,adamw,gpt,backend,checkpoint,inference,weights}.py  # Stage 2：自研张量/自动微分/纯后端
│  └─ web/{server,stages,inference_worker}.py   # 标准库 API + 只读阶段扫描 + PyTorch 推理子进程
├─ web/{index.html,app.js,style.css,js/{dashboard,router,ui,data,eval,export,generation}.js,vendor/chart.umd.min.js,CHARTJS_LICENSE.md}
├─ bundle_templates/{run.py,serve.py}   # 包内自包含运行器模板
├─ prepare.py
├─ prepare_sft.py
├─ pretrain.py
├─ sft.py
├─ chat.py
├─ evaluate.py
├─ observe.py
├─ visualize.py
├─ export.py
├─ dashboard.py
├─ scripts/build_demo_data.py
├─ data/{demo_sft_en.jsonl,demo_sft_zh.jsonl,demo_eval_en.jsonl,demo_eval_zh.jsonl}
├─ screenshots/
├─ dist/                       # 导出的模型包/zip
├─ runs/
├─ LICENSE                     # 代码 MIT
└─ tests/
```

### 7.2 Tokenizer 契约

- `CharTokenizer.build(text) -> CharTokenizer`（仅 train 文本）
- `vocab_size`, `encode(text, add_eos=False) -> list[int]`, `decode(ids, skip_special=True) -> str`
- `state_dict() -> dict`, `from_state(dict) -> CharTokenizer`
- `save(dir)`, `load(dir)`
- 特殊 token：`<unk>`, `<eos>`, `<pad>`；id 稳定写入 state。

### 7.3 数据契约

- 预训练目录：`train.bin/val.bin/test.bin`（uint16 小端）、`tokenizer.json`、`meta.json`、`split_report.json`。
- SFT 目录：`train.jsonl/val.jsonl/test.jsonl`、`sft_meta.json`、`split_report.json`。
- `meta.json` 至少包含：dataset、source、source_sha256、vocab_size、specials、splits counts/sha256、unk_rate、val/test fraction、created_at。
- SFT JSONL 字段：`q`, `a`（必填），`group`, `category`, `source`（可选）。

### 7.4 事件与状态 schema

`events.jsonl` 每行：

```json
{"v":1,"ts":"ISO-8601","run_id":"...","event":"run_start|step|eval|checkpoint|sample|minirun|early_stop|run_end|error","step":0,"payload":{}}
```

`status.json`：stage、status（running/completed/failed/stopped）、pid、started_at、updated_at、step、max_steps、best_val、checkpoint、error。

`metrics.csv`：step,train_loss,val_loss,val_ppl,val_bpc,lr,grad_norm,tokens_per_sec,elapsed_s,best_val。

### 7.5 Checkpoint schema

```text
format_version, backend, backend_version, stage, model, optimizer,
iter_num, best_val_loss, config, model_config, tokenizer_state,
data_meta, rng{python,torch,cuda}, early_stopping,
python_version, torch_version, created_at, run_id

纯后端使用 `.mgckpt`（JSON 头 + float32 张量 + sha256），字段集合与上面一致。
```

### 7.6 Web API

- `GET /api/v1/health`
- `GET /api/v1/presets`
- `GET/POST /api/v1/configs`；`GET/PUT/DELETE /api/v1/configs/<name>`；`POST /api/v1/configs/validate`
- `GET/POST /api/v1/jobs`；`GET /api/v1/jobs/<id>`；`POST /api/v1/jobs/<id>/stop`（POST 为“创建并立即启动”的兼容入口）
- `GET/POST /api/v1/runs`（POST = 只新建任务草稿，不启动）；`POST /api/v1/runs/<id>/start`；`DELETE /api/v1/runs/<id>[?force=1]`（非活动任务；`force` 仅用于心跳过期的卡住任务）
- `GET /api/v1/runs/<id>/summary|events?after=|metrics|samples|console`
- `GET /api/v1/system`
- 只允许 `configs/`、`runs/` 白名单；防路径穿越；固定脚本 + 配置校验，禁止任意 shell。
- 默认最多 1 个活动任务；无鉴权；默认 localhost。

### 7.7 `ModelBackend` 协议

- `name`, `version`
- `build_model(model_config, vocab_size)`
- `build_optimizer(model, train_config)`
- `train_step(model, optimizer, batch, grad_clip) -> {loss, grad_norm, tokens}`
- `eval_step(model, batch) -> {loss, tokens}`
- `state_dict(model)`, `load_state_dict(model, state)`
- `optimizer_state_dict(optimizer)`, `load_optimizer_state_dict(optimizer, state)`
- `generate(model, prompt_ids, max_new_tokens, temperature, top_k, top_p, eos_id) -> list[int]`
- `device_info() -> dict`

### 7.8 模型包契约（MiniGPT Bundle v1）

分层：训练 checkpoint（可续训）→ 推理目录（可交换）→ zip（传输）。当前不提供严格 HF adapter。

- 格式标识：`minigpt-bundle-v1`；`config.json` 的 `model_type="minigpt"`、`architectures=["MiniGPT"]`。
- 必含文件：`config.json`、`pytorch_model.bin`、`tokenizer.json`、`special_tokens_map.json`、`generation_config.json`、`run.py`、`serve.py`、`minigpt/` 最小源码子集、`MODEL_CARD.md`、`README.md`、`MANIFEST.json`、`LICENSE`。
- 权重为 CPU state_dict，运行器优先 `torch.load(weights_only=True)`；不引入 `safetensors` 依赖。
- `MANIFEST.json` 记录所有文件的 sha256/字节数；`export.py --verify` 必须校验通过。
- 包内运行时只依赖 Python 标准库 + PyTorch；不需要本仓库、不联网、不需要 `transformers`/`huggingface_hub`。
- **明确不声称** `transformers.AutoModel/AutoTokenizer` 兼容；字符级 tokenizer 不能直接交给 HF 标准 tokenizer 加载。
- 许可：代码 MIT；合成/公版 demo 权重 CC BY 4.0；Alpaca-ZH 权重 CC BY-NC 4.0（仅研究），模型卡须提示。
- 包内不嵌入绝对路径；数据只保留数据集摘要、哈希与统计。

---

## 8. 训练方法

- 模型：6 层/6 头/384 维/block 256（V=68 约 10.75M，10M 参考配置）；debug 4/4/128/128（约 0.81M）。
- Pre-LN、QKV 合并因果注意力、GELU(tanh)、weight tying、Linear/LayerNorm 无 bias、可学习绝对位置。
- AdamW、betas(0.9,0.95)、wd 0.1、warmup + cosine、clip 1.0、AMP（CUDA）、梯度累积、非有限梯度跳过。
- 固定 eval 窗口 + 固定 eval seed。
- mini-run：block/batch 减半、200 步；初始 loss 偏差 ≤0.5、梯度有限且 >0、前 100 步 loss 下降 ≥0.2。
- 早停：预训练 patience=10 次评估；SFT patience=5；min_delta=0；best 覆盖保存。
- `--resume` 恢复 model/optimizer/step/best/RNG/早停；`--init_from` 只加载模型权重。
- 10M：5000 步预训练 + 2000 步 SFT，batch 32/16，lr 6e-4/1e-4，warmup 200；本机不训练。

### 8.1 预设

| preset | stage | 模型 | 数据 | batch | block | steps | lr | 用途 |
|---|---|---|---|---|---|---|---|---|
| `debug_en` | pretrain | 4/4/128 | shakespeare_en | 8 | 128 | 3000 | 1e-3 | 英文 CPU demo |
| `debug_zh` | pretrain | 4/4/128 | zh_gutenberg | 8 | 128 | 1000 | 1e-3 | 中文 CPU demo |
| `en_10m` | pretrain | 6/6/384 | shakespeare_en | 32 | 256 | 5000 | 6e-4 | 10M 参考配置 |
| `zh_10m` | pretrain | 6/6/384 | zh_gutenberg | 32 | 256 | 5000 | 6e-4 | 10M 参考配置 |
| `sft_debug_en` | sft | match init | sft_demo_en | 8 | 128 | 300 | 1e-4 | 英文 SFT |
| `sft_debug_zh` | sft | match init | sft_demo_zh | 8 | 128 | 300 | 1e-4 | 中文 SFT |
| `sft_10m` | sft | match init | 真实 SFT | 16 | 256 | 2000 | 1e-4 | 10M 参考配置 |

---

## 9. 风险登记

| 风险 | 应对 |
|---|---|
| 纯后端被误认为能替代 PyTorch 大模型训练 | `minigpt/pure/` 已接入 tiny 训练与 `.mgckpt` 推理，但 0.81M/10M 在纯 Python 下不实用；文档按此边界描述 |
| CPU 训练慢 | debug 限步；10M 不本机全训；手册估算 |
| 旧 GPU 不可用 | 本机 CPU；现代 GPU 按手册 |
| 公开数据许可 | 仅研究、只缓存不提交、注明 CC BY-NC |
| 看板无鉴权 | 默认 localhost；内网警告；固定脚本/白名单/校验 |
| 任务孤儿 | 服务退出停止活动任务；启停测试 |
| 配置编辑破坏运行 | 启动时冻结配置；编辑只影响下一次 |
| 文档漂移 | 本文件唯一合同；MANUAL/README 与实现同步；测试检查章节 |
| test 污染 | 三分 + group 检查 + test 只评 best 一次 |
| 无 numpy 数据性能 | 数据 1–10MB；array 小端 + 批量转 tensor |

---

## 10. 当前实现清单

当前实现覆盖：环境与包骨架、config/优先级、registry、backend 协议、字符级 tokenizer、数据下载/清洗/二进制读写、SFT 合成与 Alpaca-ZH 导入、三分、模型/TorchBackend/Trainer/checkpoint/callbacks/early stop/telemetry、CLIs、mini-run、双主 debug 预训练、合成 SFT、Web JobManager、配置编辑、原生 ESM + 六阶段页签 SPA（数据/预训练/SFT/评测/生成/导出）、只读数据集/评测/导出扫描接口、Chart.js 前端、/api/v1/inference/stream 流式子进程推理与逐 token 渲染、Alpaca-ZH SFT smoke、评估、chat、流式 token 概率、生成过程 HTML 可视化、截图、MANUAL/README/docstrings、10M 校验、全量 unittest/移植/依赖测试、模型包导出、纯 Python 张量/autograd/纯 GPT/`PureBackend`/`.mgckpt` 与对拍、`--backend_type pure` tiny 训练、`.mgckpt` 的 chat/evaluate 分派。纯后端大模型训练、严格 HF 兼容、HF Hub 上传当前不包含。





