# 知识问答 Agent — 智能客服系统

> 本文档面向**完全没有接触过本项目**的读者：读完即可理解系统做什么、怎么工作（完整数据流链路）、每个文件的作用、如何从零运行、如何运维与二次开发。

---

## 1. 项目是什么？解决什么问题？

一个**企业智能客服系统**。使用场景：

- 公司/部门把自己的业务文档（退货政策、保修规则、支付说明、物流规则等 Markdown 文件）放进 `data/knowledge/`
- 用户（员工或客户）打开网页聊天窗口提问，例如"退款多久到账？"
- 系统自动**检索知识库**找到最相关的内容，交给**大模型**组织成准确的口语化回答，实时流式打字返回
- 对于需要**实时数据**的问题（"我的订单到哪了"），系统会自动**调用工具**查询后作答
- 如果用户问题无法回答，系统**不编造**，明确告知"建议联系人工客服"

一句话概括：**RAG（知识库检索增强生成）+ Agent（工具调用）+ Web 服务** 的完整工程实现。

### 项目实现了哪些功能

| 功能 | 说明 |
|------|------|
| 知识库问答 | Markdown 文档导入 → 向量化 → 混合检索 → LLM 生成带引用的答案 |
| 防幻觉 | 检索分数过滤 + System Prompt 强约束"无依据答不知道" + 答案标注参考来源 |
| 混合检索 | 向量语义检索 + BM25 关键词检索双路召回，加权 RRF 融合，LLM Rerank 精排 |
| 语义缓存 | 相似问题直接秒回（0.1s），知识库更新后自动失效 |
| 意图路由 | 自动区分"知识咨询 / 实时查询 / 闲聊"三类问题，分别走 RAG / 工具 / 引导话术 |
| Agent 工具调用 | ReAct 循环：查订单、查售后工单、查物流（模拟数据，可换真实 API），支持多步推理 |
| 流式 Web 前端 | 聊天界面（打字机效果）、模型切换下拉框、常见问题快捷入口、👍/👎 反馈 |
| 多模型切换 | 页面右上角切换 GLM-5.3-Flash / DeepSeek-V4-Flash / GLM-5.3（OpenAI 兼容推理服务） |
| 评测闭环 | 20 题测试集 + Ragas 四指标自动评测，指标台账自动追加，量化每次迭代 |
| 全链路日志 | 每日日志文件，记录每个环节（缓存/检索/Rerank/生成）的分项耗时，慢在哪一目了然 |
| 可靠性 | LLM 5xx/超时自动重试、生成失败兜底话术、Rerank 失败回退 RRF，服务不空窗 |
| 可观测性 | 可选接入 LangFuse：每次问答上报 trace（意图/检索/工具/答案/耗时），控制台可视化 |
| 容器化 | Dockerfile + docker-compose 一键部署，含可选 vLLM GPU 推理与 LangFuse 自托管 |

---

## 2. 技术栈与运行架构

```
┌────────────────────── 浏览器（用户） ─────────────────────────┐
│  static/  原生 HTML/CSS/JS 聊天页面（无框架，SSE 流式渲染）      │
└──────────────────────────┬───────────────────────────────────┘
                           │ POST /api/chat (SSE 长连接)
┌──────────────────────────▼───────────────────────────────────┐
│  server.py  FastAPI 服务                                      │
│    /api/chat    聊天（SSE）   /api/health  健康检查            │
│    /api/feedback 反馈落盘    /  静态页面                       │
└──────────────────────────┬───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│  kbqa/pipeline.py  全链路编排（核心调度）                       │
│                                                               │
│  ①意图路由  ②语义缓存  ③混合检索  ④Rerank  ⑤LLM生成  ⑥缓存写入 │
│     │          │            │        │                        │
│     ▼          ▼            ▼        ▼                        │
│  agent.py    cache.py    retriever  generator                  │
│  tools.py    embedder    store/bm25                             │
│                                                               │
│  observability.py（LangFuse 上报） logger.py（每日日志文件）    │
└───────┬──────────────────┬────────────────────┬───────────────┘
        ▼                  ▼                    ▼
  data/models/         data/chroma/        LLM 服务
  bge-small-zh-v1.5    Chroma 向量库        (OpenAI 兼容接口)
  (本地 Embedding)     (持久化向量库)
```

- **LLM**：自建/托管 LLM 推理服务（OpenAI 兼容协议，默认地址 `http://localhost:8000/v1`，环境变量 `KBQA_API_BASE` 覆盖），可选 GLM-5.3-Flash（默认）/ DeepSeek-V4-Flash-0731 / GLM-5.3；也可用 docker-compose 里的 vLLM 服务自建
- **Embedding**：本地运行 `BAAI/bge-small-zh-v1.5`（LLM 服务不提供 embeddings 接口，故此模型必须本地跑，CPU 即可）
- **向量库**：Chroma 持久化（本地文件，零部署）

---

## 3. 数据流：一次提问的完整链路

以用户输入"退款多久到账？"为例，从敲下回车到看到答案：

```
[浏览器] 输入问题 → app.js send() → POST /api/chat {question, model}
   │
[FastAPI] server.py chat() → 建立 SSE 长连接 → 调 pipeline.stream_query()
   │
[意图路由] kbqa/agent.py classify_intent()（LLM 分类，约1s）
   ├─ chitchat → 返回引导话术，结束
   ├─ tool_use → 走 Agent 工具链路（见下文"Agent 链路"），结束
   └─ knowledge → 继续 ↓
   │
[语义缓存] kbqa/cache.py lookup()
   ├─ 命中（相似度≥0.92）→ 直接返回缓存答案，0.1s 结束
   └─ 未命中 ↓
   │
[混合检索] kbqa/retriever.py HybridRetriever.retrieve()（<0.1s）
   ├─ ①查询向量化      kbqa/embedder.py（BGE + 官方指令前缀）
   ├─ ②向量检索 Top-20  kbqa/store.py（Chroma 余弦相似度）
   ├─ ③BM25检索 Top-20  kbqa/bm25.py（jieba 分词精确匹配）
   └─ ④RRF 融合去重     0.6/(60+rank) + 0.4/(60+rank)
   │
[Rerank 精排] kbqa/reranker.py：LLM 给候选打 0-10 相关性分
   ├─ 成功 → 取 Top-3、过滤低分（<3）
   └─ 失败/超时(30s) → 自动回退 RRF 顺序（不阻塞）
   │
[Prompt 组装] kbqa/generator.py：Top-3 文档 →【参考资料】+ 防幻觉 System Prompt
   │
[LLM 流式生成] LLM 服务 /chat/completions stream=true
   ├─ 5xx/超时 → 自动重试（1s/3s 退避，最多3次）→ 仍失败 → 兜底话术
   └─ 每收到一个 token → SSE 推给前端 → 打字机渲染
   │
[收尾] 答案写入语义缓存 → 追加"参考来源: xxx.md" → 流结束
   │
[前端] 打字机渲染 + "有帮助吗？👍/👎" 反馈条 → /api/feedback → data/feedback/*.jsonl
```

### Agent 工具链路（实时查询类问题）

意图为 `tool_use` 时，不走上面的检索，而是 **ReAct 循环**（`kbqa/agent.py run()`）：

```
Thought: 分析问题，决定调用哪个工具
Action: query_order
Action Input: {"order_id": "20260901001"}
        ↓ 应用层解析指令 → kbqa/tools.py execute() 安全执行
Observation: {"found": true, "status": "已发货", ...}  ← 回填给 LLM
        ↓ （LLM 判断信息不足可继续调用，最多 3 轮）
Thought: 已获得足够信息
Final Answer: 您的订单已发货，顺丰SF...，预计明天送达
```

每步工具调用以 `tool_call` SSE 事件推给前端，渲染为蓝色步骤条。全程记录在日志与 LangFuse trace 中。

---

## 4. 目录结构与每个文件的作用

```
智能客服agent/
├── server.py                ★ FastAPI 入口：/api/chat(SSE) /api/health /api/feedback + 静态页托管
├── cli.py                   命令行调试入口：ingest(建库) / chat(终端问答) / ask(单次提问)
├── requirements.txt         Python 依赖清单
├── Dockerfile               生产镜像（python:3.11-slim，内置本地模型）
├── docker-compose.yml       一键部署编排（主应用 + 可选 vLLM + 可选 LangFuse）
│
├── kbqa/                    ★ 核心代码包
│   ├── config.py            全局配置中心：所有参数 + 环境变量覆盖（KBQA_*）+ 模型路径解析
│   ├── logger.py            日志系统：log/<日期>.log 按日分文件，控制台 INFO / 文件 DEBUG
│   ├── loader.py            文档解析：读取 data/knowledge/ 下 .md/.txt → Document(text, metadata)
│   ├── chunker.py           文本切分：按 Markdown 标题→段落→句子递归切分，块间重叠 60 字符
│   ├── embedder.py          Embedding：BGE-small-zh 懒加载，查询加官方指令前缀，本地模型优先
│   ├── store.py             向量库封装：Chroma 持久化、cosine 相似度、Top-K 检索
│   ├── bm25.py              BM25 索引：jieba 搜索引擎分词 + rank_bm25，补足精确关键词场景
│   ├── retriever.py         检索器：HybridRetriever（双路召回+RRF 融合）+ Rerank 调度，全链路分项计时
│   ├── reranker.py          Rerank 精排：LLM 给候选文档打 0-10 相关性分（30s 短超时，失败回退 RRF）
│   ├── cache.py             语义缓存：问题向量 cosine≥0.92 直接返回旧答案（跨模型共享），ingest 后清空
│   ├── agent.py             Agent：意图路由（LLM 分类）+ ReAct 循环（解析 LLM 工具指令→执行→回填）
│   ├── tools.py             工具集：3 个模拟工具的 Schema + 实现 + 安全执行器
│   ├── generator.py         LLM 生成：服务客户端（防代理劫持/重试/超时）+ 防幻觉 Prompt + 流式输出
│   ├── pipeline.py          ★ 全链路编排：意图→缓存→检索→Rerank→生成→缓存写入→兜底→观测上报
│   └── observability.py     LangFuse 可观测：每次问答上报 trace（意图/耗时/工具/答案），可选开关
│
├── static/                  Web 前端（无框架）
│   ├── index.html           页面骨架：头部(模型切换下拉框)、消息区、快捷问题、输入框
│   ├── style.css            样式：仿微信客服、流式气泡、工具步骤条、反馈按钮
│   └── app.js               逻辑：SSE 解析渲染、模型切换、快捷提问、反馈发送、状态自检
│
├── data/                    数据目录（chroma/models/feedback 为运行时生成）
│   ├── knowledge/           ★ 知识库文档：放这里的 .md 会被 ingest 导入（当前 5 篇示例）
│   ├── eval/testset.json    评测测试集：20 题（问题-标准答案-预期来源三元组）
│   ├── eval/results/        历次评测报告 JSON
│   ├── feedback/            用户 👍/👎 反馈记录（JSONL，按月分文件）
│   ├── models/              本地模型缓存（bge-small-zh-v1.5）
│   └── chroma/              向量库持久化文件
│
├── metrics/metrics_log.md   指标台账：每次评测自动追加一行（命中率/Ragas四指标/分项耗时）
│
├── log/                     运行日志：YYYY-MM-DD.log，含每个环节分项耗时
├── scripts/
│   ├── download_models.py   模型预下载（外网环境跑一次）
│   └── evaluate.py          评测脚本：跑20题→Ragas四指标→报告JSON→自动写台账
├── Dockerfile               镜像构建（Phase 5）
└── docker-compose.yml       部署编排：kbqa 主服务 + 可选 vllm/langfuse profile
```

---

## 5. 从零运行（完整步骤）

### 0. 环境要求

- Python 3.11
- 可访问一个 OpenAI 兼容 LLM 服务（或按 6.2 节自建 vLLM）

### 1. 安装依赖

```bash
cd 智能客服agent
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 下载 Embedding 模型（仅一次，约 100MB）

> 部分受限网络环境无法直连 HuggingFace，可在有外网的环境执行一次（如手机热点），模型缓存到 `data/models/` 后永久离线可用。

```bash
python scripts/download_models.py --endpoint https://hf-mirror.com
```

### 3. 导入知识库

把业务 Markdown 文档放入 `data/knowledge/`（当前有 5 篇示例），然后：

```bash
python cli.py ingest     # 切分→向量化→写入 Chroma + 重建 BM25 + 清空缓存
```

### 4. 启动服务

```bash
python server.py
# 浏览器打开 http://localhost:8080
```

日常调试可用 `python cli.py chat`（终端）或 `python cli.py ask "怎么退货？" --model GLM-5.3`。

### 5. 配置（可选）

所有配置在 `kbqa/config.py`，均可用环境变量覆盖（Docker 部署主要靠它）：`KBQA_API_BASE`、`KBQA_API_KEY`、`KBQA_LLM_MODEL`、`KBQA_LLM_MODELS`、`KBQA_SERVER_PORT`、`KBQA_LANGFUSE_*`。检索/切分/缓存参数直接改 `config.py` 字段。

### 6. 运行评测（量化效果）

```bash
python scripts/evaluate.py              # 20 题全量：Ragas 四指标 + 耗时 → 自动写 metrics/metrics_log.md
python scripts/evaluate.py --limit 3    # 冒烟
python scripts/evaluate.py --model GLM-5.3   # 换模型对比
```

指标含义与调参方向见下文「8. 评测体系」。

---

## 6. 部署（Phase 5）

### 6.1 Docker 一键部署（推荐）

前置：装好 Docker；如有 GPU 且想自建推理，装 NVIDIA Container Toolkit。

```bash
# 1) 确认 data/models/ 已有 Embedding 模型、data/knowledge/ 已有文档
# 2) 构建并启动（LLM 用自建/托管服务，通过环境变量配置）
docker compose up -d --build

# 3) 首次启动需要导入知识库（容器内执行）
docker compose exec kbqa python cli.py ingest

# 4) 访问 http://localhost:8080
```

数据全部挂载卷：改知识库文档后重新 `exec ingest` 即可，无需重建镜像。

### 6.2 自建 vLLM 推理（可选，需 GPU）

```bash
docker compose --profile vllm up -d
# 然后设置环境变量 KBQA_API_BASE=http://vllm:8000/v1 并重启 kbqa 容器
```

### 6.3 LangFuse 可观测（可选）

```bash
docker compose --profile langfuse up -d
# 1) 浏览器打开 http://localhost:3000 注册，取 Public/Secret Key
# 2) 在 kbqa 服务环境变量中设置：
#    KBQA_LANGFUSE_ENABLED=true
#    KBQA_LANGFUSE_HOST=http://langfuse:3000
#    KBQA_LANGFUSE_PUBLIC_KEY=pk-...
#    KBQA_LANGFUSE_SECRET_KEY=sk-...
# 3) 重启 kbqa。之后每次问答都会上报 trace（意图/检索/工具/耗时/答案），
#    LangFuse 控制台可按 trace 检索、统计、排查 bad case
```

观测模块**失败绝不阻塞问答**：LangFuse 未配置/宕机时只打 warning 日志。

### 6.4 环境变量一览

| 变量 | 说明 | 默认 |
|------|------|------|
| `KBQA_API_BASE` | LLM 服务地址 | `http://localhost:8000/v1` |
| `KBQA_API_KEY` | LLM API Key | 内置 |
| `KBQA_LLM_MODEL` | 默认模型 | GLM-5.3-Flash |
| `KBQA_LLM_MODELS` | 可切换模型列表（逗号分隔） | 三个预置模型 |
| `KBQA_SERVER_PORT` | 服务端口 | 8080 |
| `KBQA_LANGFUSE_ENABLED/HOST/PUBLIC_KEY/SECRET_KEY` | LangFuse 接入 | 关闭 |

---

## 7. 可靠性设计

| 机制 | 位置 | 说明 |
|------|------|------|
| 防幻觉三道闸 | retriever/reranker + generator prompt | 检索阈值过滤 → Prompt 强约束"无依据答不知道" → 答案强制标注来源 |
| LLM 自动重试 | generator `_create_with_retry` | 5xx/超时按 1s/3s 退避重试 3 次，全部失败才兜底 |
| 兜底降级 | pipeline stream_query | 生成失败返回预设话术；检索无命中返回引导话术；不空窗 |
| Rerank 降级 | reranker | 30s 短超时，LLM 繁忙时快速失败回退 RRF 顺序 |
| 语义缓存 | cache.py | 相似问题 0.1s 返回；ingest 后自动失效；跨模型共享 |
| 分项耗时日志 | retriever/generator/pipeline | embed/vector/bm25/rrf/rerank/LLM首字/总耗时逐项记录，"慢在哪"一目了然 |
| 日志按日归档 | log/YYYY-MM-DD.log | 控制台 INFO、文件 DEBUG（含每轮工具调用、rerank 分数明细） |

---

## 8. 评测体系（Phase 2 建立并持续使用）

**当前基线成绩**（20 题测试集，GLM-5.3-Flash，混合检索+LLM Rerank）：

| 指标 | 得分 | 含义 |
|------|------|------|
| 预期文档命中率 | 100% | 检索是否召回正确来源文档 |
| Faithfulness | 0.917 | 忠实度：答案是否完全基于检索上下文（防幻觉核心指标） |
| Context Precision | 1.000 | 相关文档是否排在候选前列 |
| Context Recall | 1.000 | 标准答案的知识点是否被上下文覆盖 |
| Answer Relevancy | 0.864 | 回答是否切题 |
| 平均端到端耗时 | 6.3s | 提问到答案完成 |

**四指标诊断法**（评估 → 诊断 → 优化 → 再评估）：
- Faithfulness 低 → 模型在编造：加强 Prompt 约束
- Context Precision 低 → 检索噪音大：降 Top-K / 强化 Rerank
- Context Recall 低 → 漏召回：调大 chunk_size 或 top_k
- Answer Relevancy 低 → 回答跑题：强化任务目标约束

**延迟诊断**（Phase 3 引入分项耗时日志后定位）：检索本身 <0.1s；大头是 LLM Rerank（3~22s，服务负载波动）与 LLM 生成（约 3s）→ 已用语义缓存（命中 0.1s）+ Rerank 短超时回退缓解。

每次运行 `python scripts/evaluate.py` 会自动把成绩追加到 `metrics/metrics_log.md` 台账，横向对比每次迭代。

---

## 9. 迭代历史（Phase 0 → 5）

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 0 | 手写 RAG 全链路（解析→切分→向量化→检索→生成）+ FastAPI + SSE Web 前端 | ✅ |
| Phase 1 | BM25+向量混合检索（加权 RRF 融合）+ LLM Rerank 精排 + 消融开关 | ✅ |
| Phase 2 | 20 题三元组测试集 + Ragas 四指标评测脚本 + 指标台账（metrics/） | ✅ |
| Phase 3 | 语义缓存 + 兜底降级 + 👍/👎 反馈闭环 + 全链路分项耗时日志 | ✅ |
| Phase 4 | 意图路由（knowledge/tool_use/chitchat）+ 3 个工具 + ReAct 多步推理 + 前端工具步骤条 | ✅ |
| Phase 5 | Docker 容器化 + docker-compose 编排（可选 vLLM/LangFuse）+ LangFuse trace 上报 + 环境变量化配置 | ✅（本阶段） |

各阶段关键设计决策与踩坑记录：

- **Phase 1**：BM25 补足纯向量检索对"退款到账"类精确关键词的短板；Rerank 用 LLM 打分替代本地 CrossEncoder（所用服务无独立 reranker 接口，且免下载 500MB 模型）。
- **Phase 3**：日志显示 rerank 单次调用最高 22s（服务负载波动）→ 引入语义缓存（重复类问题 0.1s）+ Rerank 30s 短超时回退 RRF + LLM 5xx 自动重试。
- **Phase 4**：意图判断 Prompt 中的 JSON 示例曾被 `str.format()` 当占位符解析（KeyError），改用 `replace()`——**教训：Prompt 含花括号示例时禁用 format**。
- **Phase 5**：配置全部支持环境变量覆盖，容器内凭 `KBQA_*` 注入；LangFuse 为可选 profile，不启用零开销。

---

## 10. 已知限制与后续方向

- 工具数据为内存模拟数据（订单/工单/物流），接入真实业务 API 只需替换 `kbqa/tools.py` 中的执行函数
- 语义缓存为进程内存缓存，多副本部署时需换 Redis
- Rerank 依赖 LLM 服务，延迟受其负载影响（后续可换本地 bge-reranker）
- 知识库更新需手动 `ingest`，可加文件监听自动重建
