# RAG Demo 部署方案

> 基于 LangGraph + Ollama(远程) + Qdrant(本地模式) 的最小可用 RAG 系统
>
> 最后更新：方案定稿版

---

## 1. 目标与约束

### 1.1 目标

构建一个可运行的 RAG demo：文档入库 → 向量检索 → LLM 生成回答，并支持多轮对话记忆。

### 1.2 硬件约束

本方案的核心约束来自**两台机器的算力落差**：

| 机器 | 角色 | 配置 | 关键限制 |
|---|---|---|---|
| **A** | 开发/编排端 | 8 GB 内存，Intel Iris Xe 核显（无独显），公司内网笔记本 | 内存不足，跑不动模型 |
| **B** | 模型推理端 | 8 GB 显存（RTX 3070 / 4060 Ti 级），内存 ≥16 GB | 与 A 不在同一网络 |

**核心决策**：模型全部放 B 上跑，A 只承担编排与向量检索。这解决了两件事：

1. A 的内存瓶颈消失（模型权重是大头）
2. 模型规模不再受 A 的 8 GB 限制，可用 8B 级模型，显著改善 RAG 的指令遵循能力

### 1.3 网络约束

A 在公司内网（域 `gildata.com`），B 在家里，**不在同一局域网**。

对策：用 **Tailscale** 组建虚拟局域网（WireGuard 加密 P2P），使方案等价于同一内网。

> 已确认：公司电脑 A 允许安装此类工具，无合规障碍。

---

## 2. 架构

```
┌──────────────────────────┐          ┌───────────────────────────┐
│  机器 A（公司笔记本 8GB）  │          │  机器 B（家用，8GB 显存）   │
│                          │          │                           │
│  LangGraph 编排           │ Tailscale│  Ollama :11434            │
│  Qdrant 本地模式          │ ═══════► │   ├─ qwen3.5:9b  对话     │
│  CLI / FastAPI           │ 加密 P2P │   └─ bge-m3      嵌入     │
│                          │          │                           │
│  内存占用 ≈ 0.5–1 GB      │          │  模型全部驻留 GPU          │
└──────────────────────────┘          └───────────────────────────┘
      100.x.y.z                              100.89.245.44
   (Tailscale IP / MagicDNS)
```

### 2.1 数据流

```
Phase 摄取（离线，建议在 B 本地执行）：
  文档 → 解析 → 切块 → bge-m3 嵌入 → Qdrant 存储目录

Phase 查询（在线）：
  用户问题 → bge-m3 嵌入 → Qdrant 检索 Top-K
           → 拼装 Prompt → qwen3.5:9b 生成 → 回答
           → InMemorySaver 记录会话
```

### 2.2 嵌入阶段：跨网络入库可接受（结论已修正）

> **⚠️ 本节结论曾经历修正。** 初版基于逐条 `embed_query` 的实测外推，判断"必须在 B 本地跑 ingest"；Phase 2 实测后**该判断被推翻**。

**实测数据**：

| 调用方式 | 耗时 |
|---|---|
| `embed_query` 单条（逐条循环） | 3.11 s |
| **`embed_documents` 10 条（批量）** | **0.17 s** |
| **实际入库 243 chunk（完整流程）** | **20.7 s** |

**`langchain-ollama` 会自动批量化**，批量接口比逐条快 18 倍以上。

**结论**：只要用 `embed_documents` 批量接口（本项目 `ingest.py` 已如此实现），**跨网络入库完全可接受**——243 个 chunk 仅需 20.7 秒。无需把 ingest 搬到 B 上执行。

即便是更大的语料（如 2000 chunk），按此速率也只需约 3 分钟，不会成为瓶颈。

> 参考：单次最小嵌入请求的**纯网络往返**开销约 2.7 s（Tailscale 中继），这是批量必须的理由——逐条调用会把该开销乘以文档数。

---

## 3. 技术选型

| 组件 | 选型 | 版本 | 理由 |
|---|---|---|---|
| 编排 | LangGraph | 1.x | 状态机式流程，便于后续加 Self-RAG |
| LLM 运行时 | Ollama | 最新 | 部署简单，HTTP API 无状态 |
| 对话模型 | `qwen3.5:9b` | — | 实测 6.14 GB，8 GB 显存可用（需关思考） |
| 嵌入模型 | `bge-m3` | — | 中文检索强，1024 维，多语言 |
| 向量库 | Qdrant **本地嵌入式模式** | `qdrant-client` 1.19.x | 免 Docker，零端口，后期可无缝迁移服务端 |
| 记忆 | `InMemorySaver` | 随 LangGraph | 零依赖，接口预留可换 Redis |
| 网络 | Tailscale | 最新 | 免端口转发，设备级认证 |

### 3.1 被否决的方案及原因

| 方案 | 否决原因 |
|---|---|
| 模型跑在 A 本机 | 8 GB 内存，3B 已是上限，质量不够 |
| Qdrant 用 Docker | 需 WSL2 + 管理员 + 重启，且额外吃 2 GB 内存 |
| Redis 做记忆 | 需 Docker 或 install exe，demo 阶段收益低 |
| 路由器端口转发暴露 Ollama | **Ollama API 无鉴权**，等于公开 GPU 和会话记录，严重不安全 |
| cloudflared 快速隧道 | 同样暴露公网，需自建认证层，安全性差 |

---

## 4. 模型选型依据

### 4.1 实际可用模型（经 API 实测确认）

以下为远程 Ollama（`http://100.89.245.44:11434`，Ollama 0.34.2）上的真实模型清单：

| 模型 | 体积 | 实测 capabilities | 参数/量化 | 定位 |
|---|---|---|---|---|
| **`qwen3.5:9b`** | **6.14 GB** | completion, **vision**, tools, thinking | 9.7B / Q4_K_M | ✅ **对话主力**（需关思考） |
| `deepseek-r1:8b` | 4.87 GB | completion, tools, thinking | 8.2B / Q4_K_M | 推理专用，本项目不需要 |
| `bge-m3:latest` | 1.08 GB | embedding | 566.7M / F16 | ✅ **嵌入模型，实测 1024 维** |

> 数据来源：`POST /api/show` 实测返回，非文档推测。

#### 关于 `deepseek-r1:8b` 的重要提醒

其 `details.family` 字段返回 **`qwen3`**，而非 DeepSeek 自有架构。说明这是**社区蒸馏版**（以 Qwen3 为基座），不是 DeepSeek 官方原版。

蒸馏版在推理基准上表现接近，但**通用对话能力会打折扣**。本项目做 RAG 问答，**建议用 `qwen3.5:9b`**。

#### 未被采用的候选

| 模型 | 体积（约） | 说明 |
|---|---|---|
| `qwen3:8b` | 5.2 GB | 原规划方案，实际清单中不存在 |
| `qwen2.5:7b` | 4.7 GB | 显存余量更大的保守选择 |
| `qwen3-embedding:0.6b` | 0.6 GB | 显存吃紧时的嵌入备选 |

### 4.2 显存预算核算（实测模型组合 @ 8 GB）

| 项目 | 实测/估算 |
|---|---|
| `qwen3.5:9b` 权重（Q4_K_M） | **6.14 GB** |
| `bge-m3` 权重（F16） | **1.08 GB** |
| **权重合计** | **7.22 GB** |
| KV Cache @ 4096 上下文 | ~0.6 GB |
| 运行时开销 | ~0.5 GB |
| **总计** | **~8.3 GB** |
| 余量 | **-0.3 GB（已超）** |

**结论**：8 GB 显存**装不下两者的完整驻留**。

⚠️ 额外风险：`qwen3.5:9b` 是**多模态模型**（带视觉编码器），实际显存占用可能高于纯文本模型的同参数量估算。

**对策（按推荐度）**：

| 方式 | 做法 | 效果 |
|---|---|---|
| **A. 下调上下文到 4096**（推荐） | `OLLAMA_CONTEXT_LENGTH=4096` | 腾出 KV Cache 空间；RAG 场景下 `TOP_K=4` × 500 字符通常够用 |
| B. 嵌入溢出到 CPU | 接受 `bge-m3` 走 CPU | 嵌入只影响入库阶段，可接受 |
| C. 换更小对话模型 | 如 `qwen2.5:7b` | 需重新拉模型 |

### 4.3 嵌入模型与对话模型共存问题

查询流程是「**1 次嵌入 + 多次 LLM 调用**」。这决定了不该设 `OLLAMA_MAX_LOADED_MODELS=1`：

| 方式 | 做法 | 代价 |
|---|---|---|
| **A. 两模型驻留**（推荐） | 默认行为，接受嵌入溢出到 CPU | 嵌入稍慢，但只影响入库 |
| B. 嵌入跑纯 CPU | 给嵌入请求设 `OLLAMA_NUM_GPU=0` | 需配置分离，略复杂 |
| C. 单模型驻留 | `OLLAMA_MAX_LOADED_MODELS=1` | ❌ 每次查询要经历「卸载 LLM → 加载嵌入 → 卸载 → 重载 LLM」，重载 6.14 GB 需数秒 |

**关键认知**：让嵌入溢出到 CPU 的代价，远小于每次查询重载 6 GB 对话模型。

### 4.4 ⚠️ 思考模式必须显式关闭（实测关键发现）

`qwen3.5:9b` 的 capabilities 含 `thinking`，Ollama 对这类模型**默认开启思考**。这会导致**模型只思考不回答**。

**实测对比**：

| 配置 | 总耗时 | `content` 字段 | 结果 |
|---|---|---|---|
| 默认（think 开启） | **73.8 s** | **空字符串** | ❌ 功能不可用 |
| `think=false` | **3.9 s** | 完整回答 | ✅ 正常 |

**19 倍耗时差距，且默认配置直接返回空内容。**

原因：思考内容占用 `num_predict` 配额。默认 `num_predict=100` 时，100 个 token 全部被思考过程消耗，`content` 为空。

**这不是性能问题，是功能不可用**。不显式关闭的话，代码会表现为"模型什么都不说"。

API 层传参：

```json
{
  "model": "qwen3.5:9b",
  "messages": [...],
  "think": false
}
```

LangChain 层：

```python
llm = ChatOllama(
    model="qwen3.5:9b",
    base_url="http://100.89.245.44:11434",
    temperature=0.1,
    reasoning=False,   # ⚠️ 参数名待验证，见下
)
```

> ⚠️ **待实测确认**：API 层参数名是 `think`，但 `langchain-ollama` 的 Python 参数名可能不同（可能是 `reasoning`，或需通过 `extra_body`/`model_kwargs` 传递）。**装库后必须先验证再写进代码，不可凭猜。**

---

## 5. 机器 B 部署步骤

### 5.1 安装

1. 安装 **Ollama**（Windows/Linux 均可）：https://ollama.com/download
2. 安装 **Tailscale**，用同一账号登录（A、B 都要装）

### 5.2 环境变量

```powershell
# 【关键】绑到所有网卡，否则 Tailscale 也连不上
[Environment]::SetEnvironmentVariable("OLLAMA_HOST","0.0.0.0:11434","User")

# 【关键】RAG 必须显式设置。默认上下文短，会静默截断检索结果
# 【实测调整】因 qwen3.5:9b(6.14GB) + bge-m3(1.08GB) 已占 7.22GB，
#            8GB 显存需下调到 4096 以保证 KV Cache 空间
[Environment]::SetEnvironmentVariable("OLLAMA_CONTEXT_LENGTH","4096","User")
```

Linux 用 systemd override：

```bash
sudo systemctl edit ollama.service
```

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0:11434"
Environment="OLLAMA_CONTEXT_LENGTH=4096"
```

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

> ⚠️ Windows 上设完环境变量**必须完全退出 Ollama 再重启**（托盘图标 → Quit），否则不生效。

### 5.3 验证绑定

```powershell
netstat -ano | findstr 11434
```

| 输出 | 含义 |
|---|---|
| `0.0.0.0:11434  LISTENING` | ✅ 成功 |
| `127.0.0.1:11434` 或 `[::1]:11434` | ❌ `OLLAMA_HOST` 未生效，回去重启 Ollama |

### 5.4 拉取模型

**当前 B 上已有模型（无需重复拉取）**：

| 模型 | 体积 | 用途 |
|---|---|---|
| `qwen3.5:9b` | 6.14 GB | 对话主力 |
| `bge-m3:latest` | 1.08 GB | 嵌入 |
| `deepseek-r1:8b` | 4.87 GB | 推理（本项目未使用，可保留） |

如需补充：

```powershell
ollama pull qwen2.5:7b    # 仅在需要降低显存占用时
ollama show qwen3.5:9b    # 核实体积与上下文长度
```

### 5.5 防火墙（仅同局域网时需考虑）

走 Tailscale 时通常无需额外放行。若确实需要：

```powershell
# 需管理员权限；按来源 IP 白名单，不要全开
New-NetFirewallRule -DisplayName "Ollama LAN" -Direction Inbound `
  -Protocol TCP -LocalPort 11434 `
  -RemoteAddress 100.64.0.0/10 -Action Allow
```

---

## 6. 机器 A 部署步骤

### 6.1 A 不需要安装 Ollama

`langchain-ollama` 是**纯 HTTP 客户端**，与 Ollama 是否在本机无关。不要被同类教程误导去装。

### 6.2 连通性验证（写代码前先做）

```powershell
# 1. 查看 B 的 Tailscale IP
tailscale status

# 2. 用 Tailscale IP 或 MagicDNS 主机名测试
Invoke-RestMethod http://100.89.245.44:11434/api/tags | Select-Object -Expand models | Select-Object name
```

能列出模型 = 链路通。**这一步不通就不要往下写代码。**

> 注：`OLLAMA_ORIGINS` 是给浏览器端跨域用的，Python 服务端客户端**不需要**设置。

### 6.3 建虚拟环境

```powershell
cd D:\project\ai-learn
uv venv --python 3.12 .venv
```

> ⚠️ **本机 PowerShell 执行策略为 `Restricted`**，`.\.venv\Scripts\Activate.ps1` 会直接报错。
>
> 绕法（推荐前者）：
>
> ```powershell
> # 方式 A：不用激活，直接调解释器
> .\.venv\Scripts\python.exe src\cli.py
>
> # 方式 B：改策略（需确认是否符合公司规范）
> Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
> ```

### 6.4 安装依赖

```powershell
uv pip install --python .\.venv\Scripts\python.exe `
  langgraph langchain langchain-core `
  langchain-ollama langchain-qdrant langchain-text-splitters `
  qdrant-client pymupdf
```

| 包 | 最新版 | 作用 |
|---|---|---|
| `langgraph` | 1.2.x | 流程编排 + `InMemorySaver` |
| `langchain` / `langchain-core` | 1.4.x / 1.6.x | 基础抽象 |
| `langchain-ollama` | 1.1.0 | `ChatOllama` / `OllamaEmbeddings` |
| `langchain-qdrant` | 1.1.0 | `QdrantVectorStore` |
| `qdrant-client` | 1.19.x | **本地嵌入式模式的关键** |
| `langchain-text-splitters` | 1.1.x | 中文切块 |
| `pymupdf` | 1.28.x | PDF 解析；渲染为图片用于 OCR 备用 |

**暂不安装**（Phase 4 再装）：`redis`、`langgraph-checkpoint-redis`

**关于 `llama-index`**：本机全局已装 `llama-index 0.14`，`llama/llama.py` 用的就是它。既然主方案走 LangGraph，**建议二选一，不要混用**——两套框架的 Document/Node 抽象不同，混写只会自找麻烦。

> PyPI 直连可用，**无需配置镜像**。

---

## 7. 项目结构

```
ai-learn/
├── .venv/                    # uv 创建的隔离环境
├── data/
│   ├── raw/                  # 原始语料（预留）
│   └── qdrant_storage/       # Qdrant 本地库（已含 243 条向量）
├── pdf/
│   └── 2412.19437v2.pdf      # DeepSeek-V3 技术报告（53 页）
├── src/
│   ├── config.py             # ✅ 模型名 / 地址 / 检索参数 集中管理
│   ├── ingest.py             # ✅ 文档 → 切块 → 嵌入 → 入库
│   ├── retriever.py          # ✅ LLM / Embeddings / VectorStore 单例封装
│   ├── graph.py              # ✅ LangGraph 状态机（retrieve → generate）
│   └── cli.py                # ✅ 问答入口
├── llama/                    # 早期实验（llama-index 路线，与本方案并存）
│   └── llama.py
├── PLAN.md                   # 本文档
└── requirements.txt          # ⏳ 待生成
```

### 7.1 运行方式

```powershell
# 首次：建库（已执行过，需重新入库时才跑）
cd D:\project\ai-learn\src
..\.venv\Scripts\python.exe ingest.py

# 单次提问
..\.venv\Scripts\python.exe cli.py "你的问题"

# 交互式多轮
..\.venv\Scripts\python.exe cli.py
```

> ⚠️ **PowerShell 传中文参数给 stdin 时会乱码**（`UnicodeEncodeError: surrogates not allowed`）。
> 这是 Windows PowerShell 5.1 的 `$OutputEncoding` 默认非 UTF-8 所致，**不是代码问题**。
> 交互使用时直接键盘输入不受影响；若要管道传入，先设置：
>
> ```powershell
> $OutputEncoding = [System.Text.Encoding]::UTF8
> ```
>
> 并给 Python 加 `-X utf8` 参数。

### 7.2 模块职责

| 文件 | 职责 | 关键点 |
|---|---|---|
| `config.py` | 所有可调参数 | 改配置只动这里 |
| `ingest.py` | 加载 → 切块 → 嵌入 → 入库 | 中文友好分隔符；分批写入 |
| `retriever.py` | 组件单例 | `reasoning=False` 在此设置 |
| `graph.py` | 状态机编排 | `InMemorySaver` 挂在 compile |
| `cli.py` | 交互界面 | 退出时显式 `close()` Qdrant |

配置集中到 `config.py` 是有意为之：网络链路和模型选型都可能需要调整，集中管理才能改一处生效全局。

---

## 8. 配置设计

```python
# ---------------- 后端选择 ----------------
# "ollama" = 远程 Ollama（机器 B） | "cloud" = 云端 API
BACKEND = "ollama"

# ---------------- 远程 Ollama ----------------
OLLAMA_BASE_URL = "http://100.89.245.44:11434"   # 实测可用
LLM_MODEL       = "qwen3.5:9b"                  # 实测 6.14 GB
EMBED_MODEL     = "bge-m3:latest"               # 实测 1.08 GB
EMBED_DIM       = 1024        # ✅ 实测确认；建库时锁死，事后不可改

# ---------------- 关键行为开关 ----------------
THINK = False                 # ⚠️ 必须关闭！否则模型只思考不回答（实测见 4.4 节）

# ---------------- 云端备选（仅 BACKEND="cloud" 时生效）----------------
CLOUD_BASE_URL  = "https://api.deepseek.com"
CLOUD_API_KEY   = ""

# ---------------- 检索参数 ----------------
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 80
TOP_K           = 4
QDRANT_PATH     = "./data/qdrant_storage"
COLLECTION_NAME = "demo_docs"
```

**设计要点**：切换后端只改 `BACKEND` 一行。这把「网络不通」的风险降级为一个配置项，而不是重写代码。

---

## 9. LangGraph 流程设计

### 9.1 Phase 3：最小闭环

```
START → retrieve → generate → END
```

`compile(checkpointer=InMemorySaver())` 挂上记忆，即可支持多轮对话。

### 9.2 Phase 4+：检索质量把关（可选增强）

```
START → retrieve → grade ─┬─（合格）→ generate → END
              ↑           │
              └─ rewrite ─┘（不合格）
```

当检索结果与问题相关性不足时，改写问题重新检索（即 Self-RAG 简化版）。

> 建议先把两节点闭环跑通，再加条件边。不要一上来就搭复杂图。

---

## 10. 已知风险与对策

### 10.1 环境类

| # | 风险 | 症状 | 对策 |
|---|---|---|---|
| 1 | 全局 site-packages 不可写 | pip 静默装到用户目录，环境越滚越乱 | **一律用 venv** |
| 2 | PowerShell 策略 `Restricted` | `Activate.ps1` 直接报错 | 直接用 `.venv\Scripts\python.exe` |
| 3 | LangChain 1.x API 大改 | 照抄网上 0.2/0.3 教程必崩 | `AgentExecutor` 已废弃 → `create_agent` |
| 4 | Qdrant 维度不匹配 | 插入即报错 | **维度建库时固定，改不了，只能删库重建** |
| 5 | 中文默认切块效果差 | 语义割裂，检索命中率低 | Splitter 显式传中文分隔符 |
| 6 | Qdrant 本地模式并发限制 | 多进程访问锁冲突 | demo 单进程运行，别同时开两个脚本 |

### 10.2 网络类

| # | 风险 | 症状 | 对策 |
|---|---|---|---|
| 7 | Tailscale 打洞失败 | 走 DERP 中继，延迟升高 | `tailscale status` 查看连接类型；正常现象 |
| 8 | 中继下批量嵌入极慢 | ingest 卡住 | **在 B 本地跑 ingest，拷贝存储目录到 A** |
| 9 | B 进入睡眠 | 连接断开 | 关闭 B 的睡眠；或设 `OLLAMA_KEEP_ALIVE=24h` |
| 10 | Tailscale IP 变动 | 配置失效 | 用 MagicDNS 主机名代替裸 IP |
| 11 | 跨 WAN 请求超时 | 客户端报连接错误 | 增大 HTTP 超时；见 10.3 |

### 10.3 性能类

| # | 风险 | 对策 |
|---|---|---|
| 12 | 跨 WAN 首次连接慢 | 客户端超时设 ≥120s（首次请求要加载 5 GB 模型） |
| 13 | 上下文截断 | 必须设 `OLLAMA_CONTEXT_LENGTH`（实测调整为 4096） |
| 14 | **思考模式导致 content 为空** | ✅ **已实测解决**：显式传 `think=false`（见 4.4 节） |

---

## 11. 安全说明

**Ollama 的 API 完全没有鉴权**，这是设计如此。因此：

1. ✅ 用 Tailscale 做设备级认证，**不暴露任何公网端口**
2. ❌ **绝对不要**做路由器端口转发 / DMZ
3. ❌ 不要用 cloudflared 快速隧道直接暴露（除非自建认证层）
4. ⚠️ 若用 `0.0.0.0` 绑定且处于共享网络，防火墙务必按 IP 白名单放行

---

## 12. 执行顺序

| Phase | 内容 | 依赖 | 状态 |
|---|---|---|---|
| **0** | B 装 Ollama + Tailscale，设环境变量，拉模型；A 装 Tailscale，验证连通 | 无 | ✅ **已完成** |
| **1** | 语料确认：判断是文本件还是扫描件 | 无 | ✅ **已完成**（`pdf/2412.19437v2.pdf`，53 页，有文本层） |
| **2** | 建 venv + 装依赖 + `config.py` + `ingest.py` | Phase 1 | ✅ **已完成**（243 chunk 入库，20.7s） |
| **3** | `graph.py`：retrieve → generate 闭环，CLI 可问答 | Phase 2 | ✅ **已完成** |
| **4** | 换 Redis 替换 `InMemorySaver`（可选） | Phase 3 | ⏳ 待办 |
| **5** | Self-RAG 增强：grade + rewrite 节点（可选） | Phase 3 | ⏳ 待办 |

**Phase 0–3 全部完成，系统已端到端可用。** 实测记录见第 15.5 节。

---

## 13. 语料注意事项

### 13.1 必须先判断文档类型

扫描件没有文本层，直接解析会得到空字符串。**判断方法**：

```powershell
python -c "from pypdf import PdfReader; r=PdfReader('你的文件.pdf'); print(sum(len(p.extract_text() or '') for p in r.pages))"
```

| 输出 | 含义 | 处理 |
|---|---|---|
| 0 | **扫描件**，无文本层 | 必须先 OCR |
| > 0 | 有文本层 | 可直接解析 |

### 13.2 已知情况

项目根目录的 `e6ffad93680a46a1b40c4bc496b4fc03.pdf`（15 页）经实测：

```
全部 15 页 extract_text() 均返回空字符串  →  纯扫描件
```

因此现有 `llama/llama.py` 运行**必然抛出 `RuntimeError`**（该脚本第 25-30 行的显式检查）。

若此文件属于候选语料，需先做 OCR。推荐方案：

| 工具 | 说明 |
|---|---|
| `rapidocr-onnxruntime` | 中文 OCR，纯 CPU，**不需要 torch**（推荐） |
| `pymupdf` + 上述 OCR | 先把 PDF 渲染成图片，再逐页识别 |

> 💡 **多模态新思路**：`qwen3.5:9b` 实测支持 `vision`。若扫描件 OCR 效果不佳，可考虑直接把页面图片交给它做识别。但这会增加显存压力，且需评估准确率，**建议先试传统 OCR**。

---

## 14. 待确认事项

- [x] ~~机器 B 显卡显存~~ → 8 GB
- [x] ~~机器 B 操作系统~~ → Windows（`D:\develop\LLM\blobs\` 路径可证）
- [x] ~~Tailscale 穿透效果~~ → **实测走中继**（单次请求 2.68s，非直连）
- [ ] **`langchain-ollama` 关闭思考的确切参数名**（API 层是 `think`，Python 层待验证）
- [ ] 实际语料的类型（文本件 / 扫描件）与规模
- [ ] `OLLAMA_CONTEXT_LENGTH=4096` 是否足够（取决于 TOP_K 和 chunk 大小）

---

## 15. 实测验证记录

本章记录对远程 Ollama（`http://100.89.245.44:11434`）的实际测试结果，供后续排查对照。

### 15.1 环境信息

| 项目 | 实测值 |
|---|---|
| Ollama 版本 | **0.34.2** |
| 服务端 OS | Windows（模型路径 `D:\develop\LLM\blobs\`） |
| 链路方式 | Tailscale **中继**（非 P2P 直连） |
| 本机位置 | 公司内网 `10.3.110.89` |

### 15.2 模型清单（`GET /api/tags`）

| 模型 | 体积 | capabilities |
|---|---|---|
| `bge-m3:latest` | 1.08 GB | embedding |
| `qwen3.5:9b` | 6.14 GB | completion, vision, tools, thinking |
| `deepseek-r1:8b` | 4.87 GB | completion, tools, thinking |

### 15.3 嵌入模型测试（`POST /api/embed`）

| 项目 | 结果 |
|---|---|
| **维度** | **1024** ✅ |
| L2 范数 | 1.0（已归一化） |
| 批量支持 | ✅ `input` 传数组，返回多条 |
| 单条最小请求 | **2681 ms**（纯网络+调度开销） |
| 10 条批量（冷启动） | 9134 ms（913 ms/条） |
| 10 条批量（预热后） | 3715 ms（**371 ms/条**） |

**关键结论**：瓶颈是网络往返而非算力；批量请求可显著摊销开销（2.5 倍）；模型预热效果明显。

### 15.4 对话模型测试（`POST /api/chat`）

| 配置 | 总耗时 | content | eval 吞吐 |
|---|---|---|---|
| `qwen3.5:9b` 默认（think 开） | **73.8 s** | **空** | 32.4 tok/s |
| `qwen3.5:9b` `think=false` | **3.9 s** | 完整 | **60.2 tok/s** |

**关键结论**：思考模式默认开启会导致**只思考不回答**（content 为空），必须显式关闭；关闭后吞吐翻倍。

### 15.5 Phase 2-3 实现实测记录

#### 15.5.1 环境与依赖

| 包 | 实测版本 |
|---|---|
| `langgraph` | 1.2.11 |
| `langchain` | 1.4.2 |
| `langchain-core` | 1.6.3 |
| `langchain-ollama` | **1.1.0** |
| `langchain-qdrant` | 1.1.0 |
| `qdrant-client` | 1.19.1 |
| `langchain-text-splitters` | 1.1.2 |
| `pymupdf` | 1.28.2 |

#### 15.5.2 ✅ 关键未知项解决：关闭思考的参数名

**`ChatOllama` 的参数名是 `reasoning`（不是 API 层的 `think`）**，实测效果：

| 参数 | 耗时 | content 长度 |
|---|---|---|
| 默认（不传） | 13.0 s | **0** ❌ |
| `reasoning=False` | **0.7 s** | 56 ✅ |

#### 15.5.3 ✅ 嵌入性能（推翻原「必须在 B 本地建库」的结论）

| 调用方式 | 耗时 |
|---|---|
| `embed_query` 单条 | 3.11 s |
| **`embed_documents` 10 条** | **0.17 s** |

**`langchain-ollama` 会自动批量化**，10 条仅 0.17s。实际入库 243 个 chunk 总耗时 **20.7 秒**。

> **⚠️ 结论修正**：`PLAN.md` 第 2.2 节原判断"必须在 B 本地跑 ingest"**已被推翻**。只要用 `embed_documents` 批量接口，跨网络入库完全可接受（243 chunk / 20.7s）。原判断基于逐条 `embed_query` 的外推，是错的。

#### 15.5.4 ⚠️ Qdrant 本地模式锁冲突（已解决）

`QdrantVectorStore.from_documents()` 的 `from_texts` **只接受 `path`/`url`**，会在内部再建一个 client，而本地模式对存储目录加独占锁：

```
RuntimeError: Storage folder ... is already accessed by another instance of Qdrant client
```

**解法**：不用 `from_documents`，改为复用单例 client + 手动建 collection + `add_documents` 分批写入。

#### 15.5.5 ⚠️ TOP_K 是关键调参点（实测证据）

同一问题，不同 TOP_K 的回答质量差异巨大：

| TOP_K | 召回页码 | 回答质量 |
|---|---|---|
| 4 | [1, 9, 4, 22] | ❌ "无法回答" |
| 8 | [1, 9, 4, 22, 35, 10, 13, 7] | ✅ 完整回答 |
| 12 | 同上 + [8, 18, 32] | ✅ 最完整 |

**原因**：embedding 相似度分数密集在 0.60–0.70，**区分度低**。含 MoE 路由细节的 chunk 排在 5-8 位，Top-4 直接漏掉。

**配置定为 `TOP_K = 12`**。实测 prompt 占用：

| 项目 | 实测值 |
|---|---|
| 上下文（TOP_K=8） | 5706 字符 |
| `prompt_eval_count` | **1615 token** |
| `num_ctx` | 4096 |
| **占用率** | **39.4%**（余量 2481 token） |

#### 15.5.6 ⚠️ 已知局限：小模型辨析相近数字会出错

**可复现案例**：问「专家是如何路由的？激活几个路由专家？」

| 位置 | 原文 | 含义 |
|---|---|---|
| p.22 | "**8** experts will be activated for each token" | 每 token 激活的路由专家数 |
| p.14 | "selects only **8** routed experts... scale up to maximum of **13**" | 同上 + 扩展上限 |
| p.19 | "**9** experts" | **解码阶段每 GPU 部署数（不同概念）** |

模型会把 p.19 的 **9** 当成答案，答错。

**根因**：不是检索失败（p.22 在召回结果内），而是 **9B 模型在长上下文中辨析相近数字的能力不足**。这是小模型的固有限制。

**缓解方向**（按可行性）：
1. 换更大模型（如 14B+，但 8 GB 显存不够）
2. prompt 中显式要求标注数字出处页码
3. 后期加 Self-RAG 的 grade 节点做答案自检

#### 15.5.7 多轮记忆验证

`InMemorySaver` + `thread_id` 工作机制正常：

- 「它采用了什么 MoE 架构？」中的"它"被正确解析为本轮话题 ✅
- `clear` 命令通过换 `thread_id` 实现清空 ✅

> ⚠️ **指代型追问的检索会退化**：问题缺少主语时（如"专家是如何路由的？"），向量检索的召回质量下降。实测 LLM 重写（contextualize）能部分改善但不彻底——**这是后续加 Self-RAG 的主要动机**。

#### 15.5.8 环境坑

| 坑 | 现象 | 解法 |
|---|---|---|
| PowerShell 管道编码 | 中文经 stdin 传入报 `UnicodeEncodeError: surrogates not allowed` | 设 `$OutputEncoding = [Text.Encoding]::UTF8`，Python 加 `-X utf8` |
| Qdrant 析构噪音 | 退出时 `ImportError: sys.meta_path is None` | 显式调用 `client.close()`（已在 `cli.py` 处理） |
| docstring 转义警告 | `SyntaxWarning: invalid escape sequence '\.'` | docstring 前加 `r` 前缀 |

### 15.6 残余待验证项

- [ ] `OLLAMA_CONTEXT_LENGTH` 的实际显存占用（需在 B 上看 `nvidia-smi`）
- [ ] 更大的 TOP_K（如 16、20）是否继续改善，以及延迟代价
- [ ] Self-RAG 的 grade 节点能否修复 15.5.6 的数字辨析错误