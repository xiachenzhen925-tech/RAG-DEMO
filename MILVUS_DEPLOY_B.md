# Milvus 部署方案（机器 B · Docker Standalone）

> 目标架构：**模型与向量库全部迁到机器 B**，机器 A 只保留 LangGraph 编排。
>
> 本文档中的命令分「A 上执行」与「B 上执行」，请注意区分。
> 标注 ✅ 的为**本机实测确认**；标注 ⚠️ 的为**待你在 B 上验证**。

---

## 1. 为什么装在 B 上（而不是 A）

### 1.1 机器 A 的三个硬障碍（已实测确认）

| 障碍 | 实测证据 | 影响 |
|---|---|---|
| **无管理员权限** | `管理员: False`（用户 `GILDATA\xiacz58567`） | ❌ `wsl --install`、启用来宾功能、装 Docker Desktop **全部需要提权** |
| **WSL 完全未启用** | `wsl --status` → **exit code 50**；`wsl --list --verbose` → 打印帮助文本 exit 1；无 Ubuntu/Debian 发行版 | ❌ 未启用功能 |
| **系统为 LTSC 版本** | `Windows 10 Enterprise LTSC 2021`（21H2, Build 19044.7725） | ⚠️ Docker Desktop **不支持 LTSC**；且 LTSC 无 Store，`wsl --install` 依赖的 Store 分发路径不可用 |

> **决定性结论**：A 上装 Docker/WSL2 不可行。这不是"麻烦"，是"做不到"。

### 1.2 机器 B 的优势

| 项 | A（公司笔记本） | B（家用机） |
|---|---|---|
| 内存 | 8 GB | **16 GB+** |
| 显卡 | Intel Iris Xe 核显 | **NVIDIA 8 GB 显存** |
| 管理员权限 | ❌ 无 | ✅ **有** |
| Ollama | 不跑 | ✅ **已在跑**（`100.89.245.44:11434`） |
| Tailscale | 已通 | 已通 |

**架构合理性**：模型（Ollama）和向量库（Milvus）都在 B，两者通信走本机回环（`localhost`），**延迟从 ~2.7 s 降到 ~0**。A 退化为纯编排端。

### 1.3 附带收益：解决 milvus-lite 的 BM25 问题

上一轮在 A 上用 `milvus-lite 3.2.1` 实测的 BM25 混合检索**接不通**：
`MilvusException: weighted merge requires 2 weights, got 4`（详见 `MILVUS_MIGRATION.md`）。

改用 **Milvus Standalone（原生 C++ 实现）** 后，混合检索是官方成熟功能，不再有纯 Python 重写的兼容问题。

---

## 2. 目标架构

```
┌──────────────────────────────┐          ┌─────────────────────────────────┐
│  机器 A（公司笔记本 8 GB）     │          │  机器 B（家用，8GB 显存 / 16GB+）│
│                              │          │                                 │
│  LangGraph 编排               │ Tailscale│  ┌─ Ollama :11434 ────────────┐  │
│  ingest.py / cli.py           │ ═══════► │  │  qwen3.5:9b / bge-m3       │  │
│                              │          │  └────────────────────────────┘  │
│  不再需要：                   │          │  ┌─ Milvus Standalone :19530 ─┐  │
│   ├─ Qdrant（本地嵌入式）      │          │  │  ├─ milvus-standalone      │  │
│   └─ data/qdrant_storage/     │          │  │  ├─ milvus-etcd            │  │
│                              │          │  │  └─ milvus-minio           │  │
│  内存占用 ≈ 0.5–1 GB          │          │  └────────────────────────────┘  │
└──────────────────────────────┘          │  两者走 localhost，延迟≈0         │
                                          └─────────────────────────────────┘
```

**关键变化**：A 上的 `data/qdrant_storage/` 目录将**弃用**（不是删除，保留以便回退）。

---

## 3. ⚠️ 装之前必须在 B 上验证的三件事

这三项**任一不满足都会导致安装失败**。请在 B 上先跑完再动手。

### 3.1 检查项 1：Windows 版本是否为 LTSC

```powershell
# 在 B 上执行
Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion' |
  Select-Object ProductName,DisplayVersion,CurrentBuild
```

| 结果 | 结论 |
|---|---|
| `Windows 10/11 专业版/家庭版 21H2+` | ✅ Docker Desktop 可装 |
| **含 `LTSC` / `LTSB`** | ❌ **Docker Desktop 官方不支持**，需改用替代方案（见第 8 节） |

> ⚠️ A 是 LTSC，如果 B 也是（企业批量部署的常见情况），整个 Docker 路线要改。

### 3.2 检查项 2：CPU 是否支持 AVX2 / SSE4.2

**这是 Milvus 最经典的安装失败原因**——Milvus 的向量计算依赖 SIMD 指令集，CPU 不支持会直接崩溃。

```powershell
# 在 B 上执行（PowerShell 无内置指令集查询，用 Python）
python -c "
import platform
print('CPU:', platform.processor())
# 或用 coreinfo（Sysinternals）更准确：
"
```

更可靠的做法（二选一）：

```powershell
# 方式 1：下载 Sysinternals coreinfo（无需安装）
# https://learn.microsoft.com/sysinternals/downloads/coreinfo
coreinfo.exe -f | Select-String "AVX2|SSE4"

# 方式 2：WSL 内执行（装好 WSL 后）
cat /proc/cpuinfo | grep -o "avx2\|sse4_2" | sort -u
```

| 结果 | 结论 |
|---|---|
| 含 `AVX2` | ✅ 满足 |
| 只有 `SSE4.2` | ⚠️ 可能可用但受限，需实测 |
| 都没有 | ❌ Milvus 无法运行 |

> 参考：A 的 CPU 是 `Intel Core i5-10210U`（10 代，支持 AVX2）。**但 B 的 CPU 需单独确认。**

### 3.3 检查项 3：内存与磁盘

```powershell
# 在 B 上执行
# 内存
(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB
# 磁盘
Get-PSDrive D | Select-Object @{n='FreeGB';e={[math]::Round($_.Free/1GB,1)}}
```

| 项 | 要求 | 说明 |
|---|---|---|
| 内存 | **≥ 8 GB 可用** | Milvus + etcd + minio 三容器；**且 B 上还跑着 Ollama** |
| 磁盘 | **≥ 10 GB** | 镜像约 2-3 GB + 数据卷 |

> ⚠️ **资源竞争提醒**：B 上已有 Ollama（`qwen3.5:9b` 占 6.14 GB 显存）。Milvus **主要吃内存而非显存**，但要确认 16 GB 内存够分。
> 若紧张，可给 WSL2 设上限（见 4.4 节）。

---

## 4. B 上安装 Milvus Standalone

### 4.1 安装 Docker Desktop

1. 下载：https://www.docker.com/products/docker-desktop/
2. 安装时**勾选 "Use WSL 2 instead of Hyper-V"**
3. 安装完成后重启
4. 验证：

```powershell
docker --version
docker compose version
docker run hello-world
```

> 若 WSL2 未自动装好，Docker Desktop 首次启动会提示，按其指引执行 `wsl --install`（此时已有管理员权限）。

### 4.2 下载官方 compose 文件

```powershell
mkdir D:\milvus
cd D:\milvus

# 从官方仓库拉取（已实测该 URL 可访问）
curl.exe -L -o docker-compose.yml https://raw.githubusercontent.com/milvus-io/milvus/master/deployments/docker/standalone/docker-compose.yml
```

**实测获取到的官方 compose 内容**（供核对）：

| 容器 | 镜像 | 端口 |
|---|---|---|
| `milvus-etcd` | `quay.io/coreos/etcd:v3.5.25` | 内部 2379 |
| `milvus-minio` | `minio/minio:RELEASE.2024-05-28T17-19-04Z` | 9000 / 9001 |
| `milvus-standalone` | **`milvusdb/milvus:v3.0.1`** | **19530**（gRPC/SDK）、9091（健康检查） |

> **重要**：镜像版本是 **v3.0.1**，与 `pymilvus 3.0.2` 同属 3.x 线，**版本匹配**。
> 这正是修复 milvus-lite BM25 问题的关键——之前的错配出在纯 Python 重写版。

初始化 compose 文件：

```powershell
# 官方要求：先创建 compose 文件（部分版本需手动）
curl.exe -L -o docker-compose.yml https://github.com/milvus-io/milvus/releases/download/v3.0.1/milvus-standalone-docker-compose.yml
```

### 4.3 启动

```powershell
cd D:\milvus
docker compose up -d
```

验证容器状态：

```powershell
docker compose ps
```

期望看到 3 个容器均为 `running` / `healthy`。

检查 Milvus 健康端点：

```powershell
# B 本机
curl.exe http://localhost:9091/healthz
# 期望返回 OK
```

### 4.4 暴露给 Tailscale（关键步骤）

**默认 Milvus 只绑 `19530` 到宿主所有网卡**，但需确认防火墙放行。先查 Tailscale IP：

```powershell
tailscale ip -4
# 例如 100.89.245.44（与 Ollama 同一 IP）
```

放行防火墙（B 上有管理员权限）：

```powershell
New-NetFirewallRule -DisplayName "Milvus via Tailscale" -Direction Inbound `
  -Protocol TCP -LocalPort 19530 `
  -RemoteAddress 100.64.0.0/10 -Action Allow
```

> 同理，若要用 Attu（Milvus 的 Web UI，可选）需再放行 8000 端口。

⚠️ **安全提醒**：Milvus 默认**无认证**。Tailscale 提供设备级认证，但**不要**把 19530 暴露到公网。

### 4.5 （可选）限制 WSL2 资源占用

B 上已有 Ollama 在跑，若内存吃紧，在 `C:\Users\<用户名>\.wslconfig` 中限制：

```ini
[wsl2]
memory=6GB
processors=4
swap=2GB
```

改完执行 `wsl --shutdown` 重启 WSL。

---

## 5. A 上的代码改造

### 5.1 前置检查：pymilvus 版本对齐

⚠️ **当前 A 的 venv 里装的是 `pymilvus 3.0.2`**（上一轮测试 milvus-lite 时装入）。

由于 B 上的 Milvus 是 **v3.0.1**，两者同属 3.x 线，**理论上匹配**。但需实测确认。

同时**建议卸载 milvus-lite**（不再需要）：

```powershell
# A 上执行
cd D:\project\ai-learn
uv pip uninstall --python .\.venv\Scripts\python.exe `
  milvus-lite faiss-cpu pyarrow pandas
```

> 保留 `pymilvus` 和 `langchain-milvus`（要用来连 B）。

### 5.2 连通性验证（写代码前必做）

```powershell
# A 上执行
```

```python
# src/_probe_b.py（临时验证脚本）
from pymilvus import MilvusClient

client = MilvusClient(uri="http://100.89.245.44:19530")
print("Milvus 版本:", client.get_server_version())
print("已有 collections:", client.list_collections())
```

**这一步不通，后面全部无从谈起。**

### 5.3 config.py 改造

```python
# ---------------- 向量库后端 ----------------
# "qdrant" = A 本地嵌入式（旧） | "milvus" = B 上的 Standalone（新）
VECTOR_BACKEND = "milvus"

# Qdrant（保留以便回退）
QDRANT_PATH = str(DATA_DIR / "qdrant_storage")

# Milvus（机器 B，走 Tailscale）
MILVUS_URI = "http://100.89.245.44:19530"
MILVUS_TOKEN = ""          # 默认无认证

# ---------------- 混合检索开关 ----------------
# Milvus Standalone 支持 BM25 稠密+稀疏混合检索
USE_HYBRID_SEARCH = True
HYBRID_RANKER = "rrf"      # "rrf" | "weighted"
HYBRID_WEIGHTS = [0.7, 0.3]  # 仅 weighted 时生效（dense, sparse）
```

### 5.4 retriever.py 改造

```python
@lru_cache(maxsize=1)
def get_vectorstore():
    if VECTOR_BACKEND == "qdrant":
        return QdrantVectorStore(
            client=_get_qdrant_client(),
            collection_name=COLLECTION_NAME,
            embedding=get_embeddings(),
        )

    if VECTOR_BACKEND == "milvus":
        from langchain_milvus import BM25BuiltInFunction, Milvus

        kwargs = {}
        if USE_HYBRID_SEARCH:
            kwargs["builtin_function"] = BM25BuiltInFunction()

        return Milvus(
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
            **kwargs,
        )

    raise ValueError(f"未知后端: {VECTOR_BACKEND}")
```

搜索时启用 ranker：

```python
def get_retriever(k: int = TOP_K):
    vs = get_vectorstore()
    search_kwargs = {"k": k}
    if VECTOR_BACKEND == "milvus" and USE_HYBRID_SEARCH:
        search_kwargs["ranker_type"] = HYBRID_RANKER
    return vs.as_retriever(search_kwargs=search_kwargs)
```

> ⚠️ **注意**：`ranker_type` 在 `langchain-milvus 0.4.0` 中已标记 deprecated（推荐用 `reranker` Function 对象）。
> 但上一轮实测发现新版 `reranker` 参数在 milvus-lite 上不被识别。
> **在 Standalone 上要用哪种写法，必须在 B 上实测确定**——这是第 9 节的头号待验证项。

### 5.5 ingest.py 改造

现有 `build_vectorstore()` 是 Qdrant 专用的（手动建 collection + 分批 add）。
Milvus 侧可以简化——它对 `from_documents` 的支持更好，因为它连的是**独立服务**，没有本地锁问题：

```python
def build_vectorstore_milvus(chunks, *, recreate=True):
    from langchain_milvus import BM25BuiltInFunction, Milvus

    return Milvus.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        builtin_function=BM25BuiltInFunction(),
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=recreate,     # 等价于 Qdrant 侧的 delete_collection
    )
```

> **对比**：Qdrant 本地模式必须手动复用单例 client（否则锁冲突），而 Milvus 走网络服务**没有这个问题**——这是迁移的一个附带简化。

---

## 6. 迁移执行顺序

| 阶段 | 位置 | 内容 | 验收标准 |
|---|---|---|---|
| **P0** | B | 跑第 3 节三项检查 | 三项全部通过 |
| **P1** | B | 装 Docker Desktop + 启动 Milvus | `curl localhost:9091/healthz` 返回 OK |
| **P2** | B | 放行防火墙 | — |
| **P3** | A | `curl http://100.89.245.44:19530` 连通 | ✅ |
| **P4** | A | 卸载 milvus-lite，改 `config.py` | — |
| **P5** | A | 改 `retriever.py` + `ingest.py` | — |
| **P6** | A | **重建索引** | 243 chunk 全部入库 |
| **P7** | A | **验证混合检索** | BM25 生效，结果与纯向量**不同** |
| **P8** | A | 端到端问答 | `cli.py` 正常，多轮正常 |

**P0 是硬阻塞点**：任一项不通过，整个方案要改（见第 8 节）。

---

## 7. 风险与对策

| # | 风险 | 症状 | 对策 |
|---|---|---|---|
| 1 | **B 也是 LTSC** | Docker Desktop 装不上或运行异常 | 改用第 8 节替代方案 |
| 2 | **CPU 不支持 AVX2** | Milvus 容器启动后崩溃 | 无解，Milvus 硬性要求 |
| 3 | 内存不足 | 容器被 OOM Kill | 设 `.wslconfig` 限制；或停掉 B 上其他程序 |
| 4 | Tailscale 走中继 | 检索延迟高 | 检索请求体积小，影响有限；**但建库必须批量化** |
| 5 | `ranker_type` 写法不兼容 | 混合检索报错 | 见第 9 节，需实测两种写法 |
| 6 | 端口未放行 | A 连不上 19530 | 检查防火墙 + `netstat` |
| 7 | 建库慢 | 243 chunk 嵌入跨网络 | 用 `embed_documents` 批量（已验证自动批量） |
| 8 | 镜像拉取慢/失败 | `docker compose up` 卡住 | 配国内镜像加速器 |

### 7.1 镜像加速（国内网络）

若拉取镜像慢，在 Docker Desktop → Settings → Docker Engine 中加入：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://dockerproxy.com"
  ]
}
```

---

## 8. 如果 B 也是 LTSC：替代方案

| 方案 | 做法 | 评价 |
|---|---|---|
| **A. Milvus 二进制直接跑** | Milvus 提供 Linux 二进制，**不提供 Windows 原生版** → 仍需 WSL2 | ⚠️ 回到 WSL2 问题 |
| **B. Podman Desktop** | 比 Docker Desktop 对 LTSC 更宽容 | ⚠️ 需实测；Windows 上仍依赖 WSL2 |
| **C. B 装 Linux 双系统** | 最干净，Milvus 原生 | ✅ 一劳永逸，但要重装系统 |
| **D. 用远程/云 Milvus** | Zilliz Cloud 免费额度 | ✅ 零运维，但数据出内网 |
| **E. 保持 Qdrant** | 放弃 Milvus | ✅ 当前方案已可用 |

> **注意**：WSL2 本身在 LTSC 2021 上是**可以启用**的（LTSC 2021 = 21H2 内核），
> 问题主要在 **Docker Desktop 官方不支持 LTSC**。所以方案 B（Podman）值得一试。

---

## 9. 待验证事项（按优先级）

- [ ] **P0-1**：B 的 Windows 版本是否 LTSC
- [ ] **P0-2**：B 的 CPU 是否支持 AVX2
- [ ] **P0-3**：B 的可用内存与磁盘
- [ ] **P1**：Milvus v3.0.1 容器能否正常启动（`healthz` 返回 OK）
- [ ] **P3**：A 能否连上 `100.89.245.44:19530`
- [ ] **P7-1**：`ranker_type="rrf"` 在 Standalone 上是否可用
- [ ] **P7-2**：`reranker` Function 对象写法是否可用（0.4.0 推荐方式）
- [ ] **P7-3**：**BM25 是否真的参与打分**（对比混合 vs 纯向量的召回结果是否不同）
- [ ] **P7-4**：`pymilvus 3.0.2` ↔ Milvus v3.0.1 兼容性
- [ ] 混合检索的延迟代价（多一次 sparse 检索）
- [ ] 中文语料下 BM25 分词效果（当前语料为英文论文，`analyzer_params` 需配中文分词器）

> **P7-3 是核心验收点**。上一轮 milvus-lite 的教训是：`rrf` 调用"成功"但结果与纯向量完全相同，
> 说明 BM25 根本没参与。**必须对比两者结果是否不同**，不能只看调用是否报错。

---

## 10. 回退方案

架构改造保持**可回退**：

1. `config.py` 里 `VECTOR_BACKEND = "milvus"` 改回 `"qdrant"` 即可
2. `data/qdrant_storage/` 目录**不要删除**（243 向量仍在）
3. `retriever.py` 的两个分支并存，不删旧代码

**回退成本**：改 1 行配置 + 重启进程。

---

## 11. A 上的当前环境状态（供参考）

| 项 | 状态 |
|---|---|
| Qdrant collection | `deepseek_v3`，243 向量，1024 维，Cosine ✅ |
| 向量库文件 | `data/qdrant_storage/`（约 2.8 MB） |
| venv | `.venv`（Python 3.12.13） |
| 已装但待卸载 | `milvus-lite`、`faiss-cpu`、`pyarrow`、`pandas` |
| 待保留 | `pymilvus 3.0.2`、`langchain-milvus 0.4.0` |
| Ollama | 远程 `100.89.245.44:11434`，模型 `qwen3.5:9b` / `bge-m3` |
| 关键配置 | `TOP_K=12`、`reasoning=False`、`num_ctx=4096` |

**A 侧不改动的文件**：`graph.py`、`cli.py`（都只依赖 `retriever.py` 暴露的接口）。