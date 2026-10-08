# Qdrant → Milvus 迁移方案

> 本方案基于**实际安装与运行验证**，非文档推测。
> 验证环境：Windows / Python 3.12.13 / 远程 Ollama（`100.89.245.44:11434`）

---

## 1. 结论摘要（TL;DR）

| 项目 | 结论 |
|---|---|
| **Milvus Lite 能否在 Windows 跑** | ✅ **能**（3.2.1 起为纯 Python 实现，无需 Docker/WSL2） |
| **纯向量检索能否工作** | ✅ 能（已验证建库 + 检索） |
| **BM25 混合检索能否工作** | ❌ **不能**（版本不兼容，实测报错） |
| **迁移难度** | 低（`VectorStore` 统一接口，上层代码不用改） |
| **依赖成本** | 高（`pyarrow` 单包 82 MB） |
| **建议** | **暂不迁移**，除非有明确的生产化需求 |

**核心理由**：当前 Qdrant 方案已完整可用，而 Milvus Lite 的**最大增量价值（BM25 混合检索）实测不可用**。迁移的收益目前小于成本。

---

## 2. 验证过程与实测数据

### 2.1 环境探测

`milvus-lite 3.2.1` 的 wheel 是 **`py3-none-any`**（纯 Python，平台无关），这与"Milvus Lite 只支持 Linux/macOS"的旧印象完全不同。

包元数据确认为**官方项目**：

| 字段 | 值 |
|---|---|
| 项目地址 | `github.com/milvus-io/milvus-lite` |
| 作者 | Milvus Contributors |
| 许可 | Apache-2.0 |
| Requires-Python | `>=3.10` |

官方说明原文：

> This generation of Milvus Lite is **rebuilt from scratch in pure Python**. Instead of wrapping the full C++ Milvus core, it uses a local LSM-style storage engine with **WAL, in-memory memtables, immutable Parquet segments, segment-level FAISS indexes**, scalar filtering, **BM25 full text search**, and a Milvus-compatible **gRPC adapter**.

即：**性能特征与传统 C++ 版 Milvus 不同**，是纯 Python 重新实现的兼容层。

### 2.2 基础功能验证 ✅

| 测试项 | 结果 |
|---|---|
| `MilvusClient(uri='./xxx.db')` 启动 | ✅ `milvus_lite-3.2.1` |
| 建表 + 插入 50 条 1024 维向量 | ✅ |
| 检索 top-3 | ✅ 返回正确排序（自身 1.0，其余 0.76） |
| 删表 | ✅ |
| 并发打开两个 client | ⚠️ 能打开（**但并发写安全性未验证**） |

### 2.3 包内部结构（`milvus_lite/`）

纯 `.py` 实现，关键模块：

| 模块 | 作用 |
|---|---|
| `engine/` | 存储引擎（collection / compaction / flush / recovery） |
| `storage/` | **WAL + memtable + segment + manifest** |
| `index/` | `faiss_hnsw.py` / `brute_force.py` / `faiss_ivf_flat.py` / `sparse_inverted.py` |
| `analyzer/jieba_analyzer.py` | **内置中文分词** |
| `search/filter/` | 标量过滤（AST / parser / arrow_backend） |
| `function/expr/bm25_expr.py` | BM25 表达式 |
| `rerank/` | boost / decay 重排 |
| `adapter/grpc/` | Milvus 兼容 gRPC 适配层 |

### 2.4 依赖体积

| 包 | 体积 |
|---|---|
| **`pyarrow`** | **82.1 MB** |
| `pymilvus` | 4.2 MB |
| `milvus_lite` | 1.7 MB |
| `faiss-cpu` | （有安装，体积未单独统计） |
| `pandas` | 随依赖安装 |

> `pyarrow` 是 Parquet 存储的代价，**一个包就 82 MB**。

### 2.5 版本矩阵与兼容性 ❌

实际安装到的版本：

| 包 | 版本 |
|---|---|
| `milvus-lite` | **3.2.1** |
| `pymilvus` | **3.0.2** |
| `langchain-milvus` | **0.4.0** |

**建库成功**，schema 正确注册了 BM25：

```
fields:
  text         (VARCHAR)
  pk           (INT64)
  vector       (FLOAT_VECTOR)
  sparse       (SPARSE_FLOAT_VECTOR)     ← BM25 输出字段
  source / page / total_pages

functions:
  name: bm25_function_eba89791
  type: FunctionType.BM25
  input_field_names:  [text]
  output_field_names: [sparse]
```

**但混合检索失败**：

| 调用方式 | 结果 |
|---|---|
| `similarity_search(q, k=4)` | ✅ 纯向量正常 |
| `similarity_search(q, k=4, ranker_type="rrf")` | ⚠️ 首次成功，**后续报错** |
| 不传 ranker（存在 BM25 schema 时） | ❌ 报错 |
| `ranker_type=RRFRanker()` 对象 | ❌ `Unrecognized ranker of type` |

**报错信息**：

```
MilvusException: (code=1, message=weighted merge requires 2 weights, got 4)
```

**根因定位**（追踪到 `milvus_lite/function/ops/merge_op.py:72`）：

```python
if self._strategy == "weighted" and len(self._weights) != len(inputs):
    raise ValueError(
        f"weighted merge requires {len(self._inputs)} weights, "
        f"got {len(self._weights)}"
    )
```

**问题链条**：

1. `langchain-milvus 0.4.0` 已把 `ranker_type` 标记为 **deprecated**，推荐用 `reranker`（Function 对象）
2. 新的 `reranker` 参数是 `pymilvus 3.x` 的新 API
3. `milvus-lite 3.2.1` 服务端**未能解析该 ranker**，退化为 `weighted` 策略
4. `weights` 列表为空（`len=0`），而 `inputs` 有 2 路（dense + sparse）
5. 校验失败

> **关键证据**：报错中的数字 `got 4`、`got 6` 恰好等于我传入的 `k` 值（4 和 6）。
> 说明 **`k` 被错误地当作了 weights**，参数传递链路错位。

### 2.6 BM25 是否真的生效

对比"纯向量"与"RRF"的检索结果：

| 查询 | 纯向量页码 | RRF 页码 | 是否不同 |
|---|---|---|---|
| `Borda count 是什么？` | [17, 22, 21, 16] | [17, 22, 21, 16] | ❌ **完全相同** |

**页码完全一致**，说明即使 RRF"调用成功"，**BM25 也未实际参与打分**——结果与纯向量无异。

---

## 3. 迁移方案设计

### 3.1 设计原则

利用 `langchain` 的 `VectorStore` 统一接口，**让上层零改动**：

```
graph.py  ─┐
cli.py    ─┼─►  get_retriever()  ─►  BACKEND 分支  ─┬─► QdrantVectorStore
ingest.py ─┘                                        └─► Milvus
```

`graph.py` 和 `cli.py` **一行都不用改**，因为两者都只依赖 `retriever.py` 暴露的函数。

### 3.2 配置扩展（`config.py`）

```python
# ---------------- 向量库后端 ----------------
# "qdrant" = Qdrant 本地嵌入式（当前）
# "milvus" = Milvus Lite（纯 Python）
VECTOR_BACKEND = "qdrant"

# Qdrant
QDRANT_PATH = str(DATA_DIR / "qdrant_storage")

# Milvus
MILVUS_URI = str(DATA_DIR / "milvus_demo.db")
```

### 3.3 检索层改造（`retriever.py`）

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
        from langchain_milvus import Milvus
        return Milvus(
            embedding_function=get_embeddings(),
            collection_name=COLLECTION_NAME,
            connection_args={"uri": MILVUS_URI},
        )
    raise ValueError(f"未知后端: {VECTOR_BACKEND}")
```

> ⚠️ **不要用 `Milvus.from_documents()`**：它会重建 client 导致资源重复。
> 用构造函数 + `add_documents`，与 Qdrant 当前的实现模式保持一致。

### 3.4 关键差异对照

| 维度 | Qdrant 本地 | Milvus Lite |
|---|---|---|
| 构造 | `QdrantClient(path=...)` | `Milvus(connection_args={"uri": "...db"})` |
| 建 collection | 手动 `create_collection` + `VectorParams` | 自动（或 `drop_old=True`） |
| 距离 | 建库时指定 `Distance.COSINE` | `metric_type="COSINE"` |
| **父目录要求** | 自动创建 | ⚠️ **必须预先存在**（实测踩坑） |
| 维度 | 建时固定，不可改 | 同 |
| 并发锁 | ❌ 独占锁 | ⚠️ 可多开但不保证安全 |

> ⚠️ **实测踩坑**：`pymilvus 3.x` 要求本地库的**父目录必须先存在**，否则报
> `ConnectionConfigException: Open local milvus failed, dir: data not exists`。
> 迁移时需在初始化前 `Path(...).parent.mkdir(parents=True, exist_ok=True)`。

---

## 4. 迁移路径

### 4.1 分阶段推进

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| **P0** | 在 `config.py` 加 `VECTOR_BACKEND` 开关 | 两套代码并存，Qdrant 默认不受影响 |
| **P1** | `retriever.py` 加 Milvus 分支 + `ingest.py` 支持双写 | 同一份 PDF 能分别入两库 |
| **P2** | **平行对比检索质量** | 用同一批问题对比命中率/页码 |
| **P3** | 确认无回退后切换默认后端 | 端到端跑通 |
| **P4** | （可选）解决 BM25 兼容问题 | 见下节 |

### 4.2 必须先解决的前置问题

**升级 `langchain-milvus` 版本**（当前 0.4.0 已是最新，说明问题在 `pymilvus`/`milvus-lite` 侧）

**可行方向**：

| 方向 | 做法 | 风险 |
|---|---|---|
| A. 降 `pymilvus` 到 2.6.x | 与 `langchain-milvus` 的 `ranker_type` 老 API 对齐 | 可能不兼容 `milvus-lite 3.2.1` |
| B. 升 `milvus-lite` | 当前已是最新 3.2.1 | 无更新版本可用 |
| C. 绕过 `langchain-milvus`，直接用 `pymilvus` 原生 `hybrid_search` | 能精确控制 ranker 参数 | 需自己实现 `VectorStore` 适配 |
| D. 换用 Milvus Standalone（Docker） | C++ 版成熟，混合检索稳定 | 需 Docker + WSL2，本机不具备 |
| E. **暂不做混合检索** | 只用纯向量 | 失去 Milvus 主要优势 |

> **建议先试 C**：用 `pymilvus` 原生 API 直接验证混合检索能否跑通。
> 若原生 API 也不行，说明是 `milvus-lite` 服务端的问题，`langchain-milvus` 层面无解。

---

## 5. 收益与成本分析

### 5.1 Milvus 相对 Qdrant 的增量

| 能力 | Qdrant 本地 | Milvus Lite | 实测状态 |
|---|---|---|---|
| 向量检索 | ✅ | ✅ | 均可用 |
| 标量过滤 | ✅ | ✅ | 均可用 |
| **中文分词** | ❌ 无 | ✅ jieba | 未验证 |
| **BM25 混合检索** | ❌ 无 | ⚠️ 有但**接不通** | ❌ |
| 多 client 并发 | ❌ 独占锁 | ⚠️ 可开，安全性未知 | — |
| 迁移到生产 | 改 `url=` | 改 `uri=` | 均平滑 |

### 5.2 成本

| 项目 | 代价 |
|---|---|
| 依赖体积 | +82 MB（pyarrow）等 |
| 版本风险 | 三个包的版本错配，**混合检索不可用** |
| 新实现风险 | 纯 Python 重写，生态成熟度未知 |
| 迁移工时 | 约半天（含对比验证） |

### 5.3 决策建议

| 你的目标 | 建议 |
|---|---|
| 学习/理解 Milvus API | ✅ 值得试，成本低 |
| 解决"关键词检索召回差" | ⚠️ 本来指望 BM25，但**实测不通**，不如先调大 `TOP_K` |
| 上生产 | ⚠️ 建议用 **Milvus Standalone（Docker）**，而非 Lite |
| 保持现状可用 | ✅ 继续用 Qdrant |

---

## 6. 当前 Qdrant 方案的既有优势（对照参考）

迁移前值得明确知道**会失去什么**：

| 项 | 说明 |
|---|---|
| `TOP_K=12` 已验证 | 实测发现 4 会漏关键内容，12 最佳 |
| prompt 占用 39.4% | 1615 / 4096 token，余量充足 |
| 入库 20.7 秒 | 243 chunk，`langchain-ollama` 自动批量 |
| 已知局限 | 小模型辨析相近数字会错（见 PLAN.md 15.5.6） |

**其中"关键词检索召回差"这个痛点**，Milvus 的 BM25 本是对症的解法，但实测接不通。
**替代方案**：先靠 `TOP_K` 调优 + 后续加 Self-RAG 的 rewrite 节点改善。

---

## 7. 待验证事项

- [ ] `pymilvus` 原生 `hybrid_search` 能否绕开 `langchain-milvus` 跑通 BM25
- [ ] `pymilvus` 降级到 2.6.x 是否与 `milvus-lite 3.2.1` 兼容
- [ ] `milvus-lite` 的并发写是否安全（仅测了"能打开"）
- [ ] 纯 Python 实现的检索性能 vs Qdrant（需基准测试）
- [ ] jieba 中文分词对中文语料的实际效果（当前语料是英文论文）

---

## 8. 附：本次验证的可复现命令

```powershell
# 安装
uv pip install --python .\.venv\Scripts\python.exe pymilvus langchain-milvus milvus-lite

# 基础可用性
.\.venv\Scripts\python.exe -c "
from pymilvus import MilvusClient
c = MilvusClient(uri='./data/test.db')
c.create_collection(collection_name='t', dimension=1024, metric_type='COSINE')
print('OK', c.get_server_version() if hasattr(c,'get_server_version') else '')
"

# 混合检索（预期失败，用于复现问题）
# 需先建库，然后：
#   vs.similarity_search(q, k=4, ranker_type='rrf')
#   → MilvusException: weighted merge requires 2 weights, got 4
```

**注意**：`pymilvus 3.x` 要求本地库父目录已存在。

---

## 9. 临时文件说明

本次验证产生的临时文件**已全部清理**：

| 文件/目录 | 状态 |
|---|---|
| `src/_probe_milvus.py` | 已删除 |
| `src/_verify_milvus*.py` | 已删除 |
| `data/milvus_test.db` | 已删除 |
| `data/milvus_verify*.db` | 已删除 |

**Qdrant 主链路已复验完好**：`deepseek_v3` collection 存在，243 向量，检索正常。

⚠️ 但 **`pymilvus` / `langchain-milvus` / `milvus-lite` / `pyarrow` / `pandas` 等包仍在 venv 中**（共 11 个包）。
若决定不迁移，可执行以下命令卸载：

```powershell
uv pip uninstall --python .\.venv\Scripts\python.exe `
  pymilvus langchain-milvus milvus-lite faiss-cpu pyarrow pandas `
  python-dotenv cachetools
```

`requirements.txt` 也需相应重新生成。