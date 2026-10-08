"""全局配置。

所有模型、地址、检索参数集中在此，便于单点修改。
实测数据见 PLAN.md 第 15 节。
"""

from pathlib import Path

# ---------------- 向量库后端选择 ----------------
# "qdrant" = 本机嵌入式（默认，已验证可用）
# "milvus" = 机器 B 上的 Milvus Standalone（含 BM25 混合检索）
#
# ⚠️ 默认保持 "qdrant"：Milvus 路线尚在 A/B 验证阶段，
#    切换只改这一行，Qdrant 数据完整保留可随时回退。
VECTOR_BACKEND = "milvus"

# ---------------- 路径 ----------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = PROJECT_DIR / "pdf"
QDRANT_PATH = str(DATA_DIR / "qdrant_storage")

# ---------------- Milvus（机器 B，走 Tailscale）----------------
MILVUS_URI = "http://100.89.245.44:19530"
# B 上的 Milvus 版本 3.0.1（实测）
#
# 混合检索开关：Milvus 支持 BM25 稠密+稀疏混合检索，
# 但 langchain-milvus 0.4.0 无法读取含 sparse 字段的表
# （报 "not allowed to retrieve raw data of field sparse"），
# 故必须走 pymilvus 原生 hybrid_search（见 retriever.MilvusHybridRetriever）。
USE_HYBRID_SEARCH = True
# "rrf" = Reciprocal Rank Fusion（无需调权重，推荐）
# "weighted" = 加权融合，需配 HYBRID_WEIGHTS
HYBRID_RANKER = "rrf"
HYBRID_WEIGHTS = [0.7, 0.3]   # 仅 weighted 时生效：(dense, sparse)

# ---------------- 远程 Ollama ----------------
# 机器 B 的 Tailscale IP，实测可用（Ollama 0.34.2）
OLLAMA_BASE_URL = "http://100.89.245.44:11434"

LLM_MODEL = "qwen3.5:9b"        # 实测 6.14 GB
EMBED_MODEL = "bge-m3:latest"   # 实测 1.08 GB
EMBED_DIM = 1024                # 实测确认；建库时锁死，变更需删库重建

# ---------------- 关键行为开关 ----------------
# ⚠️ 必须为 False。实测：qwen3.5:9b 默认开启思考模式，会只思考不回答
#    （content 为空字符串，耗时 13s vs 0.7s，差 18 倍）
#    LangChain 层参数名是 reasoning（对应 API 层的 think）
REASONING = False

# 生成参数
TEMPERATURE = 0.1
NUM_PREDICT = 512

# ---------------- 检索参数 ----------------
COLLECTION_NAME = "deepseek_v3"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ---------------- 噪声页过滤 ----------------
# 目录页与图表页会污染检索。实测（DeepSeek-V3 论文 53 页）：
#   噪声页占用 6.7% 的 Top-K 槽位，20 个测试问题中 9 个受影响
#   最严重一例：「有哪些附录？」12 个槽位中 6 个是噪声页
# 关掉可对比过滤效果，但建议保持开启。
FILTER_NOISE_PAGES = True
# 目录页判定：连续省略号数量 > TOC_DOTS_THRESHOLD
# 实测 p.2=244 个、p.3=147 个，正常页最多 3 个，分离度 80 倍
TOC_DOTS_THRESHOLD = 20
# 图表页判定：数字字符占比 > CHART_DIGIT_RATIO
# 实测 p.49-53 达 36%-49%
CHART_DIGIT_RATIO = 0.35

# TOP_K 需注意：embedding 检索的分数区分度低，
# Top-4 会漏掉关键细节（实测：MoE 路由机制因排在 5-8 位而丢失，
# 导致模型回答"无法回答"）。实测 K=8 → prompt 占 1615 token (39.4%)，
# 4096 上下文有余量，故取 12 以获得更好的召回。
#
# ⚠️ 已知局限：即使召回到正确 chunk，小模型仍可能答错相近数字。
#    实测例：p.22 明确"8 experts activated per token"，但 p.19 有
#    "9 experts"（解码阶段每 GPU 部署数），模型会张冠李戴。
#    这是 9B 模型的辨析能力限制，非检索问题。加大 K 或换更大模型可缓解。
TOP_K = 12

# 服务端上下文需容纳：system + 检索上下文(TOP_K*CHUNK_SIZE) + 问题 + 回答
# 4096 在 8GB 显存下是安全值（见 PLAN.md 4.2 节）
NUM_CTX = 4096

# ---------------- 超时 ----------------
# 首次请求需加载 6.14GB 模型到显存，且链路走 Tailscale 中继（单次往返 ~2.7s）
REQUEST_TIMEOUT = 180