"""向量检索与模型封装。

集中构造 LLM / Embeddings / VectorStore，避免各处重复。

支持两种向量库后端（config.VECTOR_BACKEND）：
- "qdrant"：QdrantVectorStore（langchain 封装，本机嵌入式）
- "milvus"：MilvusHybridRetriever（pymilvus 原生 hybrid_search，含 BM25）

两者对上层暴露同一个接口：get_retriever().invoke(question) -> list[Document]
因此 graph.py / cli.py 无需感知后端差异。
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from config import (
    COLLECTION_NAME,
    EMBED_MODEL,
    HYBRID_RANKER,
    HYBRID_WEIGHTS,
    LLM_MODEL,
    MILVUS_URI,
    NUM_CTX,
    NUM_PREDICT,
    OLLAMA_BASE_URL,
    QDRANT_PATH,
    REASONING,
    REQUEST_TIMEOUT,
    TEMPERATURE,
    TOP_K,
    USE_HYBRID_SEARCH,
    VECTOR_BACKEND,
)


# ============================================================
# 模型
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings() -> OllamaEmbeddings:
    """嵌入模型。lru_cache 保证全局单例，避免重复建连接。"""
    return OllamaEmbeddings(
        model=EMBED_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_ctx=NUM_CTX,
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """对话模型。

    ⚠️ reasoning=False 是关键：qwen3.5:9b 默认开启思考模式，
       会导致 content 为空（实测 13s 返回空 vs 0.7s 正常返回）。
    """
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=TEMPERATURE,
        num_predict=NUM_PREDICT,
        num_ctx=NUM_CTX,
        reasoning=REASONING,
        # 首次请求需加载 6GB 模型，且链路走中继，超时给足
        client_kwargs={"timeout": REQUEST_TIMEOUT},
    )


# ============================================================
# 后端 1：Qdrant（langchain 封装）
# ============================================================

@lru_cache(maxsize=1)
def _get_qdrant_client() -> QdrantClient:
    """Qdrant 本地 client 单例。

    ⚠️ 本地模式对存储目录加独占锁，整个进程只能有一个 client 指向该路径，
       否则报 "already accessed by another instance"。故必须复用此单例。
    """
    return QdrantClient(path=QDRANT_PATH)


@lru_cache(maxsize=1)
def get_qdrant_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        client=_get_qdrant_client(),
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )


# ============================================================
# 后端 2：Milvus（pymilvus 原生，支持 BM25 混合检索）
# ============================================================

@lru_cache(maxsize=1)
def _get_milvus_client():
    from pymilvus import MilvusClient

    return MilvusClient(uri=MILVUS_URI, timeout=REQUEST_TIMEOUT)


class MilvusHybridRetriever(BaseRetriever):
    """用 pymilvus 原生 hybrid_search 的检索器。

    为什么不用 langchain-milvus？
        实测 langchain-milvus 0.4.0 无法读取含 sparse 字段的表，所有检索路径
        均报 "not allowed to retrieve raw data of field sparse"。
        而 pymilvus 原生 hybrid_search 完全正常（实测 0.01s）。

    本类继承 BaseRetriever，实现 _get_relevant_documents，
    对上层暴露与其他 Retriever 完全一致的接口。
    """

    k: int = TOP_K
    use_hybrid: bool = USE_HYBRID_SEARCH

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker

        client = _get_milvus_client()
        qvec = get_embeddings().embed_query(query)

        OUTPUT = ["text", "source", "page"]

        # --- 非混合模式：走普通 search，避免 sparse 字段引起的兼容问题 ---
        if not self.use_hybrid:
            res = client.search(
                collection_name=COLLECTION_NAME,
                data=[qvec],
                anns_field="vector",
                limit=self.k,
                output_fields=OUTPUT,
            )
            hits = res[0]
        # --- 混合模式：稠密 + BM25 稀疏，用 ranker 融合 ---
        else:
            reqs = [
                AnnSearchRequest(
                    data=[qvec], anns_field="vector", param={}, limit=self.k
                ),
                AnnSearchRequest(
                    data=[query], anns_field="sparse", param={}, limit=self.k
                ),
            ]
            ranker = (
                WeightedRanker(*HYBRID_WEIGHTS)
                if HYBRID_RANKER == "weighted"
                else RRFRanker()
            )
            res = client.hybrid_search(
                collection_name=COLLECTION_NAME,
                reqs=reqs,
                ranker=ranker,
                limit=self.k,
                output_fields=OUTPUT,
            )
            hits = res[0]

        return [
            Document(
                page_content=h["entity"].get("text", ""),
                metadata={
                    "source": h["entity"].get("source", "?"),
                    "page": h["entity"].get("page", "?"),
                    "score": float(h.get("distance", 0.0)),
                },
            )
            for h in hits
        ]


# ============================================================
# 统一入口
# ============================================================

def get_vectorstore():
    """返回当前后端的 VectorStore（仅 Qdrant 后端适用）。"""
    if VECTOR_BACKEND != "qdrant":
        raise RuntimeError(
            f"get_vectorstore() 仅支持 qdrant 后端，当前为 {VECTOR_BACKEND}。"
            "Milvus 后端请使用 get_retriever()。"
        )
    return get_qdrant_store()


def get_retriever(k: int = TOP_K) -> BaseRetriever:
    """按 config.VECTOR_BACKEND 返回检索器。

    两种后端返回的检索器接口一致：
        retriever.invoke(question) -> list[Document]
    """
    if VECTOR_BACKEND == "qdrant":
        return get_qdrant_store().as_retriever(search_kwargs={"k": k})

    if VECTOR_BACKEND == "milvus":
        return MilvusHybridRetriever(k=k)

    raise ValueError(f"未知后端: {VECTOR_BACKEND}")


def format_context(docs) -> str:
    """把检索结果拼成带页码标注的上下文，便于模型引用与用户溯源。"""
    parts = []
    for d in docs:
        src = d.metadata.get("source", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[来源: {src} 第 {page} 页]\n{d.page_content}")
    return "\n\n---\n\n".join(parts)