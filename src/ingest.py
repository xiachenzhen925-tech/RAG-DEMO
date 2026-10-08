"""文档加载与切块。

用 PyMuPDF 提取 PDF 文本（实测比 pypdf 更稳），再按中文友好方式切块。

包含噪声页过滤：目录页与图表页会污染检索，见 filter_noise_pages()。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, PDF_DIR


# ============================================================
# 噪声页识别
# ============================================================

def count_toc_dots(text: str) -> int:
    """统计目录式的连续省略号数量。

    实测依据（DeepSeek-V3 论文 53 页）：
        目录页 p.2 = 244 个, p.3 = 147 个
        正常内容页最多 3 个
    分离度达 80 倍，是极可靠的判别特征。
    """
    return len(re.findall(r"\.\s*\.\s*\.", text))


def is_toc_page(text: str, threshold: int | None = None) -> bool:
    """是否为目录页。"""
    if threshold is None:
        from config import TOC_DOTS_THRESHOLD

        threshold = TOC_DOTS_THRESHOLD
    return count_toc_dots(text) > threshold


def is_chart_page(text: str, ratio_threshold: float | None = None) -> bool:
    """是否为低信息密度的图表页。

    实测依据：该论文 p.49-53 的数字字符占比达 36%-49%，
    正文极少（多为图表坐标轴数值），检索价值低。
    """
    if not text:
        return False
    if ratio_threshold is None:
        from config import CHART_DIGIT_RATIO

        ratio_threshold = CHART_DIGIT_RATIO
    digits = sum(c.isdigit() for c in text)
    return digits / len(text) > ratio_threshold


def classify_page(text: str) -> str:
    """返回页面类型："toc" | "chart" | "content"。"""
    if is_toc_page(text):
        return "toc"
    if is_chart_page(text):
        return "chart"
    return "content"


def extract_pdf(path: Path, *, filter_noise: bool = True) -> list[Document]:
    """提取单个 PDF 的文本，按页生成 Document。

    逐页保留 page 元数据，便于回答时溯源到具体页码。

    filter_noise=True 时跳过目录页和图表页。

    为什么需要过滤（实测数据）：
        噪声页占用了 6.7% 的 Top-K 召回槽位，20 个测试问题中 9 个受影响。
        最严重的一例：「有哪些附录？」12 个槽位中 6 个是目录/噪声页，
        近半数上下文被浪费。
    """
    docs: list[Document] = []
    skipped: list[tuple[int, str]] = []

    with pymupdf.open(path) as pdf:
        for page_no, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if not text:
                continue

            if filter_noise:
                kind = classify_page(text)
                if kind != "content":
                    skipped.append((page_no, kind))
                    continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "page": page_no,
                        "total_pages": pdf.page_count,
                    },
                )
            )

    if skipped:
        toc = [p for p, k in skipped if k == "toc"]
        chart = [p for p, k in skipped if k == "chart"]
        parts = []
        if toc:
            parts.append(f"目录页 {toc}")
        if chart:
            parts.append(f"图表页 {chart}")
        print(f"    已过滤噪声页: {'; '.join(parts)}")

    return docs


def load_documents(pdf_dir: Path = PDF_DIR) -> list[Document]:
    """加载目录下所有 PDF。"""
    from config import FILTER_NOISE_PAGES

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"未在 {pdf_dir} 找到 PDF 文件")

    all_docs: list[Document] = []
    for pdf in pdfs:
        docs = extract_pdf(pdf, filter_noise=FILTER_NOISE_PAGES)
        chars = sum(len(d.page_content) for d in docs)
        print(f"  {pdf.name}: {len(docs)} 页, {chars} 字符")
        if not docs:
            print(f"    ⚠️ 无文本层（扫描件），已跳过。需先 OCR。")
            continue
        all_docs.extend(docs)
    return all_docs


def split_documents(docs: list[Document]) -> list[Document]:
    """切块。

    分隔符优先级针对技术文档设计：先按段落，再按中英文句末标点，
    最后才退化到硬切。避免把句子从中间切断。
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",      # 段落
            "\n",        # 换行
            "。",         # 中文句号
            "！",
            "？",
            ". ",        # 英文句末
            "! ",
            "? ",
            "；",
            "; ",
            "，",
            ", ",
            " ",         # 词
            "",          # 硬切兜底
        ],
    )
    chunks = splitter.split_documents(docs)
    # 过滤过短碎片（多为页眉页脚、图表残留）
    return [c for c in chunks if len(c.page_content.strip()) >= 50]


def reset_qdrant_storage() -> None:
    """彻底清空 Qdrant 本地存储。

    ⚠️ 为什么不用 client.delete_collection()：
        实测发现它报告成功但数据仍在。连续重建后 collection 出现
        457 = 243(旧) + 214(新) 的脏状态，且新数据全部重复一遍。
        因此改为关闭 client、删掉整个存储目录，保证干净重建。

    代价：会一并删除该目录下所有 collection（当前只有 deepseek_v3）。
    """
    import shutil

    from config import QDRANT_PATH
    from retriever import _get_qdrant_client

    path = Path(QDRANT_PATH)

    # 先关闭单例 client 释放独占锁，否则删不掉
    try:
        _get_qdrant_client().close()
    except Exception:
        pass
    _get_qdrant_client.cache_clear()

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        print(f"  已清空 Qdrant 存储目录: {path.name}")

    path.mkdir(parents=True, exist_ok=True)


def build_qdrant(chunks: list[Document], *, recreate: bool = True):
    """把 chunks 嵌入并写入 Qdrant 本地库。

    recreate=True 时彻底清空存储目录再重建 —— 维度或模型变更后必须重建，
    因为 Qdrant 的向量维度在创建时即固定，无法原地修改。

    ⚠️ 不用 QdrantVectorStore.from_documents()：
       它的 from_texts 只接受 path/url，会在内部**再建一个 client**，
       而 Qdrant 本地模式对存储目录加独占锁，于是报
       "already accessed by another instance of Qdrant client"。
       正确做法是复用单例 client，用构造函数 + add_documents。
    """
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import models

    from config import COLLECTION_NAME, EMBED_DIM
    from retriever import _get_qdrant_client, get_embeddings

    if recreate:
        reset_qdrant_storage()

    client = _get_qdrant_client()
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=EMBED_DIM,
            distance=models.Distance.COSINE,
        ),
    )
    print(f"  已创建 collection: {COLLECTION_NAME} (dim={EMBED_DIM}, Cosine)")

    store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings(),
    )

    # 分批写入：Qdrant 本地模式一次性写太多会占大量内存
    batch = 64
    for i in range(0, len(chunks), batch):
        store.add_documents(chunks[i : i + batch])
        done = min(i + batch, len(chunks))
        print(f"  写入 {done}/{len(chunks)}", end="\r")
    print()

    return store


def build_milvus(chunks: list[Document], *, recreate: bool = True):
    """把 chunks 写入 Milvus（机器 B），并挂上 BM25 函数以支持混合检索。

    与 Qdrant 的关键差异：
        1. 走网络服务，不存在本地独占锁问题
        2. 用 from_documents 即可（内部自建 client 无妨）
        3. 挂 BM25BuiltInFunction 后，表里会有 2 个向量字段：
           vector（稠密 1024 维）+ sparse（BM25 稀疏）

    ⚠️ 挂了 BM25 后，langchain-milvus 自己的检索路径会报
       "not allowed to retrieve raw data of field sparse"，
       因此检索必须走 retriever.MilvusHybridRetriever（pymilvus 原生）。
    """
    from langchain_milvus import BM25BuiltInFunction, Milvus

    from config import COLLECTION_NAME, MILVUS_URI, USE_HYBRID_SEARCH
    from retriever import get_embeddings

    kwargs = {}
    if USE_HYBRID_SEARCH:
        kwargs["builtin_function"] = BM25BuiltInFunction()
        print("  启用 BM25 函数（稠密 + 稀疏混合检索）")

    t0 = __import__("time").time()
    store = Milvus.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=COLLECTION_NAME,
        connection_args={"uri": MILVUS_URI},
        drop_old=recreate,
        **kwargs,
    )
    print(f"  写入 {len(chunks)} 条，耗时 {__import__('time').time()-t0:.1f}s")

    return store


def build_vectorstore(chunks: list[Document], *, recreate: bool = True):
    """按 config.VECTOR_BACKEND 分发到对应后端。"""
    from config import VECTOR_BACKEND

    if VECTOR_BACKEND == "qdrant":
        return build_qdrant(chunks, recreate=recreate)
    if VECTOR_BACKEND == "milvus":
        return build_milvus(chunks, recreate=recreate)
    raise ValueError(f"未知后端: {VECTOR_BACKEND}")


def self_check(query: str, k: int = 2) -> None:
    """用统一检索接口自检，验证当前后端可查。"""
    from retriever import get_retriever

    docs = get_retriever(k=k).invoke(query)
    print(f"  查询: {query}")
    for i, d in enumerate(docs, 1):
        print(f"  [{i}] p.{d.metadata.get('page')} | {d.page_content[:90]}...")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")

    import time

    from config import (
        EMBED_DIM,
        EMBED_MODEL,
        OLLAMA_BASE_URL,
        QDRANT_PATH,
        USE_HYBRID_SEARCH,
        VECTOR_BACKEND,
    )

    print("=== 1. 加载 PDF ===")
    docs = load_documents()

    print("\n=== 2. 切块 ===")
    chunks = split_documents(docs)
    print(f"  {len(docs)} 页 → {len(chunks)} 个 chunk")
    print(f"  平均长度: {sum(len(c.page_content) for c in chunks) // len(chunks)} 字符")

    print(f"\n=== 3. 入库（后端 = {VECTOR_BACKEND}）===")
    print(f"  嵌入模型: {EMBED_MODEL} @ {OLLAMA_BASE_URL}")
    print(f"  向量维度: {EMBED_DIM}")
    if VECTOR_BACKEND == "qdrant":
        print(f"  Qdrant 路径: {QDRANT_PATH}")
    else:
        from config import MILVUS_URI

        print(f"  Milvus URI : {MILVUS_URI}")
        print(f"  混合检索   : {USE_HYBRID_SEARCH}")

    t0 = time.time()
    build_vectorstore(chunks)
    print(f"  ✅ 完成，耗时 {time.time()-t0:.1f}s")

    print("\n=== 4. 自检检索 ===")
    self_check("DeepSeek-V3 有多少参数？")

    # 显式关闭，避免析构噪音
    from retriever import _get_milvus_client, _get_qdrant_client

    for getter in (_get_qdrant_client, _get_milvus_client):
        try:
            getter().close()
        except Exception:
            pass

    print(f"\n总耗时: {time.time()-t0:.1f}s")