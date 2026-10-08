r"""RAG 问答 CLI。

用法：
    .\.venv\Scripts\python.exe src\cli.py              # 交互模式
    .\.venv\Scripts\python.exe src\cli.py "问题"        # 单次提问
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.messages import AIMessage

from graph import build_graph
from retriever import _get_milvus_client, _get_qdrant_client


def print_sources(result: dict) -> None:
    sources = result.get("sources") or []
    if not sources:
        return
    print("\n来源:")
    seen = set()
    for s in sources:
        key = (s["source"], s["page"])
        if key in seen:
            continue
        seen.add(key)
        print(f"  · {s['source']} 第 {s['page']} 页 — {s['preview']}...")


def cleanup() -> None:
    """显式关闭向量库 client，避免解释器退出时的析构噪音。

    Qdrant 不关的话，QdrantClient.__del__ 会在 sys.meta_path 已清空后
    尝试 close，抛出无害但刺眼的 ImportError。
    Milvus 走网络连接，同样显式关闭更干净。
    """
    for getter in (_get_qdrant_client, _get_milvus_client):
        try:
            getter().close()
        except Exception:
            pass


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    cfg = {"configurable": {"thread_id": thread_id}}

    # 单次提问模式
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
        result = graph.invoke({"question": q}, config=cfg)
        print(result["answer"])
        print_sources(result)
        cleanup()
        return

    # 交互模式
    print("=" * 60)
    print("RAG 问答（基于 DeepSeek-V3 技术报告）")
    print("输入问题回车提问；exit / quit 退出；clear 清空对话历史")
    print(f"会话 ID: {thread_id[:8]}")
    print("=" * 60)

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            break
        if q.lower() == "clear":
            # 换 thread_id 即开新会话，等价于清空记忆
            thread_id = str(uuid.uuid4())
            cfg = {"configurable": {"thread_id": thread_id}}
            print(f"已清空对话历史（新会话 {thread_id[:8]}）")
            continue

        try:
            result = graph.invoke({"question": q}, config=cfg)
        except Exception as e:
            print(f"\n❌ 出错: {type(e).__name__}: {e}")
            continue

        print(f"\n{result['answer']}")
        print_sources(result)

    cleanup()
    print("再见。")


if __name__ == "__main__":
    main()