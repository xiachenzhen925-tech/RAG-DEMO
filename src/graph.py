"""LangGraph 编排：retrieve → generate。

State 设计：
- question:  当前问题（多轮时由用户输入覆盖）
- context:   检索到的上下文文本（供 prompt 使用）
- sources:   结构化的来源信息（供 UI 显示引用）
- answer:    最终回答
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from retriever import format_context, get_llm, get_retriever

SYSTEM_PROMPT = """你是一个严谨的技术文档助手。请仅依据下面提供的【参考资料】回答问题。

要求：
1. 只使用参考资料中的信息，不要引入外部知识
2. 如果参考资料不足以回答，明确说"根据已有资料无法回答"，不要编造
3. 回答末尾用「来源：xxx 第 N 页」标注引用的资料
4. 用中文回答，技术术语可保留英文原文
5. 直接给出答案，不要描述你的思考过程

【参考资料】
{context}"""


class RAGState(TypedDict):
    """图的状态。

    messages 用 add_messages reducer，让 checkpointer 能正确累积多轮对话。
    其余字段每轮被节点覆盖（普通赋值语义）。
    """
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    context: str
    sources: list[dict]
    answer: str


def retrieve_node(state: RAGState) -> dict:
    """检索节点：用问题查向量库，组装上下文。"""
    question = state["question"]
    docs = get_retriever().invoke(question)

    sources = [
        {
            "source": d.metadata.get("source", "?"),
            "page": d.metadata.get("page", "?"),
            "preview": d.page_content[:80].replace("\n", " "),
        }
        for d in docs
    ]

    return {
        "context": format_context(docs),
        "sources": sources,
    }


def generate_node(state: RAGState) -> dict:
    """生成节点：拼 prompt，调 LLM。

    ⚠️ LLM 已在 retriever.get_llm() 中设 reasoning=False，
       否则 qwen3.5:9b 会只输出思考、content 为空。
    """
    llm = get_llm()

    system = SYSTEM_PROMPT.format(context=state["context"])
    # 只带历史对话，不带历史检索上下文（避免 prompt 膨胀 + 上下文串扰）
    history = state.get("messages") or []

    messages = [
        SystemMessage(content=system),
        *history,
        HumanMessage(content=state["question"]),
    ]

    response = llm.invoke(messages)
    answer = response.content if isinstance(response.content, str) else str(response.content)

    return {
        "answer": answer,
        "messages": [HumanMessage(content=state["question"]), AIMessage(content=answer)],
    }


def build_graph(*, checkpointer=None):
    """构建并编译图。

    checkpointer 传 InMemorySaver() 即获得多轮记忆；
    后续换 Redis 只需改这一个参数（见 PLAN.md Phase 4）。
    """
    builder = StateGraph(RAGState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")
    builder.add_edge("generate", END)

    return builder.compile(checkpointer=checkpointer or InMemorySaver())


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    graph = build_graph()
    cfg = {"configurable": {"thread_id": "test"}}

    for q in ["DeepSeek-V3 的总参数量和激活参数量分别是多少？",
              "它采用了什么注意力机制？"]:
        print(f"\n{'='*60}\n问: {q}\n{'='*60}")
        result = graph.invoke({"question": q}, config=cfg)
        print(f"答: {result['answer']}")
        print(f"\n来源:")
        for s in result["sources"]:
            print(f"  - {s['source']} 第 {s['page']} 页 | {s['preview']}...")