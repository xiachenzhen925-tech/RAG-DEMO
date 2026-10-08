"""A/B 量化对比实验：Qdrant 纯向量 vs Milvus 混合检索。

设计要点：
1. 问题集带【标准答案关键词】，可自动判定回答正确性
2. 评估三个层次：
   - recall@K  ：标准答案所在的页码是否被召回
   - answer    ：回答是否包含关键事实
   - latency   ：耗时分布
3. 问题分类，看不同类别下两者差异：
   - 英文关键词型（BM25 理论优势区）
   - 中文语义型（BM25 理论无效区）
   - 数字精确型（易错区）
"""

import json
import statistics
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.messages import HumanMessage, SystemMessage

import config
from graph import SYSTEM_PROMPT
from retriever import _get_milvus_client, _get_qdrant_client, format_context, get_llm

# ============================================================
# 问题集：每条含 [类别, 问题, 标准答案页, 关键词列表]
# 关键词用于自动判定回答正确性
# ============================================================
CASES = [
    # --- 英文关键词型：BM25 理论优势区 ---
    ("EN-关键词", "What is node-limited routing?", [10, 13], ["node-limited", "节点", "受限"]),
    ("EN-关键词", "What is DualPipe?", [12, 13], ["DualPipe", "流水线", "pipeline"]),
    ("EN-关键词", "What is auxiliary-loss-free load balancing?", [9, 27], ["bias", "偏置", "负载"]),
    ("EN-关键词", "What is Multi-head Latent Attention?", [7, 8], ["MLA", "Latent", "压缩", "low-rank"]),
    ("EN-关键词", "What is Multi-Token Prediction?", [22, 23], ["MTP", "token", "预测"]),

    # --- 中文语义型：BM25 理论无效区 ---
    ("ZH-语义", "DeepSeek-V3 的总参数量是多少？", [1, 22], ["671", "37"]),
    ("ZH-语义", "它采用了什么注意力机制？", [7, 8], ["MLA", "Latent", "注意力"]),
    ("ZH-语义", "DeepSeek-V3 相比 V2 有什么改进？", [1, 4, 32], ["MLA", "DeepSeekMoE", "改进"]),
    ("ZH-语义", "训练用了多少 GPU 小时？", [5, 23], ["2.664", "H800", "GPU"]),
    ("ZH-语义", "什么是 FP8 混合精度训练？", [8, 9, 10], ["FP8", "量化", "精度"]),

    # --- 数字精确型：易错区（实测曾答错 8 vs 9）---
    ("数字精确", "每个 token 激活几个路由专家？", [22, 14], ["8"]),
    ("数字精确", "每个 token 最多被发送到几个节点？", [19, 10], ["4"]),
    ("数字精确", "DeepSeek-V3 有多少个路由专家？", [9, 22], ["256"]),
    ("数字精确", "MLA 有多少个注意力头？", [22], ["128"]),
]


def build_retriever(backend: str, k: int = 12):
    config.VECTOR_BACKEND = backend
    from retriever import get_qdrant_store

    get_qdrant_store.cache_clear()

    if backend == "qdrant":
        return get_qdrant_store().as_retriever(search_kwargs={"k": k})

    from retriever import MilvusHybridRetriever

    return MilvusHybridRetriever(k=k)


def judge_answer(answer: str, keywords: list[str]) -> bool:
    """回答是否包含任一关键词（宽松判定）。"""
    return any(kw.lower() in answer.lower() for kw in keywords)


def recall_hit(pages: list, gold_pages: list) -> bool:
    """召回页码是否命中标准页。"""
    return bool(set(pages) & set(gold_pages))


print("=" * 78)
print("A/B 量化对比实验")
print(f"问题数: {len(CASES)}  |  TOP_K = {config.TOP_K}")
print("=" * 78)

llm = get_llm()
report = {"qdrant": [], "milvus": []}

for backend in ("qdrant", "milvus"):
    print(f"\n{'#'*78}\n# 后端: {backend.upper()}\n{'#'*78}")
    retriever = build_retriever(backend)

    for tag, q, gold_pages, keywords in CASES:
        t0 = time.time()
        try:
            docs = retriever.invoke(q)
            t_ret = time.time() - t0
            pages = [d.metadata.get("page") for d in docs]

            ctx = format_context(docs)
            t1 = time.time()
            resp = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT.format(context=ctx)),
                HumanMessage(content=q),
            ])
            t_gen = time.time() - t1
            ans = resp.content if isinstance(resp.content, str) else str(resp.content)

            hit = recall_hit(pages, gold_pages)
            ok = judge_answer(ans, keywords)

            report[backend].append({
                "tag": tag, "q": q, "pages": pages,
                "recall": hit, "correct": ok,
                "t_ret": t_ret, "t_gen": t_gen,
            })
            print(f"  [{'R' if hit else '-'}{'C' if ok else '-'}] {tag:9} "
                  f"检索{t_ret:5.2f}s 生成{t_gen:5.1f}s | {q[:40]}")
        except Exception as e:
            print(f"  [ERROR] {tag:9} {q[:40]} -> {type(e).__name__}: {str(e)[:90]}")
            report[backend].append({
                "tag": tag, "q": q, "pages": [], "recall": False, "correct": False,
                "t_ret": 0, "t_gen": 0, "error": str(e)[:100],
            })

# ============================================================
# 汇总
# ============================================================
print(f"\n\n{'='*78}\n汇总统计\n{'='*78}")

summary = {}
for backend in ("qdrant", "milvus"):
    rows = report[backend]
    n = len(rows)
    recall = sum(1 for r in rows if r["recall"])
    correct = sum(1 for r in rows if r["correct"])
    lats = [r["t_ret"] for r in rows if r["t_ret"] > 0]
    summary[backend] = {
        "recall": recall, "correct": correct, "n": n,
        "lat_med": statistics.median(lats) if lats else 0,
        "lat_max": max(lats) if lats else 0,
    }

print(f"{'指标':<20}{'Qdrant':<16}{'Milvus':<16}{'差异'}")
print("-" * 62)
s_q, s_m = summary["qdrant"], summary["milvus"]
print(f"{'召回命中 R@12':<18}{s_q['recall']}/{s_q['n']:<12}{s_m['recall']}/{s_m['n']:<12}"
      f"{s_m['recall']-s_q['recall']:+d}")
print(f"{'回答正确':<19}{s_q['correct']}/{s_q['n']:<12}{s_m['correct']}/{s_m['n']:<12}"
      f"{s_m['correct']-s_q['correct']:+d}")
print(f"{'检索中位延迟':<17}{s_q['lat_med']:<15.2f}s{s_m['lat_med']:<15.2f}s"
      f"{s_m['lat_med']-s_q['lat_med']:+.2f}s")
print(f"{'检索最大延迟':<17}{s_q['lat_max']:<15.2f}s{s_m['lat_max']:<15.2f}s"
      f"{s_m['lat_max']-s_q['lat_max']:+.2f}s")

# 按类别细分
print(f"\n{'='*78}\n分类别对比\n{'='*78}")
CATS = ["EN-关键词", "ZH-语义", "数字精确"]
print(f"{'类别':<14}{'Qdrant R/C':<18}{'Milvus R/C':<18}{'结论'}")
print("-" * 62)
for cat in CATS:
    for backend in ("qdrant", "milvus"):
        rows = [r for r in report[backend] if r["tag"] == cat]
        key = f"{sum(1 for r in rows if r['recall'])}/{sum(1 for r in rows if r['correct'])}"
        if backend == "qdrant":
            qk = key
        else:
            mk = key
    r_q, c_q = qk.split("/")
    r_m, c_m = mk.split("/")
    diff_r = int(r_m) - int(r_q)
    diff_c = int(c_m) - int(c_q)
    verdict = "Milvus 优" if (diff_r > 0 or diff_c > 0) else ("相同" if diff_r == 0 and diff_c == 0 else "Qdrant 优")
    print(f"{cat:<14}{qk:<18}{mk:<18}{verdict}")

# 输赢明细
print(f"\n{'='*78}\n逐题差异明细（只看两者不同的）\n{'='*78}")
for i, (tag, q, _, _) in enumerate(CASES):
    q_r = report["qdrant"][i]
    m_r = report["milvus"][i]
    if q_r["correct"] != m_r["correct"] or q_r["recall"] != m_r["recall"]:
        print(f"\n[{tag}] {q}")
        print(f"  Qdrant: R={q_r['recall']} C={q_r['correct']} pages={q_r['pages'][:6]}")
        print(f"  Milvus: R={m_r['recall']} C={m_r['correct']} pages={m_r['pages'][:6]}")

# 保存原始数据
out = config.DATA_DIR / "ab_report.json"
out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n原始数据已保存: {out}")

for g in (_get_qdrant_client, _get_milvus_client):
    try:
        g().close()
    except Exception:
        pass