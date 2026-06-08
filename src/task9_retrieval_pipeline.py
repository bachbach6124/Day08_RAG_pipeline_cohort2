"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results
"""

from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # "cross_encoder" | "mmr" | "rrf"


def _safe_search(search_fn, query: str, top_k: int) -> list[dict]:
    """Run one retrieval backend without letting it take down the pipeline."""
    try:
        results = search_fn(query, top_k=top_k)
    except Exception:
        return []

    return results if isinstance(results, list) else []


def _with_source(results: list[dict], source: str) -> list[dict]:
    """Normalize result shape and attach the retrieval source marker."""
    normalized = []
    for item in results:
        if not isinstance(item, dict) or not item.get("content"):
            continue

        normalized_item = {
            "content": str(item.get("content", "")),
            "score": float(item.get("score", 0.0) or 0.0),
            "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            "source": source,
        }

        for key, value in item.items():
            if key not in normalized_item:
                normalized_item[key] = value

        normalized.append(normalized_item)

    return normalized


def _fallback_pageindex(query: str, top_k: int) -> list[dict]:
    """Query PageIndex/vectorless fallback and normalize its output."""
    fallback_results = _safe_search(pageindex_search, query, top_k=top_k)
    return _with_source(fallback_results, "pageindex")[:top_k]


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → results_dense
          ├→ Lexical Search  → results_sparse
          │
          ├→ Merge (RRF) → merged_results
          ├→ Rerank → reranked_results
          │
          └→ If best_score < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm tối thiểu cho hybrid results
        use_reranking: Có áp dụng reranking hay không

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    query = query.strip()
    if top_k <= 0 or not query:
        return []

    pool_size = max(top_k * 4, 10)

    dense_results = _with_source(
        _safe_search(semantic_search, query, top_k=pool_size),
        "semantic",
    )
    sparse_results = _with_source(
        _safe_search(lexical_search, query, top_k=pool_size),
        "lexical",
    )

    merged = rerank_rrf([dense_results, sparse_results], top_k=pool_size)
    merged = _with_source(merged, "hybrid")

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = sorted(
            merged,
            key=lambda item: float(item.get("score", 0.0) or 0.0),
            reverse=True,
        )[:top_k]

    final_results = _with_source(final_results, "hybrid")[:top_k]
    best_score = final_results[0]["score"] if final_results else 0.0

    if not final_results or best_score < score_threshold:
        fallback_results = _fallback_pageindex(query, top_k=top_k)
        if fallback_results:
            return fallback_results

    return final_results


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
