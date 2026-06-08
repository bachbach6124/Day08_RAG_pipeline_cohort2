"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt, bỏ dấu câu nhưng giữ chữ có dấu."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _cosine_sim(left: list[float], right: list[float]) -> float:
    """Cosine similarity cho MMR."""
    if not left or not right:
        return 0.0

    length = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _lexical_relevance(query: str, content: str) -> float:
    """
    Lightweight local reranking score.

    Dùng token overlap có IDF giả lập trong phạm vi query, cộng phrase bonus để
    ưu tiên chunk chứa cụm từ truy vấn liên tiếp. Cách này chạy offline, không
    cần API key, và đủ ổn định cho pipeline/test local.
    """
    query_tokens = _tokenize(query)
    content_tokens = _tokenize(content)
    if not query_tokens or not content_tokens:
        return 0.0

    query_counts = Counter(query_tokens)
    content_counts = Counter(content_tokens)
    unique_query_tokens = set(query_tokens)

    overlap_score = 0.0
    for token in unique_query_tokens:
        if token not in content_counts:
            continue
        tf = content_counts[token] / len(content_tokens)
        query_weight = 1.0 + query_counts[token] / len(query_tokens)
        overlap_score += query_weight * math.log1p(content_counts[token]) + tf

    coverage = sum(1 for token in unique_query_tokens if token in content_counts) / len(unique_query_tokens)
    normalized_overlap = overlap_score / len(unique_query_tokens)

    content_text = " ".join(content_tokens)
    phrase_bonus = 0.0
    for ngram_size in range(min(4, len(query_tokens)), 1, -1):
        ngrams = [
            " ".join(query_tokens[index:index + ngram_size])
            for index in range(len(query_tokens) - ngram_size + 1)
        ]
        if any(ngram in content_text for ngram in ngrams):
            phrase_bonus = 0.15 * ngram_size
            break

    return normalized_overlap + coverage + phrase_bonus


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if top_k <= 0 or not candidates:
        return []

    reranked = []
    for rank, candidate in enumerate(candidates):
        content = str(candidate.get("content", ""))
        original_score = float(candidate.get("score", 0.0) or 0.0)
        relevance_score = _lexical_relevance(query, content)

        item = candidate.copy()
        # Blend retrieval confidence with local query-document relevance.
        item["score"] = float(0.75 * relevance_score + 0.25 * original_score)
        item.setdefault("metadata", {})
        item["rerank_method"] = "local_lexical"
        item["_original_rank"] = rank
        reranked.append(item)

    reranked.sort(key=lambda item: (item["score"], -item["_original_rank"]), reverse=True)
    for item in reranked:
        item.pop("_original_rank", None)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    if top_k <= 0 or not candidates:
        return []

    lambda_param = max(0.0, min(1.0, lambda_param))
    selected: list[int] = []
    remaining = [
        index for index, candidate in enumerate(candidates)
        if candidate.get("embedding")
    ]

    for _ in range(min(top_k, len(remaining))):
        best_idx = remaining[0]
        best_score = float("-inf")

        for idx in remaining:
            candidate_embedding = candidates[idx].get("embedding", [])
            relevance = _cosine_sim(query_embedding, candidate_embedding)

            max_sim_to_selected = 0.0
            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding", [])
                max_sim_to_selected = max(
                    max_sim_to_selected,
                    _cosine_sim(candidate_embedding, selected_embedding),
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["score"] = float(_cosine_sim(query_embedding, item.get("embedding", [])))
        item["rerank_method"] = "mmr"
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    if top_k <= 0:
        return []

    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            content = str(item.get("content", ""))
            if not content:
                continue

            key = content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            if key not in content_map:
                content_map[key] = item
            elif item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda pair: pair[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        item.setdefault("metadata", {})
        item["rerank_method"] = "rrf"
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Interface chính không nhận query embedding, nên fallback sang local scorer.
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k=top_k)
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
