"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import json
import math
import os

try:
    from .task4_chunking_indexing import (
        EMBEDDING_MODEL,
        VECTOR_INDEX_PATH,
        _hash_embedding,
        run_pipeline,
    )
except ImportError:  # Cho phép chạy trực tiếp: python src/task5_semantic_search.py
    from task4_chunking_indexing import (
        EMBEDDING_MODEL,
        VECTOR_INDEX_PATH,
        _hash_embedding,
        run_pipeline,
    )


_MODEL = None


def _load_records() -> list[dict]:
    """Load local JSONL vector store, building it once if needed."""
    if not VECTOR_INDEX_PATH.exists():
        run_pipeline()

    if not VECTOR_INDEX_PATH.exists():
        return []

    records = []
    with VECTOR_INDEX_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("content") and record.get("embedding"):
                records.append(record)

    return records


def _embed_query(query: str, records: list[dict]) -> list[float]:
    """Embed query with the same practical backend used by Task 4."""
    first_model = ""
    if records:
        first_model = records[0].get("metadata", {}).get("embedding_model", "")

    use_sentence_transformers = (
        os.getenv("TASK4_EMBEDDING_BACKEND") == "sentence-transformers"
        and first_model != "hash-fallback"
    )
    if use_sentence_transformers:
        try:
            global _MODEL
            if _MODEL is None:
                from sentence_transformers import SentenceTransformer

                _MODEL = SentenceTransformer(EMBEDDING_MODEL)
            return _MODEL.encode(query, normalize_embeddings=True).tolist()
        except Exception:
            pass

    return _hash_embedding(query)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity for normalized or raw vectors."""
    if not left or not right:
        return 0.0

    length = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(length))
    left_norm = math.sqrt(sum(value * value for value in left[:length]))
    right_norm = math.sqrt(sum(value * value for value in right[:length]))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    records = _load_records()
    if not records:
        return []

    query_embedding = _embed_query(query, records)
    results = []
    for record in records:
        score = _cosine_similarity(query_embedding, record["embedding"])
        results.append({
            "content": record["content"],
            "score": float(score),
            "metadata": record.get("metadata", {}),
        })

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
