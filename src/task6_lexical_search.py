"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import json
import math
import re
from collections import Counter

try:
    from .task4_chunking_indexing import (
        VECTOR_INDEX_PATH,
        chunk_documents,
        load_documents,
        run_pipeline,
    )
except ImportError:  # Cho phép chạy trực tiếp: python src/task6_lexical_search.py
    from task4_chunking_indexing import (
        VECTOR_INDEX_PATH,
        chunk_documents,
        load_documents,
        run_pipeline,
    )


CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
BM25_INDEX = None


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản, giữ tốt từ tiếng Việt có dấu và bỏ dấu câu."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


class SimpleBM25:
    """Fallback BM25 nhỏ gọn khi môi trường chưa cài rank-bm25."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]

        document_frequency = Counter()
        for doc in tokenized_corpus:
            document_frequency.update(set(doc))

        corpus_size = len(tokenized_corpus)
        self.idf = {
            term: math.log(1 + (corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in document_frequency.items()
        }

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        if not self.corpus or not query_tokens:
            return [0.0] * len(self.corpus)

        scores = []
        for idx, term_freq in enumerate(self.term_freqs):
            score = 0.0
            doc_len = self.doc_len[idx]
            for token in query_tokens:
                tf = term_freq.get(token, 0)
                if tf == 0:
                    continue

                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf.get(token, 0.0) * (tf * (self.k1 + 1)) / denominator
            scores.append(score)

        return scores


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [_tokenize(doc.get("content", "")) for doc in corpus]

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except Exception:
        return SimpleBM25(tokenized_corpus)


def _load_corpus_from_vector_store() -> list[dict]:
    """Ưu tiên dùng chunks đã index ở Task 4 để search cùng đơn vị retrieval."""
    if not VECTOR_INDEX_PATH.exists():
        try:
            run_pipeline()
        except Exception:
            pass

    if not VECTOR_INDEX_PATH.exists():
        return []

    corpus = []
    with VECTOR_INDEX_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)
            content = record.get("content", "").strip()
            if not content:
                continue

            corpus.append({
                "content": content,
                "metadata": record.get("metadata", {}),
            })

    return corpus


def _load_corpus() -> list[dict]:
    corpus = _load_corpus_from_vector_store()
    if corpus:
        return corpus

    documents = load_documents()
    return chunk_documents(documents) if documents else []


def _get_index():
    global BM25_INDEX, CORPUS

    if BM25_INDEX is None:
        CORPUS = _load_corpus()
        BM25_INDEX = build_bm25_index(CORPUS)

    return BM25_INDEX


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    if top_k <= 0 or not query.strip():
        return []

    bm25 = _get_index()
    if not CORPUS:
        return []

    scores = bm25.get_scores(_tokenize(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    results = []
    for index in ranked_indices[:top_k]:
        results.append({
            "content": CORPUS[index]["content"],
            "score": float(scores[index]),
            "metadata": CORPUS[index].get("metadata", {}),
        })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
