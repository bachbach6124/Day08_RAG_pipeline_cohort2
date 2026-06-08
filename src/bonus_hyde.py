"""
Bonus: HyDE (Hypothetical Document Embeddings) query expansion.

HyDE creates a short hypothetical answer/document from the query, then retrieves
against query + hypothetical text. With local TF-IDF/cosine retrieval this works
as deterministic query expansion; with a real embedding model it follows the
same idea from the HyDE paper.
"""

from __future__ import annotations

from src.task5_semantic_search import semantic_search
from src.task7_reranking import rerank, rerank_rrf
from src.task6_lexical_search import lexical_search


LEGAL_HINTS = (
    "pháp luật ma túy chất cấm cai nghiện người sử dụng trái phép "
    "tàng trữ vận chuyển mua bán tổ chức sử dụng hình phạt trách nhiệm "
    "Luật Phòng chống ma túy Nghị định Bộ luật Hình sự"
)

NEWS_HINTS = (
    "nghệ sĩ ca sĩ diễn viên người mẫu showbiz bị bắt tạm giữ truy tố "
    "sử dụng ma túy tàng trữ tổ chức sử dụng trái phép chất ma túy"
)


def generate_hypothetical_document(query: str) -> str:
    """Create a deterministic Vietnamese hypothetical document for retrieval."""
    lowered = query.lower()
    if any(term in lowered for term in ["nghệ", "ca sĩ", "diễn viên", "người mẫu", "showbiz"]):
        hints = NEWS_HINTS
    elif any(term in lowered for term in ["luật", "nghị định", "hình phạt", "cai nghiện"]):
        hints = LEGAL_HINTS
    else:
        hints = f"{LEGAL_HINTS} {NEWS_HINTS}"

    return (
        "Tài liệu giả định trả lời câu hỏi về "
        f"{query}. Nội dung liên quan có thể bao gồm: {hints}."
    )


def build_hyde_query(query: str) -> str:
    """Combine the original query with a hypothetical document."""
    return f"{query}\n\n{generate_hypothetical_document(query)}"


def hyde_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Retrieve with HyDE expansion, then merge semantic and lexical evidence.

    Returns the same result format as Task 9 retrieval.
    """
    expanded_query = build_hyde_query(query)
    dense = semantic_search(expanded_query, top_k=top_k * 2)
    sparse = lexical_search(query, top_k=top_k * 2)
    merged = rerank_rrf([dense, sparse], top_k=top_k * 2)
    reranked = rerank(query, merged, top_k=top_k)
    for item in reranked:
        item["source"] = "hyde"
        item.setdefault("metadata", {})
        item["metadata"]["hyde"] = True
    return reranked
