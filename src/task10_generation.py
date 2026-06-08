"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"
"""

import os
import re
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Answer the following question comprehensively in Vietnamese.
For every statement of fact or claim, immediately insert a citation in brackets
linking to the specific source (e.g., [Luật Phòng chống ma tuý 2021, Điều 3]
or [VnExpress, 2024]).

If the information is not explicitly stated in the provided context or knowledge
base, state 'Tôi không thể xác minh thông tin này từ nguồn hiện có' rather than
guessing.

Rules:
- Only use information from the provided context
- Every factual claim MUST have a citation
- If context is insufficient, say so clearly
- Structure your answer with clear paragraphs"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return list(chunks)

    reordered = []
    for index in range(0, len(chunks), 2):
        reordered.append(chunks[index])

    last_even_index = len(chunks) - 1 if len(chunks) % 2 == 0 else len(chunks) - 2
    for index in range(last_even_index, 0, -2):
        reordered.append(chunks[index])

    return reordered


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
        source = metadata.get("source") or metadata.get("path") or f"Source {index}"
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        chunk_index = metadata.get("chunk_index", index - 1)
        score = float(chunk.get("score", 0.0) or 0.0)

        context_parts.append(
            f"[Document {index} | Source: {source} | Type: {doc_type} | "
            f"Chunk: {chunk_index} | Score: {score:.4f}]\n"
            f"{str(chunk.get('content', '')).strip()}"
        )

    return "\n\n---\n\n".join(context_parts)


def _citation_label(chunk: dict, fallback_index: int) -> str:
    """Build a compact citation label from retrieval metadata."""
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    source = metadata.get("source") or metadata.get("path") or f"Nguồn {fallback_index}"
    year_match = re.search(r"(20\d{2}|19\d{2})", source)
    if year_match:
        return f"{source}, {year_match.group(1)}"
    return source


def _split_sentences(text: str) -> list[str]:
    """Sentence splitter nhỏ gọn cho fallback answer tiếng Việt."""
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def _offline_answer(query: str, chunks: list[dict]) -> str:
    """
    Generate a conservative cited answer without an external LLM.

    Fallback này giữ đúng tinh thần RAG: chỉ nói từ context có sẵn và gắn
    citation ngay sau từng ý. Nếu không có context thì từ chối xác minh.
    """
    if not chunks:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    answer_parts = []
    for index, chunk in enumerate(chunks[:TOP_K], start=1):
        content = str(chunk.get("content", "")).strip()
        if not content:
            continue

        sentence = _split_sentences(content)
        excerpt = sentence[0] if sentence else content
        if len(excerpt) > 360:
            excerpt = excerpt[:357].rstrip() + "..."

        citation = _citation_label(chunk, index)
        answer_parts.append(f"{excerpt} [{citation}]")

    if not answer_parts:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    return (
        f"Dựa trên các nguồn truy xuất được cho câu hỏi \"{query}\": "
        + " ".join(answer_parts)
    )


def _call_openai_llm(query: str, context: str) -> str | None:
    """Call OpenAI only when an API key is configured; otherwise use fallback."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        user_message = f"""Context:
{context}

---

Question: {query}"""

        response = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        return response.choices[0].message.content or None
    except Exception:
        return None


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    top_k = max(1, int(top_k or TOP_K))
    query = query.strip()
    if not query:
        return {
            "answer": "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources": [],
            "retrieval_source": "none",
        }

    chunks = retrieve(query, top_k=top_k)
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)

    answer = _call_openai_llm(query, context)
    if not answer:
        answer = _offline_answer(query, reordered)

    retrieval_source = "none"
    if chunks:
        retrieval_source = chunks[0].get("source", "hybrid")

    return {
        "answer": answer,
        "sources": chunks,
        "reordered_sources": reordered,
        "context": context,
        "retrieval_source": retrieval_source,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
