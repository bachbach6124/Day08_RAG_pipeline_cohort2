"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_DOC_IDS = [
    doc_id.strip()
    for doc_id in os.getenv("PAGEINDEX_DOC_IDS", "").split(",")
    if doc_id.strip()
]
PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
UPLOAD_MANIFEST = PROJECT_DIR / "data" / "landing" / "pageindex_uploads.json"

DEFAULT_TOP_K = 5
MAX_CHARS_PER_CHUNK = 1200
CHUNK_OVERLAP_CHARS = 200


def _tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text well enough for a no-embedding fallback."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def _iter_markdown_chunks() -> list[dict]:
    """
    Load markdown files and split them into compact structural chunks.

    This is the local vectorless fallback used when PageIndex credentials/doc ids
    are not configured. It keeps section boundaries where possible, which is the
    same spirit as PageIndex-style document structure retrieval.
    """
    chunks: list[dict] = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8").strip()
        if not text:
            continue

        sections = re.split(r"\n(?=#{1,6}\s+)", text)
        section_index = 0
        for section in sections:
            section = section.strip()
            if not section:
                continue

            heading_match = re.match(r"^(#{1,6})\s+(.+)$", section.splitlines()[0])
            heading = heading_match.group(2).strip() if heading_match else ""

            start = 0
            while start < len(section):
                end = min(start + MAX_CHARS_PER_CHUNK, len(section))
                content = section[start:end].strip()
                if content:
                    chunks.append(
                        {
                            "content": content,
                            "metadata": {
                                "filename": md_file.name,
                                "path": str(md_file.relative_to(PROJECT_DIR)),
                                "type": md_file.parent.name,
                                "heading": heading,
                                "section_index": section_index,
                                "chunk_start": start,
                            },
                        }
                    )
                if end >= len(section):
                    break
                start = max(end - CHUNK_OVERLAP_CHARS, start + 1)

            section_index += 1

    return chunks


def _local_vectorless_search(query: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Small BM25-style lexical retriever for offline PageIndex fallback."""
    chunks = _iter_markdown_chunks()
    if not chunks or not query.strip():
        return []

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    tokenized_docs = [_tokenize(item["content"]) for item in chunks]
    doc_freq: Counter[str] = Counter()
    for doc_tokens in tokenized_docs:
        doc_freq.update(set(doc_tokens))

    avg_doc_len = sum(len(tokens) for tokens in tokenized_docs) / max(len(tokenized_docs), 1)
    k1 = 1.5
    b = 0.75
    raw_results: list[dict] = []

    for item, doc_tokens in zip(chunks, tokenized_docs):
        if not doc_tokens:
            continue

        term_counts = Counter(doc_tokens)
        score = 0.0
        for token in query_tokens:
            tf = term_counts[token]
            if tf == 0:
                continue
            idf = math.log(1 + (len(chunks) - doc_freq[token] + 0.5) / (doc_freq[token] + 0.5))
            denom = tf + k1 * (1 - b + b * len(doc_tokens) / max(avg_doc_len, 1))
            score += idf * (tf * (k1 + 1) / denom)

        if score > 0:
            raw_results.append(
                {
                    "content": item["content"],
                    "score": score,
                    "metadata": item["metadata"],
                    "source": "pageindex",
                }
            )

    raw_results.sort(key=lambda result: result["score"], reverse=True)
    if not raw_results:
        return []

    max_score = raw_results[0]["score"] or 1.0
    for result in raw_results:
        result["score"] = round(float(result["score"] / max_score), 4)

    return raw_results[:top_k]


def _pageindex_client():
    if not PAGEINDEX_API_KEY:
        return None

    try:
        from pageindex import PageIndexClient

        return PageIndexClient(api_key=PAGEINDEX_API_KEY)
    except Exception:
        return None


def _load_manifest_doc_ids() -> list[str]:
    if PAGEINDEX_DOC_IDS:
        return PAGEINDEX_DOC_IDS
    if not UPLOAD_MANIFEST.exists():
        return []

    try:
        manifest = json.loads(UPLOAD_MANIFEST.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    return [
        item.get("doc_id", "")
        for item in manifest.get("documents", [])
        if item.get("doc_id")
    ]


def _extract_pageindex_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    if not isinstance(result, dict):
        return str(result)

    for key in ("text", "content", "markdown", "answer", "page_content"):
        if isinstance(result.get(key), str):
            return result[key]

    for key in ("blocks", "chunks", "nodes", "results", "retrieval"):
        value = result.get(key)
        if isinstance(value, list):
            parts = [_extract_pageindex_text(item) for item in value]
            return "\n\n".join(part for part in parts if part)

    return json.dumps(result, ensure_ascii=False)


def _query_pageindex_api(query: str, top_k: int) -> list[dict]:
    client = _pageindex_client()
    doc_ids = _load_manifest_doc_ids()
    if not client or not doc_ids:
        return []

    results: list[dict] = []
    for doc_id in doc_ids:
        submitted = client.submit_query(doc_id=doc_id, query=query, thinking=False)
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            continue

        retrieval: dict[str, Any] = {}
        for _ in range(12):
            retrieval = client.get_retrieval(retrieval_id)
            status = str(retrieval.get("status", "")).lower()
            if status in {"completed", "succeeded", "success", "done", "ready"}:
                break
            if retrieval.get("result") or retrieval.get("results"):
                break
            time.sleep(1)

        payload = retrieval.get("result") or retrieval.get("results") or retrieval
        if isinstance(payload, list):
            candidates = payload
        else:
            candidates = [payload]

        for candidate in candidates:
            content = _extract_pageindex_text(candidate).strip()
            if not content:
                continue

            score = 1.0
            metadata: dict[str, Any] = {"doc_id": doc_id}
            if isinstance(candidate, dict):
                score = float(candidate.get("score") or candidate.get("relevance_score") or 1.0)
                candidate_metadata = candidate.get("metadata")
                if isinstance(candidate_metadata, dict):
                    metadata.update(candidate_metadata)

            results.append(
                {
                    "content": content,
                    "score": score,
                    "metadata": metadata,
                    "source": "pageindex",
                }
            )

    results.sort(key=lambda result: result["score"], reverse=True)
    return results[:top_k]


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    client = _pageindex_client()
    if not client:
        local_docs = _iter_markdown_chunks()
        return {
            "mode": "local_fallback",
            "message": "PAGEINDEX_API_KEY is not set; using local vectorless markdown retrieval.",
            "chunks_available": len(local_docs),
        }

    uploaded: list[dict] = []
    pdf_files = sorted(LANDING_LEGAL_DIR.glob("*.pdf"))
    for pdf_file in pdf_files:
        response = client.submit_document(str(pdf_file))
        doc_id = response.get("doc_id") or response.get("id")
        uploaded.append(
            {
                "filename": pdf_file.name,
                "path": str(pdf_file.relative_to(PROJECT_DIR)),
                "doc_id": doc_id,
                "response": response,
            }
        )

    UPLOAD_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    UPLOAD_MANIFEST.write_text(
        json.dumps({"documents": uploaded}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"mode": "pageindex_api", "documents": uploaded, "manifest": str(UPLOAD_MANIFEST)}


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if top_k <= 0:
        return []

    try:
        api_results = _query_pageindex_api(query, top_k=top_k)
        if api_results:
            return api_results[:top_k]
    except Exception:
        # Keep the retrieval pipeline usable during local development and tests.
        pass

    return _local_vectorless_search(query, top_k=top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
