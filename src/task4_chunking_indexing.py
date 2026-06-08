"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client
"""

import hashlib
import json
import math
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
INDEX_DIR = PROJECT_DIR / "data" / "index"
VECTOR_INDEX_PATH = INDEX_DIR / "vector_store.jsonl"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Recursive chunking phù hợp vì dữ liệu trộn cả luật và bài báo Markdown:
# ưu tiên tách theo đoạn/dòng/câu, nhưng vẫn có fallback xuống ký tự.
CHUNK_SIZE = 500        # Đủ ngắn để retrieval chính xác và vừa context citation.
CHUNK_OVERLAP = 80      # Giữ liên kết giữa các điều/khoản nằm sát ranh giới chunk.
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Model nhỏ, nhanh, dễ chạy local trong lớp; pipeline mặc định dùng hash fallback
# offline để test/lab không bị kẹt khi chưa tải model.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Dùng JSONL local để không phụ thuộc server Weaviate trong bài tập/test local.
VECTOR_STORE = "local_jsonl"  # "weaviate" | "chromadb" | "faiss" | "local_jsonl"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        return documents

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue

        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"
        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path),
                "type": doc_type,
            },
        })

    return documents


def _split_oversized_text(text: str, max_size: int = CHUNK_SIZE) -> list[str]:
    """Hard-wrap any splitter output that remains longer than CHUNK_SIZE."""
    if len(text) <= max_size:
        return [text]

    parts = []
    start = 0
    while start < len(text):
        end = min(start + max_size, len(text))
        parts.append(text[start:end].strip())
        start = end
    return [part for part in parts if part]


def _recursive_split_text(text: str, separators: list[str] | None = None) -> list[str]:
    """Recursive character splitter tuned for Markdown/legal text."""
    separators = separators or ["\n\n", "\n", ". ", "; ", ", ", " ", ""]

    def split_piece(piece: str, seps: list[str]) -> list[str]:
        piece = piece.strip()
        if not piece:
            return []
        if len(piece) <= CHUNK_SIZE:
            return [piece]
        if not seps:
            return _split_oversized_text(piece)

        separator = seps[0]
        if separator and separator in piece:
            chunks = []
            current = ""
            for part in piece.split(separator):
                candidate = part if not current else f"{current}{separator}{part}"
                if len(candidate) <= CHUNK_SIZE:
                    current = candidate
                    continue

                chunks.extend(split_piece(current, seps[1:]))
                current = part

            chunks.extend(split_piece(current, seps[1:]))
            return chunks

        return split_piece(piece, seps[1:])

    base_chunks = split_piece(text, separators)
    if CHUNK_OVERLAP <= 0:
        return base_chunks

    overlapped = []
    previous_tail = ""
    for chunk in base_chunks:
        merged = f"{previous_tail} {chunk}".strip() if previous_tail else chunk
        if len(merged) > CHUNK_SIZE:
            merged = merged[-CHUNK_SIZE:].strip()
        overlapped.append(merged)
        previous_tail = chunk[-CHUNK_OVERLAP:]

    return overlapped


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    chunks = []
    for doc in documents:
        chunk_index = 0
        for split in _recursive_split_text(doc["content"]):
            for chunk_text in _split_oversized_text(split.strip()):
                chunks.append({
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_index": chunk_index,
                        "chunk_size": len(chunk_text),
                    },
                })
                chunk_index += 1

    return chunks


def _hash_embedding(text: str, dim: int = EMBEDDING_DIM) -> list[float]:
    """Deterministic fallback embedding for offline runs."""
    vector = [0.0] * dim
    tokens = text.lower().split()
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    if not chunks:
        return chunks

    use_sentence_transformers = os.getenv("TASK4_EMBEDDING_BACKEND") == "sentence-transformers"
    if use_sentence_transformers:
        try:
            from sentence_transformers import SentenceTransformer

            texts = [chunk["content"] for chunk in chunks]
            model = SentenceTransformer(EMBEDDING_MODEL)
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            for chunk, embedding in zip(chunks, embeddings):
                chunk["embedding"] = embedding.tolist()
                chunk["metadata"]["embedding_model"] = EMBEDDING_MODEL
            return chunks
        except Exception:
            pass

    for chunk in chunks:
        chunk["embedding"] = _hash_embedding(chunk["content"])
        chunk["metadata"]["embedding_model"] = "hash-fallback"

    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with VECTOR_INDEX_PATH.open("w", encoding="utf-8") as f:
        for chunk_id, chunk in enumerate(chunks):
            record = {
                "id": chunk_id,
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "embedding": chunk.get("embedding"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return VECTOR_INDEX_PATH


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
