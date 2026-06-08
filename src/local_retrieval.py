"""
Shared local retrieval helpers for the Day 8 lab.

The lab recommends online vector stores and external APIs, but the automated
tests need a deterministic offline implementation. These helpers keep the
individual task modules small while still using the same data surface:
markdown documents in data/standardized/.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path


PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = PROJECT_DIR / "data" / "standardized"
VECTORSTORE_DIR = PROJECT_DIR / "data" / "vectorstore"
CHUNKS_PATH = VECTORSTORE_DIR / "chunks.json"

TOKEN_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)

SYNONYMS = {
    "ma": ["ma tuy", "chat cam", "heroin", "ketamin", "methamphetamine"],
    "tuy": ["ma tuy", "chat cam", "heroin", "ketamin", "methamphetamine"],
    "tuý": ["ma túy", "chất cấm", "heroin", "ketamin", "methamphetamine"],
    "hinh": ["phat", "tu", "bo luat hinh su"],
    "phat": ["hinh phat", "muc phat", "tu"],
    "tàng": ["tang tru", "cat giau"],
    "trữ": ["tang tru", "cat giau"],
    "cai": ["cai nghien", "nghien"],
    "nghiện": ["cai nghiện", "ma túy"],
}


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text in a dependency-free way."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def expanded_tokens(text: str) -> list[str]:
    tokens = tokenize(text)
    expanded = list(tokens)
    for token in tokens:
        for phrase in SYNONYMS.get(token, []):
            expanded.extend(tokenize(phrase))
    return expanded


def read_markdown_documents() -> list[dict]:
    documents: list[dict] = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue
        relative = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative.parts[0] if len(relative.parts) > 1 else "unknown"
        documents.append(
            {
                "content": content,
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative),
                    "type": doc_type,
                },
            }
        )
    return documents


def recursive_chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple recursive-ish splitter by paragraph, line, sentence, then chars."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    pieces = re.split(r"(\n\n+|\n|(?<=[.!?。])\s+)", text)
    units: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if len(piece) > chunk_size:
            if current.strip():
                units.append(current.strip())
                current = ""
            start = 0
            while start < len(piece):
                units.append(piece[start : start + chunk_size].strip())
                start += max(1, chunk_size - overlap)
            continue
        if len(current) + len(piece) <= chunk_size:
            current += piece
        else:
            if current.strip():
                units.append(current.strip())
            current = piece
    if current.strip():
        units.append(current.strip())

    chunks: list[str] = []
    for unit in units:
        if not chunks or overlap <= 0:
            chunks.append(unit[:chunk_size])
            continue
        prefix = chunks[-1][-overlap:]
        merged = f"{prefix} {unit}".strip()
        chunks.append(merged[:chunk_size])
    return [c for c in chunks if c]


def build_chunks(chunk_size: int = 500, overlap: int = 50) -> list[dict]:
    chunks: list[dict] = []
    for doc in read_markdown_documents():
        for index, text in enumerate(recursive_chunk_text(doc["content"], chunk_size, overlap)):
            chunks.append(
                {
                    "content": text,
                    "metadata": {**doc["metadata"], "chunk_index": index},
                }
            )
    return chunks


def save_chunks(chunks: list[dict]) -> None:
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


@lru_cache(maxsize=1)
def get_chunks() -> tuple[dict, ...]:
    if CHUNKS_PATH.exists():
        data = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
        return tuple(data)
    chunks = build_chunks()
    if chunks:
        save_chunks(chunks)
    return tuple(chunks)


def cosine_from_counters(a: Counter, b: Counter, idf: dict[str, float] | None = None) -> float:
    if not a or not b:
        return 0.0
    idf = idf or {}
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    terms = set(a) | set(b)
    for term in terms:
        weight = idf.get(term, 1.0)
        av = a.get(term, 0) * weight
        bv = b.get(term, 0) * weight
        dot += av * bv
        norm_a += av * av
        norm_b += bv * bv
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / math.sqrt(norm_a * norm_b)


def corpus_idf(tokenized_docs: list[list[str]]) -> dict[str, float]:
    doc_count = len(tokenized_docs)
    df: Counter = Counter()
    for tokens in tokenized_docs:
        df.update(set(tokens))
    return {term: math.log((doc_count + 1) / (freq + 1)) + 1 for term, freq in df.items()}
