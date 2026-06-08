"""
Registry for integrating six individual member RAG pipelines into the group app.

Each adapter exposes the same interface:
    answer(query, top_k) -> {"answer": str, "sources": list[dict]}
    search(query, top_k) -> list[dict]

The current repository has one shared local corpus/index, so the six adapters
use different retrieval strategies and query focus profiles over that shared
index. If a member later adds a completely separate pipeline module, only the
adapter functions below need to be replaced.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from src.bonus_hyde import hyde_search
from src.task10_generation import _offline_answer, reorder_for_llm
from src.task9_retrieval_pipeline import retrieve
from src.task5_semantic_search import semantic_search
from src.task6_lexical_search import lexical_search, tfidf_lexical_search
from src.task7_reranking import rerank, rerank_rrf
from src.task8_pageindex_vectorless import pageindex_search


@dataclass(frozen=True)
class TeamMember:
    student_id: str
    name: str
    pipeline_key: str
    role: str
    focus: str


@dataclass(frozen=True)
class PipelineAdapter:
    key: str
    owner: str
    student_id: str
    role: str
    focus: str
    description: str
    answer_fn: Callable[[str, int], dict]
    retrieve_fn: Callable[[str, int], list[dict]]

    def answer(self, query: str, top_k: int = 5) -> dict:
        return self.answer_fn(query, top_k)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.retrieve_fn(query, top_k)


TEAM_MEMBERS = [
    TeamMember(
        student_id="2A202600640",
        name="Đào Xuân Bách",
        pipeline_key="bach_hybrid_legal",
        role="Legal corpus + hybrid retrieval",
        focus="Văn bản pháp luật, điều khoản, cai nghiện, danh mục chất ma túy",
    ),
    TeamMember(
        student_id="2A202600775",
        name="Đỗ Thiện Lĩnh",
        pipeline_key="linh_news_bm25",
        role="News corpus + lexical retrieval",
        focus="Tin tức nghệ sĩ, tên riêng, sự kiện bị bắt/tạm giữ/truy tố",
    ),
    TeamMember(
        student_id="2A202600657",
        name="Lê Hoài Nam",
        pipeline_key="nam_hyde_rag",
        role="HyDE + generation with citation",
        focus="Câu hỏi mơ hồ hoặc thiếu từ khóa chính xác",
    ),
    TeamMember(
        student_id="2A202600769",
        name="Nguyễn Đức Kiên Trung",
        pipeline_key="trung_dense_semantic",
        role="Dense semantic retrieval",
        focus="Câu hỏi diễn đạt tự nhiên, semantic match",
    ),
    TeamMember(
        student_id="2A202600673",
        name="Nhan Khánh Đình",
        pipeline_key="dinh_tfidf_lexical",
        role="TF-IDF lexical bonus",
        focus="Demo lexical search khác BM25 và keyword explanation",
    ),
    TeamMember(
        student_id="2A202600890",
        name="Phan Quốc Anh",
        pipeline_key="anh_fallback_safety",
        role="Fallback + QA safety",
        focus="Fallback PageIndex-style, insufficient-evidence handling, source display",
    ),
]


def _with_focus(query: str, focus_text: str) -> str:
    return f"{query}\n\nNgữ cảnh ưu tiên: {focus_text}"


LEGAL_FOCUS = (
    "Luật Phòng chống ma túy, Nghị định 105, Nghị định 57, "
    "Bộ luật Hình sự, cai nghiện, chất cấm, danh mục chất ma túy"
)


def _is_legal_chunk(item: dict) -> bool:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    doc_type = str(metadata.get("type", "")).lower()
    path = str(metadata.get("path", "")).lower()
    source = str(metadata.get("source", "")).lower()
    return doc_type == "legal" or path.startswith("legal/") or "nghi-dinh" in source or "luat" in source


def _legal_results(results: list[dict], source: str) -> list[dict]:
    legal_only = []
    for item in results:
        if not isinstance(item, dict) or not item.get("content") or not _is_legal_chunk(item):
            continue

        normalized = item.copy()
        normalized["source"] = "bach-hybrid-legal"
        normalized["retrieval_channel"] = source
        normalized.setdefault("metadata", {})
        normalized["metadata"]["legal_only"] = True
        legal_only.append(normalized)

    return legal_only


def _answer_from_chunks(query: str, chunks: list[dict], route: str) -> dict:
    for chunk in chunks:
        chunk.setdefault("metadata", {})
        chunk.setdefault("source", route)
    ordered = reorder_for_llm(chunks)
    return {
        "answer": _offline_answer(query, ordered),
        "sources": chunks,
        "retrieval_source": route,
    }


def _bach_retrieve(query: str, top_k: int = 5) -> list[dict]:
    focused_query = _with_focus(query, LEGAL_FOCUS)
    pool_size = max(top_k * 4, 12)

    dense = _legal_results(
        semantic_search(focused_query, top_k=pool_size),
        "semantic",
    )
    sparse = _legal_results(
        lexical_search(focused_query, top_k=pool_size),
        "bm25",
    )

    merged = rerank_rrf([dense, sparse], top_k=pool_size)
    results = rerank(query, merged, top_k=top_k)
    for item in results:
        item["source"] = "bach-hybrid-legal"
        item.setdefault("metadata", {})
        item["metadata"]["legal_only"] = True
    return results


def _bach_answer(query: str, top_k: int = 5) -> dict:
    chunks = _bach_retrieve(query, top_k=top_k)
    return _answer_from_chunks(query, chunks, "bach-hybrid-legal")


def _linh_retrieve(query: str, top_k: int = 5) -> list[dict]:
    focused_query = _with_focus(query, "nghệ sĩ ca sĩ diễn viên người mẫu showbiz ma túy bị bắt tạm giữ truy tố")
    bm25 = lexical_search(focused_query, top_k=top_k * 2)
    dense = semantic_search(focused_query, top_k=top_k * 2)
    merged = rerank_rrf([bm25, dense], top_k=top_k * 2)
    results = rerank(query, merged, top_k=top_k)
    for item in results:
        item["source"] = "linh-news-bm25"
    return results


def _linh_answer(query: str, top_k: int = 5) -> dict:
    return _answer_from_chunks(query, _linh_retrieve(query, top_k), "linh-news-bm25")


def _nam_retrieve(query: str, top_k: int = 5) -> list[dict]:
    results = hyde_search(query, top_k=top_k)
    for item in results:
        item["source"] = "nam-hyde"
    return results


def _nam_answer(query: str, top_k: int = 5) -> dict:
    return _answer_from_chunks(query, _nam_retrieve(query, top_k), "nam-hyde")


def _trung_retrieve(query: str, top_k: int = 5) -> list[dict]:
    dense = semantic_search(query, top_k=top_k * 2)
    results = rerank(query, dense, top_k=top_k)
    for item in results:
        item["source"] = "trung-dense-semantic"
    return results


def _trung_answer(query: str, top_k: int = 5) -> dict:
    return _answer_from_chunks(query, _trung_retrieve(query, top_k), "trung-dense-semantic")


def _dinh_retrieve(query: str, top_k: int = 5) -> list[dict]:
    tfidf = tfidf_lexical_search(query, top_k=top_k)
    for item in tfidf:
        item["source"] = "dinh-tfidf"
    return tfidf


def _dinh_answer(query: str, top_k: int = 5) -> dict:
    return _answer_from_chunks(query, _dinh_retrieve(query, top_k), "dinh-tfidf")


def _anh_retrieve(query: str, top_k: int = 5) -> list[dict]:
    hybrid = retrieve(query, top_k=top_k, score_threshold=0.45, use_reranking=True)
    if not hybrid or max(item.get("score", 0.0) for item in hybrid) < 0.2:
        hybrid = pageindex_search(query, top_k=top_k)
    for item in hybrid:
        item["source"] = "anh-fallback"
    return hybrid


def _anh_answer(query: str, top_k: int = 5) -> dict:
    return _answer_from_chunks(query, _anh_retrieve(query, top_k), "anh-fallback")


PIPELINES = {
    "bach_hybrid_legal": PipelineAdapter(
        key="bach_hybrid_legal",
        owner="Đào Xuân Bách",
        student_id="2A202600640",
        role="Legal corpus + hybrid retrieval",
        focus="Ưu tiên văn bản pháp luật, cai nghiện, danh mục chất ma túy.",
        description="Hybrid legal RAG: semantic + BM25, RRF merge, rerank, citation generation.",
        answer_fn=_bach_answer,
        retrieve_fn=_bach_retrieve,
    ),
    "linh_news_bm25": PipelineAdapter(
        key="linh_news_bm25",
        owner="Đỗ Thiện Lĩnh",
        student_id="2A202600775",
        role="News corpus + BM25",
        focus="Ưu tiên tin tức nghệ sĩ và tên riêng trong báo chí.",
        description="News-focused retrieval: BM25 first, semantic backup, rerank for named entities.",
        answer_fn=_linh_answer,
        retrieve_fn=_linh_retrieve,
    ),
    "nam_hyde_rag": PipelineAdapter(
        key="nam_hyde_rag",
        owner="Lê Hoài Nam",
        student_id="2A202600657",
        role="HyDE + citation generation",
        focus="Query expansion cho câu hỏi mơ hồ/thiếu keyword.",
        description="HyDE RAG: hypothetical document expansion + hybrid merge + citation answer.",
        answer_fn=_nam_answer,
        retrieve_fn=_nam_retrieve,
    ),
    "trung_dense_semantic": PipelineAdapter(
        key="trung_dense_semantic",
        owner="Nguyễn Đức Kiên Trung",
        student_id="2A202600769",
        role="Dense semantic retrieval",
        focus="Semantic matching cho câu hỏi tự nhiên.",
        description="Dense semantic pipeline: TF-IDF/cosine semantic search + local reranking.",
        answer_fn=_trung_answer,
        retrieve_fn=_trung_retrieve,
    ),
    "dinh_tfidf_lexical": PipelineAdapter(
        key="dinh_tfidf_lexical",
        owner="Nhan Khánh Đình",
        student_id="2A202600673",
        role="TF-IDF lexical bonus",
        focus="Lexical explanation và keyword-based retrieval.",
        description="TF-IDF lexical pipeline: cosine over TF-IDF vectors, useful for bonus demo.",
        answer_fn=_dinh_answer,
        retrieve_fn=_dinh_retrieve,
    ),
    "anh_fallback_safety": PipelineAdapter(
        key="anh_fallback_safety",
        owner="Phan Quốc Anh",
        student_id="2A202600890",
        role="Fallback + safety",
        focus="Fallback retrieval, source QA, insufficient-evidence behavior.",
        description="Fallback-safe RAG: hybrid retrieval with PageIndex-style fallback and strict source display.",
        answer_fn=_anh_answer,
        retrieve_fn=_anh_retrieve,
    ),
}


def list_members() -> list[TeamMember]:
    return TEAM_MEMBERS


def list_pipelines() -> list[PipelineAdapter]:
    return list(PIPELINES.values())


def get_pipeline(key: str) -> PipelineAdapter:
    if key not in PIPELINES:
        raise KeyError(f"Unknown pipeline: {key}")
    return PIPELINES[key]
