from __future__ import annotations

import sys
from html import escape
from pathlib import Path

import streamlit as st


def _is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except Exception:
        return False


if __name__ == "__main__" and not _is_streamlit_runtime():
    print(
        "This is a Streamlit app. Run it with:\n\n"
        "    streamlit run app.py\n"
    )
    sys.exit(0)


PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from group_project.pipeline_registry import get_pipeline, list_members, list_pipelines
from src.bonus_hyde import build_hyde_query, hyde_search
from src.task10_generation import _offline_answer, reorder_for_llm
from src.task6_lexical_search import tfidf_lexical_search


RESULTS_PATH = PROJECT_DIR / "group_project" / "evaluation" / "results.md"


st.set_page_config(
    page_title="DrugLaw RAG",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --page: #f0f5f8;
        --panel: #ffffff;
        --panel-soft: #f7fafc;
        --line: #bfccd8;
        --ink: #111827;
        --muted: #4b5563;
        --accent: #007c89;
        --accent-2: #6d28d9;
        --warn: #a16207;
        --good: #047857;
        --mark: #fff2a8;
    }
    .stApp {
        background: var(--page);
        color: var(--ink);
    }
    .block-container {
        padding-top: .9rem;
        padding-bottom: 4.5rem;
        max-width: 1380px;
    }
    h1, h2, h3, p, li, label, span, div {
        letter-spacing: 0;
        overflow-wrap: anywhere;
    }
    .rag-title {
        font-size: clamp(1.25rem, 2.4vw, 1.65rem);
        font-weight: 720;
        margin: 0 0 .15rem 0;
        color: var(--ink);
    }
    .rag-subtitle {
        color: var(--muted);
        font-size: .9rem;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] {
        background: #102033;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
        overflow-wrap: anywhere;
    }
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] textarea,
    section[data-testid="stSidebar"] [role="combobox"],
    section[data-testid="stSidebar"] [data-baseweb="select"] * {
        color: #111827 !important;
        background: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stToggle label {
        color: #f8fafc !important;
    }
    div[data-testid="stChatMessage"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: .35rem .45rem;
        overflow: visible;
    }
    div[data-testid="stChatMessage"] p {
        color: var(--ink);
        line-height: 1.55;
        word-break: break-word;
    }
    .stTextInput input {
        background: #ffffff;
        color: var(--ink);
        border: 1px solid var(--line);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: .35rem;
    }
    .stTabs [data-baseweb="tab"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        min-height: 38px;
        padding: .35rem .8rem;
        color: var(--ink);
    }
    .stTabs [aria-selected="true"] {
        border-color: var(--accent);
        color: var(--accent);
        font-weight: 650;
    }
    .source-row {
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: .75rem .85rem;
        margin-bottom: .55rem;
        background: var(--panel);
        overflow: hidden;
    }
    .source-meta {
        color: var(--muted);
        font-size: .82rem;
        margin-bottom: .35rem;
        word-break: break-word;
    }
    .source-content {
        font-size: .91rem;
        line-height: 1.45;
        color: var(--ink);
        word-break: break-word;
        white-space: normal;
    }
    .source-content mark {
        background: var(--mark);
        color: var(--ink);
        padding: 0 .12rem;
        border-radius: 3px;
    }
    .source-pill {
        display: inline-block;
        border-radius: 999px;
        padding: .12rem .42rem;
        margin-right: .28rem;
        font-size: .76rem;
        font-weight: 650;
        color: #ffffff;
        background: var(--accent);
    }
    .source-pill.alt {
        background: var(--accent-2);
    }
    .source-pill.warn {
        background: var(--warn);
    }
    .source-pill.good {
        background: var(--good);
    }
    .team-card {
        border: 1px solid var(--line);
        border-radius: 7px;
        background: var(--panel);
        padding: .85rem;
        min-height: 9.5rem;
        margin-bottom: .75rem;
        overflow-wrap: anywhere;
    }
    .team-name {
        font-weight: 720;
        color: var(--ink);
        margin-bottom: .1rem;
    }
    .team-id {
        color: var(--muted);
        font-size: .84rem;
        margin-bottom: .45rem;
    }
    .team-role {
        color: var(--accent);
        font-weight: 650;
        font-size: .9rem;
        margin-bottom: .3rem;
    }
    .status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
        gap: .65rem;
        margin-bottom: 1rem;
    }
    .status-box {
        border: 1px solid var(--line);
        border-radius: 7px;
        background: var(--panel);
        padding: .72rem .8rem;
        color: var(--ink);
    }
    .status-k {
        color: var(--muted);
        font-size: .78rem;
    }
    .status-v {
        font-size: 1.05rem;
        font-weight: 720;
        margin-top: .12rem;
    }
    .metric-box {
        border: 1px solid var(--line);
        border-radius: 6px;
        background: var(--panel);
        padding: .8rem;
    }
    .method-note {
        border: 1px solid var(--line);
        border-left: 4px solid var(--accent);
        border-radius: 6px;
        background: var(--panel-soft);
        padding: .7rem .8rem;
        margin: .5rem 0 .8rem 0;
        color: var(--ink);
        font-size: .9rem;
        line-height: 1.45;
    }
    button, .stButton button {
        white-space: normal;
        min-height: 2.35rem;
        border-radius: 6px;
        overflow-wrap: anywhere;
    }
    div[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 7px;
        overflow: hidden;
    }
    div[data-testid="stExpander"] p {
        word-break: break-word;
        overflow-wrap: anywhere;
    }
    @media (max-width: 900px) {
        .block-container {
            padding-left: .7rem;
            padding-right: .7rem;
        }
        .source-meta, .source-content {
            font-size: .84rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("last_sources", [])
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("search_results", [])


def _source_label(source: dict, idx: int) -> str:
    metadata = source.get("metadata", {})
    name = metadata.get("source") or metadata.get("path") or f"Source {idx}"
    retrieval = source.get("source", "hybrid")
    score = source.get("score", 0.0)
    return f"{idx}. {name} | {retrieval} | score={score:.3f}"


def _highlight_terms(text: str, query: str) -> str:
    escaped = escape(text)
    terms = sorted({term for term in query.split() if len(term) >= 4}, key=len, reverse=True)
    for term in terms[:8]:
        safe_term = escape(term)
        escaped = escaped.replace(safe_term, f"<mark>{safe_term}</mark>")
        escaped = escaped.replace(safe_term.capitalize(), f"<mark>{safe_term.capitalize()}</mark>")
    return escaped


def _render_sources(sources: list[dict], query: str = "") -> None:
    if not sources:
        st.info("Chưa có source documents.")
        return

    for idx, source in enumerate(sources, 1):
        metadata = source.get("metadata", {})
        source_type = metadata.get("type", "unknown")
        chunk_index = metadata.get("chunk_index", "n/a")
        label = _source_label(source, idx)
        retrieval = source.get("source", "hybrid")
        snippet_text = source.get("content", "").replace("\n", " ").strip()
        visible_text = snippet_text[:1300] + ("..." if len(snippet_text) > 1300 else "")
        snippet = _highlight_terms(visible_text, query) if query else escape(visible_text)
        label = escape(label)
        source_type = escape(str(source_type))
        chunk_index = escape(str(chunk_index))
        pill_class = (
            "alt" if "hyde" in retrieval else
            "warn" if "tfidf" in retrieval else
            "good" if "dense" in retrieval else
            ""
        )
        with st.expander(f"{idx}. {metadata.get('source', 'Source')} | score={source.get('score', 0.0):.3f}", expanded=idx <= 3):
            st.markdown(
                f"""
                <div class="source-row">
                    <div class="source-meta">
                        <span class="source-pill {pill_class}">{escape(str(retrieval))}</span>
                        {label} | type={source_type} | chunk={chunk_index}
                    </div>
                    <div class="source-content">{snippet}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_status_cards(pipeline_count: int) -> None:
    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-box"><div class="status-k">Team pipelines</div><div class="status-v">{pipeline_count}</div></div>
            <div class="status-box"><div class="status-k">Products</div><div class="status-v">Chatbot + Search + Eval</div></div>
            <div class="status-box"><div class="status-k">Bonus</div><div class="status-v">HyDE, TF-IDF, Memory, UI</div></div>
            <div class="status-box"><div class="status-k">Demo</div><div class="status-v">Streamlit local</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_team() -> None:
    members = list_members()
    rows = [members[i:i + 3] for i in range(0, len(members), 3)]
    for row in rows:
        cols = st.columns(len(row))
        for col, member in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="team-card">
                        <div class="team-name">{escape(member.name)}</div>
                        <div class="team-id">{escape(member.student_id)} · {escape(member.pipeline_key)}</div>
                        <div class="team-role">{escape(member.role)}</div>
                        <div>{escape(member.focus)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def _render_pipeline_table(pipelines) -> None:
    rows = [
        {
            "Pipeline": p.key,
            "Owner": p.owner,
            "MSSV": p.student_id,
            "Role": p.role,
            "Focus": p.focus,
        }
        for p in pipelines
    ]
    st.dataframe(rows, width="stretch", hide_index=True)


def _conversation_query(user_input: str) -> str:
    previous_user_messages = [
        item["content"]
        for item in st.session_state.messages
        if item.get("role") == "user"
    ][-2:]
    if not previous_user_messages:
        return user_input
    history = " | ".join(previous_user_messages)
    return f"Ngữ cảnh hội thoại gần nhất: {history}. Câu hỏi hiện tại: {user_input}"


def _run_answer(pipeline, query: str, top_k: int, retrieval_mode: str) -> dict:
    if retrieval_mode == "HyDE":
        chunks = hyde_search(query, top_k=top_k)
        ordered = reorder_for_llm(chunks)
        return {
            "answer": _offline_answer(query, ordered),
            "sources": chunks,
            "retrieval_source": "hyde",
            "hyde_query": build_hyde_query(query),
        }
    if retrieval_mode == "TF-IDF":
        chunks = tfidf_lexical_search(query, top_k=top_k)
        ordered = reorder_for_llm(chunks)
        return {
            "answer": _offline_answer(query, ordered),
            "sources": chunks,
            "retrieval_source": "tfidf",
        }
    return pipeline.answer(query, top_k=top_k)


def _run_search(pipeline, query: str, top_k: int, retrieval_mode: str) -> list[dict]:
    if retrieval_mode == "HyDE":
        return hyde_search(query, top_k=top_k)
    if retrieval_mode == "TF-IDF":
        return tfidf_lexical_search(query, top_k=top_k)
    return pipeline.search(query, top_k=top_k)


def main() -> None:
    _init_state()

    pipelines = list_pipelines()
    pipeline_options = {f"{p.owner} ({p.key})": p.key for p in pipelines}

    with st.sidebar:
        st.markdown("### Pipeline")
        selected_label = st.selectbox("Member pipeline", list(pipeline_options.keys()))
        retrieval_mode = st.radio(
            "Retrieval mode",
            ["Hybrid", "HyDE", "TF-IDF"],
            help="Hybrid is the main RAG pipeline. HyDE and TF-IDF are bonus/demo modes.",
        )
        top_k = st.slider("Top K", min_value=3, max_value=10, value=5, step=1)
        use_memory = st.toggle("Conversation memory", value=True)
        pipeline = get_pipeline(pipeline_options[selected_label])
        st.caption(pipeline.description)
        if st.button("Clear chat", width="stretch"):
            st.session_state.messages = []
            st.session_state.last_sources = []
            st.rerun()

    st.markdown('<div class="rag-title">DrugLaw RAG Chatbot & Search Engine</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rag-subtitle">RAG chatbot và search engine cho pháp luật ma túy, chất cấm và tin tức nghệ sĩ liên quan.</div>',
        unsafe_allow_html=True,
    )
    _render_status_cards(len(pipelines))

    chat_tab, search_tab, team_tab, eval_tab, method_tab = st.tabs(["Chat", "Search", "Team", "Evaluation", "Methods"])

    with chat_tab:
        chat_col, source_col = st.columns([0.58, 0.42], gap="large")

        with chat_col:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            prompt = st.chat_input("Hỏi về luật phòng chống ma túy hoặc tin tức liên quan")
            if prompt:
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.session_state.last_query = prompt
                with st.chat_message("user"):
                    st.markdown(prompt)

                query = _conversation_query(prompt) if use_memory else prompt
                with st.chat_message("assistant"):
                    with st.spinner("Retrieving and generating answer..."):
                        result = _run_answer(pipeline, query, top_k=top_k, retrieval_mode=retrieval_mode)
                    answer = result.get("answer", "")
                    st.markdown(answer)
                    if result.get("hyde_query"):
                        with st.expander("HyDE expanded query"):
                            st.code(result["hyde_query"])

                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.last_sources = result.get("sources", [])

        with source_col:
            st.markdown("### Sources")
            _render_sources(st.session_state.last_sources, st.session_state.last_query)

    with search_tab:
        query = st.text_input("Search query", value="Luật Phòng chống ma túy quy định gì về cai nghiện?")
        if st.button("Run search", type="primary"):
            with st.spinner("Searching..."):
                results = _run_search(pipeline, query, top_k=top_k, retrieval_mode=retrieval_mode)
            st.session_state.search_results = results

        _render_sources(st.session_state.get("search_results", []), query)

    with team_tab:
        st.markdown("### Team Pipelines")
        _render_team()
        st.markdown("### Registry")
        _render_pipeline_table(pipelines)

    with eval_tab:
        st.markdown("### Evaluation Report")
        if RESULTS_PATH.exists():
            st.markdown(RESULTS_PATH.read_text(encoding="utf-8"))
        else:
            st.warning("Evaluation report not found. Run `python group_project/evaluation/eval_pipeline.py`.")

    with method_tab:
        st.markdown("### Bonus Methods")
        st.markdown(
            """
            <div class="method-note">
            <b>HyDE:</b> app tạo một đoạn tài liệu giả định từ câu hỏi, nối với query gốc,
            rồi retrieval trên query mở rộng. Cách này giúp tăng recall khi câu hỏi ít từ khóa
            hoặc dùng cách diễn đạt khác tài liệu.
            </div>
            <div class="method-note">
            <b>TF-IDF lexical search:</b> score dựa trên tần suất từ trong chunk và độ hiếm
            của từ trên toàn corpus, sau đó xếp hạng bằng cosine similarity. Khác BM25,
            TF-IDF không có term saturation <code>k1</code> và length-normalization <code>b</code>.
            </div>
            <div class="method-note">
            <b>UI/UX:</b> source cards hiển thị retrieval route, score, type, chunk index và
            highlight từ khóa query để người demo đối chiếu evidence nhanh hơn.
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
