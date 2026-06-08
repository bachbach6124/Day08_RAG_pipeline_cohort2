# Bài Tập Nhóm — Search Engine / RAG Chatbot

## Mục Tiêu

Sau khi hoàn thành bài cá nhân, nhóm ngồi lại để xây dựng **2 sản phẩm**:

- RAG Chatbot/Search Engine demo bằng Streamlit.
- RAG Evaluation Pipeline có golden dataset, A/B comparison và báo cáo.

---

## Yêu cầu 1:  Sản phẩm nhóm RAG Chatbot

Xây dựng chatbot trả lời câu hỏi về pháp luật ma tuý và tin tức liên quan.

**Yêu cầu:**
- Giao diện chat (Streamlit / Gradio / Chainlit)
- Trả lời có citation (dựa trên Task 10)
- Hỗ trợ follow-up questions (conversation memory)
- Hiển thị source documents đã dùng

**Stack gợi ý:**
```
Chainlit/Streamlit → Retrieval (Task 9) → Generation (Task 10) → Display
```

---

## Yêu cầu 2: RAG Evaluation Pipeline

Sử dụng **1 trong 3 framework** sau để evaluate pipeline RAG của nhóm:

### Framework lựa chọn

| Framework | Cài đặt | Đặc điểm |
|-----------|---------|-----------|
| [DeepEval](https://github.com/confident-ai/deepeval) | `pip install deepeval` | Nhiều metric built-in, dễ integrate với pytest |
| [RAGAS](https://github.com/explodinggradients/ragas) | `pip install ragas` | Chuẩn industry cho RAG eval, 3 trục chính |
| [TruLens](https://github.com/truera/trulens) | `pip install trulens` | Dashboard UI, feedback functions mạnh |

### Yêu cầu Evaluation

1. **Tạo Golden Dataset** — tối thiểu 15 cặp Q&A (question, expected_answer, expected_context)
2. **Chạy evaluation** trên toàn bộ golden dataset với các metrics sau:
   - **Faithfulness** — câu trả lời có bám đúng context không?
   - **Answer Relevance** — câu trả lời có đúng câu hỏi không?
   - **Context Recall** — retriever có lấy đủ evidence không?
   - **Context Precision** — trong context lấy về, bao nhiêu % thực sự hữu ích?
3. **So sánh A/B** — chạy eval trên ít nhất 2 config khác nhau (ví dụ: có reranking vs không reranking, hoặc hybrid vs dense-only)
4. **Báo cáo** — bảng điểm + phân tích worst performers + đề xuất cải tiến

### Code mẫu — DeepEval

```python
from deepeval import evaluate
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRecallMetric,
    ContextualPrecisionMetric,
)
from deepeval.test_case import LLMTestCase

# Tạo test cases từ golden dataset
test_cases = []
for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    test_case = LLMTestCase(
        input=item["question"],
        actual_output=result["answer"],
        expected_output=item["expected_answer"],
        retrieval_context=[c["content"] for c in result["sources"]],
    )
    test_cases.append(test_case)

# Chạy evaluation
metrics = [
    FaithfulnessMetric(threshold=0.7),
    AnswerRelevancyMetric(threshold=0.7),
    ContextualRecallMetric(threshold=0.7),
    ContextualPrecisionMetric(threshold=0.7),
]

results = evaluate(test_cases, metrics)
```

### Code mẫu — RAGAS

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from datasets import Dataset

# Chuẩn bị data
eval_data = {
    "question": [],
    "answer": [],
    "contexts": [],
    "ground_truth": [],
}

for item in golden_dataset:
    result = rag_pipeline.generate_with_citation(item["question"])
    eval_data["question"].append(item["question"])
    eval_data["answer"].append(result["answer"])
    eval_data["contexts"].append([c["content"] for c in result["sources"]])
    eval_data["ground_truth"].append(item["expected_answer"])

dataset = Dataset.from_dict(eval_data)

# Chạy evaluation
result = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
)
print(result.to_pandas())
```

### Code mẫu — TruLens

```python
from trulens.apps.custom import TruCustomApp, instrument
from trulens.core import Feedback
from trulens.providers.openai import OpenAI as TruOpenAI

provider = TruOpenAI()

# Define feedback functions
f_faithfulness = Feedback(provider.groundedness_measure_with_cot_reasons).on_output()
f_relevance = Feedback(provider.relevance).on_input_output()
f_context_relevance = Feedback(provider.context_relevance).on_input()

# Wrap RAG pipeline
tru_rag = TruCustomApp(
    rag_pipeline,
    app_name="DrugLaw_RAG",
    feedbacks=[f_faithfulness, f_relevance, f_context_relevance],
)

# Run evaluation
with tru_rag as recording:
    for item in golden_dataset:
        rag_pipeline.generate_with_citation(item["question"])

# View dashboard
from trulens.dashboard import run_dashboard
run_dashboard()
```

### Deliverable Evaluation

- [ ] File `group_project/evaluation/golden_dataset.json` — 15+ cặp Q&A
- [ ] File `group_project/evaluation/eval_pipeline.py` — script chạy evaluation
- [ ] File `group_project/evaluation/results.md` — bảng điểm + phân tích
- [ ] So sánh A/B ít nhất 2 configs

---

## Yêu Cầu Chung

1. **Tích hợp pipeline** từ bài cá nhân của các thành viên
2. **Demo hoạt động được** trong buổi trình bày (chạy local hoặc deploy)
3. **Evaluation pipeline** chạy được và có báo cáo kết quả
4. **Code push lên repository** chung của nhóm
5. **README** mô tả kiến trúc và phân công (điền bên dưới)

---

## Kiến Trúc Hệ Thống

```
Member pipelines
        │
        ├── Đào Xuân Bách        → bach_hybrid_legal
        ├── Đỗ Thiện Lĩnh        → linh_news_bm25
        ├── Lê Hoài Nam          → nam_hyde_rag
        ├── Nguyễn Đức Kiên Trung→ trung_dense_semantic
        ├── Nhan Khánh Đình      → dinh_tfidf_lexical
        └── Phan Quốc Anh        → anh_fallback_safety

group_project/pipeline_registry.py
        │
        ▼
Streamlit app.py
        ├── Chat UI with conversation memory
        ├── Search UI with score/source/highlight display
        ├── Bonus HyDE retrieval mode
        ├── Bonus TF-IDF lexical demo mode
        └── Evaluation report tab

Evaluation
        │
        ▼
group_project/evaluation/eval_pipeline.py
        ├── Golden dataset 15 Q&A
        ├── Config A: hybrid + rerank
        ├── Config B: dense-only
        ├── Bonus Config: HyDE
        └── Team benchmark: 6 member pipelines
```

---

## Phân Công Công Việc

| Thành viên | MSSV | Nhiệm vụ | Trạng thái |
|-----------|------|----------|------------|
| Đào Xuân Bách | 2A202600640 | Pipeline `bach_hybrid_legal`: legal corpus, hybrid retrieval, reranking | Hoàn thành |
| Đỗ Thiện Lĩnh | 2A202600775 | Pipeline `linh_news_bm25`: news-focused BM25 + semantic backup | Hoàn thành |
| Lê Hoài Nam | 2A202600657 | Pipeline `nam_hyde_rag`: HyDE query expansion + citation generation | Hoàn thành |
| Nguyễn Đức Kiên Trung | 2A202600769 | Pipeline `trung_dense_semantic`: dense semantic retrieval + rerank | Hoàn thành |
| Nhan Khánh Đình | 2A202600673 | Pipeline `dinh_tfidf_lexical`: TF-IDF lexical bonus/explanation | Hoàn thành |
| Phan Quốc Anh | 2A202600890 | Pipeline `anh_fallback_safety`: PageIndex-style fallback, source QA, safety | Hoàn thành |
| Cả nhóm | - | Streamlit chatbot/search UI, memory, source display, highlight, evaluation report | Hoàn thành |

---

## Hướng Dẫn Chạy

```bash
# Cài đặt dependencies
pip install -r requirements.txt

# Chạy test bài cá nhân
pytest tests/test_individual.py -q

# Rebuild dữ liệu mẫu/index local nếu cần
python -m src.bootstrap_sample_data

# Chạy evaluation nhóm và xuất results.md
python group_project/evaluation/eval_pipeline.py

# Chạy chatbot/search demo
streamlit run app.py
```

## Bonus Implemented

| Bonus | Trạng thái | Nơi demo |
|---|---|---|
| Giải thích lexical search khác BM25 | Hoàn thành | App tab `Methods`, mode `TF-IDF`, hàm `tfidf_lexical_search` trong `src/task6_lexical_search.py` |
| HyDE query expansion | Hoàn thành | App sidebar mode `HyDE`, module `src/bonus_hyde.py`, evaluation bonus config |
| Conversation memory | Hoàn thành | App sidebar toggle `Conversation memory` |
| UI/UX source, score, highlight | Hoàn thành | App tab `Chat` và `Search` hiển thị route, score, type, chunk, highlight keyword |
| Deploy chatbot online | Deploy-ready | Chạy được local bằng Streamlit; deploy online cần tài khoản/Hugging Face/Render và credential ngoài repo |
 
## UI/UX Notes

App sử dụng theme sáng có tương phản rõ trong `.streamlit/config.toml`, source documents được đặt trong expander, tự xuống dòng với nội dung dài và highlight từ khóa query để tránh treo/chồng chữ khi tài liệu PDF dài.

## Tích Hợp Pipeline Thành Viên

Trong repo nhóm, "tích hợp pipeline từ bài cá nhân của các thành viên" nghĩa là mỗi thành viên expose pipeline của mình qua cùng một interface:

- `answer(query, top_k)` trả về `{"answer": str, "sources": list[dict]}`
- `search(query, top_k)` trả về list source chunks có `content`, `score`, `metadata`

File `group_project/pipeline_registry.py` là nơi đăng ký các adapter đó. Hiện app đã tích hợp đủ 6 pipeline:

- `bach_hybrid_legal`
- `linh_news_bm25`
- `nam_hyde_rag`
- `trung_dense_semantic`
- `dinh_tfidf_lexical`
- `anh_fallback_safety`

Hướng dẫn chi tiết cho thành viên nhóm: [PIPELINE_INTEGRATION_GUIDE.md](PIPELINE_INTEGRATION_GUIDE.md).

Chi tiết 6 pipeline thành viên: [TEAM_PIPELINES.md](TEAM_PIPELINES.md).

Hướng dẫn deploy online: [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Lưu ý: Hãy giữ lại repo này nếu như bạn học track 3 giai đoạn 2, chúng ta sẽ phát triển tiếp dự án lên knowledge graph để khắc phục các câu hỏi hóc búa khi có các câu hỏi khó.
