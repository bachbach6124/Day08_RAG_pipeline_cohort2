# Team Pipelines

Nhóm có 6 thành viên. Mỗi thành viên được gán một adapter pipeline riêng trong [pipeline_registry.py](pipeline_registry.py). Các adapter dùng chung corpus/index hiện có của repo nhưng khác retrieval strategy, focus và vai trò demo.

## Thành Viên Và Pipeline

| MSSV | Họ và tên | Pipeline key | Vai trò | Focus |
|---|---|---|---|---|
| 2A202600640 | Đào Xuân Bách | `bach_hybrid_legal` | Legal corpus + hybrid retrieval | Văn bản pháp luật, cai nghiện, danh mục chất ma túy |
| 2A202600775 | Đỗ Thiện Lĩnh | `linh_news_bm25` | News corpus + BM25 | Tin tức nghệ sĩ, tên riêng, vụ bắt/tạm giữ/truy tố |
| 2A202600657 | Lê Hoài Nam | `nam_hyde_rag` | HyDE + citation generation | Query mơ hồ hoặc thiếu keyword chính xác |
| 2A202600769 | Nguyễn Đức Kiên Trung | `trung_dense_semantic` | Dense semantic retrieval | Câu hỏi diễn đạt tự nhiên, semantic match |
| 2A202600673 | Nhan Khánh Đình | `dinh_tfidf_lexical` | TF-IDF lexical bonus | Demo lexical khác BM25, keyword explanation |
| 2A202600890 | Phan Quốc Anh | `anh_fallback_safety` | Fallback + QA safety | Fallback PageIndex-style, source display, insufficient evidence |

## Mô Tả Strategy

### `bach_hybrid_legal`

Pipeline ưu tiên dữ liệu pháp luật bằng cách thêm legal focus vào query, sau đó gọi Task 9 hybrid retrieval:

- semantic search
- BM25 lexical search
- RRF merge
- rerank
- generation có citation

### `linh_news_bm25`

Pipeline ưu tiên tin tức nghệ sĩ và tên riêng:

- BM25 search với query được mở rộng theo bối cảnh showbiz/nghệ sĩ
- semantic backup
- RRF merge
- rerank theo query gốc

### `nam_hyde_rag`

Pipeline bonus HyDE:

- tạo hypothetical document từ query
- nối query gốc với tài liệu giả định
- semantic + lexical retrieval
- RRF + rerank
- generation có citation

### `trung_dense_semantic`

Pipeline dense-only:

- semantic TF-IDF/cosine retrieval
- rerank local
- phù hợp câu hỏi tự nhiên không trùng keyword chính xác

### `dinh_tfidf_lexical`

Pipeline lexical khác BM25:

- TF-IDF vector hóa query/chunks
- cosine similarity
- dùng để demo bonus giải thích khác biệt TF-IDF và BM25

### `anh_fallback_safety`

Pipeline fallback/safety:

- chạy hybrid retrieval với threshold cao hơn
- nếu retrieval yếu, dùng PageIndex-style vectorless fallback
- luôn hiển thị source chunks để kiểm tra evidence

## Cách Kiểm Tra Đủ 6 Pipeline

```bash
python - <<'PY'
from group_project.pipeline_registry import list_pipelines

for p in list_pipelines():
    print(p.student_id, p.owner, p.key, p.role)
PY
```

Kết quả phải có 6 dòng.

## Cách Chạy Demo

```bash
source /home/namle/VINAI/.venv/bin/activate
streamlit run app.py --server.headless true --server.port 8501
```

Trong app:

- Sidebar chọn từng `Member pipeline`.
- Tab `Chat` để hỏi đáp có citation.
- Tab `Search` để xem search engine và source chunks.
- Tab `Team` để xem phân công và registry đủ 6 pipeline.
- Tab `Evaluation` để xem kết quả đánh giá.
- Tab `Methods` để demo HyDE, TF-IDF và UI/UX bonus.
