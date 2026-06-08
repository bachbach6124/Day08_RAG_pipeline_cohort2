# Deployment Guide

App đã sẵn sàng để chạy local bằng Streamlit. Deploy online cần tài khoản nền tảng và quyền push repo.

## Local Demo

```bash
source /home/namle/VINAI/.venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.headless true --server.port 8501
```

Mở:

```text
http://localhost:8501
```

## Hugging Face Spaces

1. Tạo Space mới:
   - SDK: `Streamlit`
   - Python: 3.10 hoặc 3.11 nếu chọn được
2. Push các file chính:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
   - `src/`
   - `group_project/`
   - `data/standardized/`
   - `data/vectorstore/`
3. Nếu dùng OpenAI/PageIndex thật, thêm secrets:
   - `OPENAI_API_KEY`
   - `OPENAI_MODEL`
   - `PAGEINDEX_API_KEY`
4. Space sẽ tự chạy Streamlit app.

## Render

Start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

Build command:

```bash
pip install -r requirements.txt
```

## Deploy Checklist

- [ ] `pytest tests/test_individual.py -q` pass local.
- [ ] `python group_project/evaluation/eval_pipeline.py` chạy và cập nhật `results.md`.
- [ ] `streamlit run app.py` chạy local.
- [ ] Repo có `data/standardized/` và `data/vectorstore/chunks.json`.
- [ ] Secrets đã được set nếu dùng API thật.
