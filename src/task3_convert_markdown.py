"""
Task 3 — Convert toàn bộ file trong data/landing/ thành Markdown.

Sử dụng MarkItDown của Microsoft:
    https://github.com/microsoft/markitdown

Cài đặt:
    pip install markitdown

Hướng dẫn:
    1. Scan toàn bộ file trong data/landing/ (PDF, DOCX, JSON)
    2. Convert sang Markdown
    3. Lưu vào data/standardized/ giữ nguyên cấu trúc thư mục
"""

import json
import subprocess
from pathlib import Path

from markitdown import MarkItDown

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"


def convert_with_textutil(filepath: Path) -> str:
    """Fallback converter for legacy .doc files on macOS."""
    completed = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", str(filepath)],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def convert_legal_docs():
    """Convert PDF/DOCX files trong data/landing/legal/ sang markdown."""
    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)

    md = MarkItDown()

    for filepath in sorted(legal_dir.iterdir()):
        if filepath.suffix.lower() in (".pdf", ".docx", ".doc"):
            print(f"Converting: {filepath.name}")
            try:
                text_content = md.convert(str(filepath)).text_content
            except Exception as exc:
                if filepath.suffix.lower() != ".doc":
                    raise
                print(f"  MarkItDown skipped legacy .doc ({exc}); using textutil fallback")
                text_content = convert_with_textutil(filepath)

            output_path = output_dir / f"{filepath.stem}.md"
            output_path.write_text(text_content.strip() + "\n", encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_news_articles():
    """Convert JSON crawled articles trong data/landing/news/ sang markdown."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)

    for filepath in sorted(news_dir.iterdir()):
        if filepath.suffix.lower() == ".json":
            print(f"Converting: {filepath.name}")
            data = json.loads(filepath.read_text(encoding="utf-8"))
            output_path = output_dir / f"{filepath.stem}.md"

            header = f"# {data.get('title') or 'Unknown'}\n\n"
            header += f"**Source:** {data.get('source') or 'N/A'}\n"
            header += f"**URL:** {data.get('url') or data.get('original_url') or 'N/A'}\n"
            header += f"**Published:** {data.get('published_at') or 'N/A'}\n"
            header += f"**Crawled:** {data.get('crawled_at') or data.get('date_crawled') or 'N/A'}\n\n---\n\n"

            body = (
                data.get("content_markdown")
                or data.get("markdown")
                or data.get("content")
                or ""
            )
            output_path.write_text(header + body.strip() + "\n", encoding="utf-8")
            print(f"  ✓ Saved: {output_path}")


def convert_all():
    """Convert toàn bộ files."""
    print("=" * 50)
    print("Task 3: Convert to Markdown (MarkItDown)")
    print("=" * 50)

    print("\n--- Legal Documents ---")
    convert_legal_docs()

    print("\n--- News Articles ---")
    convert_news_articles()

    print("\n✓ Done! Output tại:", OUTPUT_DIR)


if __name__ == "__main__":
    convert_all()
