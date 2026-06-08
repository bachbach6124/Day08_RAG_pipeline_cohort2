"""
Create offline sample data for the Day 8 individual lab.

The files are intentionally small but structured like the required deliverables:
3 legal DOCX files in data/landing/legal, 5 news JSON files in data/landing/news,
and converted markdown in data/standardized.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from src.task3_convert_markdown import convert_all
from src.task4_chunking_indexing import chunk_documents, index_to_vectorstore, load_documents


PROJECT_DIR = Path(__file__).parent.parent
LEGAL_DIR = PROJECT_DIR / "data" / "landing" / "legal"
NEWS_DIR = PROJECT_DIR / "data" / "landing" / "news"


LEGAL_DOCS = {
    "luat-phong-chong-ma-tuy-2021.docx": [
        "Luật Phòng, chống ma túy 2021 - Luật số 73/2021/QH14",
        "Nguồn tham khảo: Cổng thông tin pháp luật và Công báo Chính phủ.",
        "Luật quy định trách nhiệm của cá nhân, gia đình, cơ quan, tổ chức trong phòng ngừa, đấu tranh, cai nghiện và quản lý sau cai nghiện ma túy.",
        "Các nội dung trọng tâm gồm chính sách của Nhà nước về phòng, chống ma túy, quản lý người sử dụng trái phép chất ma túy, cai nghiện tự nguyện, cai nghiện bắt buộc và hợp tác quốc tế.",
        "Văn bản nhấn mạnh nguyên tắc kết hợp phòng ngừa với đấu tranh, lấy phòng ngừa là chính, đồng thời bảo đảm quyền, lợi ích hợp pháp của người tham gia phòng, chống ma túy.",
        "Từ khóa phục vụ truy vấn: ma túy, chất cấm, cai nghiện, người sử dụng trái phép chất ma túy, quản lý sau cai nghiện, trách nhiệm gia đình.",
    ],
    "nghi-dinh-105-2021.docx": [
        "Nghị định 105/2021/NĐ-CP - Hướng dẫn thi hành Luật Phòng, chống ma túy",
        "Nguồn tham khảo: https://congbao.chinhphu.vn/so-do-van-ban-so-105-2021-nd-cp-34944",
        "Nghị định quy định chi tiết một số điều của Luật Phòng, chống ma túy, trong đó có quản lý người sử dụng trái phép chất ma túy và hồ sơ, thủ tục liên quan đến cai nghiện.",
        "Nội dung đáng chú ý gồm quy trình xác định tình trạng nghiện, lập hồ sơ quản lý, tổ chức cai nghiện tự nguyện tại gia đình, cộng đồng hoặc cơ sở cai nghiện.",
        "Văn bản cũng hướng dẫn trách nhiệm phối hợp của công an, y tế, lao động - thương binh và xã hội, ủy ban nhân dân cấp xã và gia đình người nghiện.",
        "Từ khóa phục vụ truy vấn: Nghị định 105, cai nghiện bắt buộc, cai nghiện tự nguyện, hồ sơ quản lý, xác định tình trạng nghiện.",
    ],
    "nghi-dinh-57-2022-danh-muc-chat-ma-tuy.docx": [
        "Nghị định 57/2022/NĐ-CP - Danh mục chất ma túy và tiền chất",
        "Nguồn tham khảo: https://congbao.chinhphu.vn/tai-ve-van-ban-so-57-2022-nd-cp-37734-41623?format=pdf",
        "Nghị định ban hành các danh mục chất ma túy và tiền chất, làm căn cứ cho quản lý nhà nước, giám định, điều tra và xử lý vi phạm.",
        "Danh mục bao gồm nhóm chất ma túy tuyệt đối cấm sử dụng trong y học và đời sống xã hội, nhóm chất được sử dụng hạn chế theo quy định đặc biệt, cùng các tiền chất cần kiểm soát.",
        "Việc xác định một chất có thuộc danh mục hay không là chứng cứ quan trọng trong các vụ án tàng trữ, vận chuyển, mua bán hoặc tổ chức sử dụng trái phép chất ma túy.",
        "Từ khóa phục vụ truy vấn: danh mục chất ma túy, tiền chất, ketamine, heroin, methamphetamine, giám định ma túy.",
    ],
}


NEWS_ARTICLES = [
    {
        "filename": "article_01_huu_tin_vnexpress.json",
        "url": "https://vnexpress.net/dien-vien-hai-bi-tam-giu-vi-lien-quan-ma-tuy-4475240.html",
        "title": "Diễn viên hài bị tạm giữ vì liên quan ma túy",
        "content_markdown": "Bài VnExpress đưa tin diễn viên Trần Hữu Tín, nghệ danh Hữu Tín, bị tạm giữ để làm rõ nghi vấn liên quan đến ma túy. Nội dung bài báo nêu lực lượng chức năng kiểm tra và phát hiện nhóm người sử dụng ma túy tại một căn hộ, đồng thời thu giữ tang vật. Vụ việc là một ví dụ tin tức về nghệ sĩ Việt Nam liên quan hành vi sử dụng hoặc tàng trữ trái phép chất ma túy. Từ khóa: Hữu Tín, diễn viên hài, ma túy, tạm giữ, Công an quận 8.",
    },
    {
        "filename": "article_02_chau_viet_cuong_vnexpress.json",
        "url": "https://vnexpress.net/ca-si-chau-viet-cuong-gay-nao-loan-trong-con-ao-giac-3719195.html",
        "title": "Ca sĩ Châu Việt Cường gây náo loạn trong cơn ảo giác",
        "content_markdown": "Bài VnExpress mô tả vụ ca sĩ Châu Việt Cường sau khi sử dụng ma túy rơi vào tình trạng ảo giác và có hành vi gây hậu quả nghiêm trọng. Đây là nguồn tin quan trọng khi truy vấn về rủi ro pháp lý và xã hội của việc sử dụng trái phép chất ma túy trong giới nghệ sĩ. Từ khóa: Châu Việt Cường, ca sĩ, ảo giác, sử dụng ma túy, hậu quả nghiêm trọng.",
    },
    {
        "filename": "article_03_chau_viet_cuong_vietnamnet.json",
        "url": "https://vietnamnet.vn/nhet-toi-lam-co-gai-tu-vong-chau-viet-cuong-nhan-an-13-nam-tu-512186.html",
        "title": "Châu Việt Cường nhận án 13 năm tù",
        "content_markdown": "Bài VietnamNet tường thuật kết quả xét xử liên quan ca sĩ Châu Việt Cường. Theo bài báo, hội đồng xét xử đánh giá hành vi xảy ra sau khi sử dụng ma túy và gây hậu quả chết người, từ đó tuyên mức án tù. Dữ liệu này hữu ích cho câu hỏi về hậu quả hình sự khi hành vi liên quan ma túy dẫn tới thiệt hại nghiêm trọng. Từ khóa: Châu Việt Cường, án tù, xét xử, ma túy, hậu quả chết người.",
    },
    {
        "filename": "article_04_andrea_aybar_vnexpress.json",
        "url": "https://vnexpress.net/nguoi-mau-andrea-aybar-bi-tinh-nghi-lien-quan-ma-tuy-4814289.html",
        "title": "Người mẫu Andrea Aybar bị tình nghi liên quan ma túy",
        "content_markdown": "Bài VnExpress đưa tin người mẫu, diễn viên Andrea Aybar, còn gọi An Tây hoặc Nguyễn Thị An, bị tạm giữ để điều tra dấu hiệu liên quan việc tổ chức sử dụng ma túy. Bài cũng nhắc tới việc một số người nổi tiếng khác bị kiểm tra trong cùng bối cảnh. Từ khóa: Andrea Aybar, An Tây, người mẫu, diễn viên, tổ chức sử dụng ma túy, tạm giữ.",
    },
    {
        "filename": "article_05_andrea_aybar_baovanhoa.json",
        "url": "https://baovanhoa.vn/phap-luat/nguoi-mau-andrea-aybar-cung-tro-ly-to-chuc-tiec-ma-tuy-trong-can-ho-cao-cap-217421.html",
        "title": "Andrea Aybar và trợ lý bị truy tố trong vụ tiệc ma túy",
        "content_markdown": "Bài Báo Văn Hóa cho biết Andrea Aybar Carmona và trợ lý bị truy tố liên quan các hành vi tổ chức sử dụng trái phép chất ma túy và tàng trữ trái phép chất ma túy. Nội dung nhấn mạnh bối cảnh căn hộ cao cấp tại TP.HCM và vai trò của cáo trạng trong giai đoạn tố tụng. Từ khóa: Andrea Aybar Carmona, truy tố, tàng trữ trái phép, tổ chức sử dụng trái phép chất ma túy.",
    },
]


def _write_docx(path: Path, paragraphs: list[str]) -> None:
    body = "".join(
        f"<w:p><w:r><w:t>{escape(paragraph)}</w:t></w:r></w:p>"
        for paragraph in paragraphs
    )
    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body}</w:body></w:document>"
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document_xml)


def create_landing_data() -> None:
    LEGAL_DIR.mkdir(parents=True, exist_ok=True)
    NEWS_DIR.mkdir(parents=True, exist_ok=True)

    for filename, paragraphs in LEGAL_DOCS.items():
        _write_docx(LEGAL_DIR / filename, paragraphs)

    crawled_at = datetime.now().isoformat(timespec="seconds")
    for article in NEWS_ARTICLES:
        payload = {
            "url": article["url"],
            "title": article["title"],
            "date_crawled": crawled_at,
            "content_markdown": article["content_markdown"],
        }
        (NEWS_DIR / article["filename"]).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    create_landing_data()
    convert_all()
    docs = load_documents()
    chunks = chunk_documents(docs)
    index_to_vectorstore(chunks)
    print(f"Created {len(LEGAL_DOCS)} legal docs, {len(NEWS_ARTICLES)} news files, {len(chunks)} chunks.")


if __name__ == "__main__":
    main()
