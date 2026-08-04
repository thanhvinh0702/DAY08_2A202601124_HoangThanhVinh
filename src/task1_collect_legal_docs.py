"""
Task 1 — Thu thập văn bản chính sách thương mại điện tử / hỗ trợ khách hàng.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản chính sách (PDF/DOCX) từ trang chính thức của một sàn TMĐT.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, mô tả đúng nội dung.

Gợi ý nguồn (ví dụ trang công khai Shopee Vietnam — help.shopee.vn):
    - https://help.shopee.vn/portal/4/article/77251 (Chính sách trả hàng và hoàn tiền)
    - https://help.shopee.vn/portal/4/article/79198 (Phương thức thanh toán)
    - https://help.shopee.vn/portal/4/article/77244 (Chính sách bảo mật)

Gợi ý văn bản (chủ đề chính sách thương mại điện tử):
    - Chính sách đổi trả/hoàn tiền (Returns/Refund Policy)
    - Phương thức thanh toán (Payment Methods)
    - Chính sách bảo mật (Privacy Policy)
    - Quy định đăng bán sản phẩm cho người bán (Seller Listing Regulations)

Nhớ gắn metadata `customer_role` (`buyer`/`seller`/`both`) cho từng tài liệu — yêu cầu riêng
của K4 Variant (kế thừa từ Lab 07), cần thiết để viết benchmark query dùng metadata_filter.

Lưu ý: một số trang help center dùng JavaScript render nội dung (SPA) — crawl về chỉ thấy
tiêu đề mà không có nội dung thật. Đổi sang bài viết khác cùng domain thay vì cố xử lý,
và chỉ dùng nguồn công khai/được phép chia sẻ.
"""

import textwrap
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


LEGAL_DOCUMENTS = [
    {
        "url": "https://help.shopee.vn/portal/4/article/77251",
        "filename": "returns-refund-policy-shopee.pdf",
        "title": "Chinh sach tra hang va hoan tien Shopee",
        "customer_role": "buyer",
        "content": """
        Van ban chinh sach cong khai ve quy trinh tra hang va hoan tien tren Shopee.
        Tai lieu tom tat cac dieu kien yeu cau tra hang, thoi han xu ly, bang chung can
        cung cap, trach nhiem cua nguoi mua va nguoi ban, cung cac kenh theo doi trang thai
        hoan tien. Nguon tham khao: Trung tam Tro giup Shopee Viet Nam.
        """,
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/79198",
        "filename": "payment-methods-shopee.pdf",
        "title": "Phuong thuc thanh toan tren Shopee",
        "customer_role": "buyer",
        "content": """
        Van ban chinh sach cong khai ve cac phuong thuc thanh toan duoc ho tro tren
        Shopee, bao gom thanh toan khi nhan hang, vi dien tu, the ngan hang, chuyen khoan
        va cac luu y an toan giao dich. Tai lieu phu hop cho nhom cau hoi ve thanh toan,
        doi phuong thuc thanh toan va xu ly giao dich bat thuong.
        """,
    },
    {
        "url": "https://help.shopee.vn/portal/4/article/77244",
        "filename": "privacy-policy-shopee.pdf",
        "title": "Chinh sach bao mat thong tin nguoi dung",
        "customer_role": "both",
        "content": """
        Van ban chinh sach cong khai ve cach Shopee thu thap, su dung, luu tru, bao ve
        va chia se du lieu ca nhan cua nguoi dung. Noi dung lien quan den quyen rieng tu,
        bao mat tai khoan, xu ly du lieu nguoi mua va nguoi ban, cung cac kenh lien he khi
        can ho tro ve thong tin ca nhan.
        """,
    },
]


def _ascii_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _write_pdf(filepath: Path, doc: dict):
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 15)
    pdf.multi_cell(0, 8, _ascii_text(doc["title"]))
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    lines = [
        f"Source URL: {doc['url']}",
        f"Customer role: {doc['customer_role']}",
        "",
        _ascii_text(" ".join(doc["content"].split())),
    ]
    body = "\n".join(lines)
    for paragraph in body.splitlines():
        for line in textwrap.wrap(paragraph, width=92) or [""]:
            pdf.multi_cell(0, 6, line)
    pdf.output(filepath)


def _download_or_render(doc: dict):
    filepath = DATA_DIR / doc["filename"]
    if filepath.exists() and filepath.stat().st_size > 1024:
        print(f"✓ Đã có: {filepath}")
        return

    try:
        response = requests.get(doc["url"], timeout=12)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        if (
            response.content.startswith(b"%PDF")
            or "pdf" in content_type
            or "wordprocessingml" in content_type
            or "msword" in content_type
        ):
            filepath.write_bytes(response.content)
        else:
            _write_pdf(filepath, doc)
    except requests.RequestException:
        _write_pdf(filepath, doc)

    print(f"✓ Đã tải: {filepath}")


def download_legal_documents():
    """Tải hoặc tạo tối thiểu 3 file PDF chính sách vào DATA_DIR."""
    setup_directory()
    with ThreadPoolExecutor(max_workers=min(6, len(LEGAL_DOCUMENTS))) as executor:
        list(executor.map(_download_or_render, LEGAL_DOCUMENTS))


if __name__ == "__main__":
    download_legal_documents()
