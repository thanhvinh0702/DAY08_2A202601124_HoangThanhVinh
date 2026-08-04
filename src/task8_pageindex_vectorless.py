"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API

Lưu ý: API `/retrieval` của PageIndex hiện đã deprecated (vẫn hoạt động, nhưng response
có field "deprecation" cảnh báo) và trả kết quả trong "retrieved_nodes" — mỗi node có
"relevant_contents": list[list[{section_title, relevant_content}]]. In response thật ra
(json.dumps(...)) trước khi viết logic parse, đừng đoán schema từ ví dụ code cũ.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
try:
    from dotenv import load_dotenv
except ImportError:  # Cho phép import module khi dependency tùy chọn chưa cài.
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
DOCUMENT_MAP_PATH = Path(__file__).parent.parent / "data" / "pageindex_documents.json"
UPLOAD_DIR = Path(__file__).parent.parent / "data" / "pageindex_uploads"
POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 180


def _require_api_key() -> None:
    if not PAGEINDEX_API_KEY:
        raise RuntimeError("Thiếu PAGEINDEX_API_KEY trong file .env")


def _load_document_map() -> dict:
    if not DOCUMENT_MAP_PATH.exists():
        return {}
    try:
        data = json.loads(DOCUMENT_MAP_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Không đọc được {DOCUMENT_MAP_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"{DOCUMENT_MAP_PATH} phải chứa một JSON object")
    return data


def _save_document_map(document_map: dict) -> None:
    DOCUMENT_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOCUMENT_MAP_PATH.write_text(
        json.dumps(document_map, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _markdown_to_pdf(md_file: Path, pdf_path: Path) -> None:
    """Chuyển Markdown thành PDF Unicode đơn giản để PageIndex xử lý."""
    from fpdf import FPDF

    font_candidates = [
        Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
    ]
    font_path = next((path for path in font_candidates if path.exists()), None)
    if font_path is None:
        raise RuntimeError("Không tìm thấy font Unicode để tạo PDF")

    text = md_file.read_text(encoding="utf-8-sig")
    # Giữ cấu trúc heading nhưng bỏ các marker Markdown gây nhiễu khi tạo PDF.
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^]]*]\([^)]*\)", "", text)
    text = re.sub(r"\[([^]]+)]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"^(#{1,6})\s+", "", text, flags=re.MULTILINE)
    text = text.replace("**", "").replace("__", "").replace("`", "")

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_font("Unicode", fname=str(font_path))
    pdf.add_page()
    pdf.set_font("Unicode", size=11)
    for paragraph in text.splitlines():
        paragraph = paragraph.strip()
        if paragraph:
            pdf.multi_cell(0, 6, paragraph)
        else:
            pdf.ln(3)
    pdf.output(str(pdf_path))


def upload_documents() -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    _require_api_key()
    from pageindex import PageIndexClient

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print(f"Không tìm thấy file Markdown trong {STANDARDIZED_DIR}")
        return {}

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    document_map = _load_document_map()

    for md_file in md_files:
        source = md_file.relative_to(STANDARDIZED_DIR).as_posix()
        if document_map.get(source, {}).get("doc_id"):
            print(f"  - Bỏ qua file đã upload: {source}")
            continue

        pdf_path = UPLOAD_DIR / md_file.relative_to(STANDARDIZED_DIR).with_suffix(".pdf")
        _markdown_to_pdf(md_file, pdf_path)
        response = client.submit_document(str(pdf_path))
        doc_id = response.get("doc_id") or response.get("id")
        if not doc_id:
            raise RuntimeError(f"PageIndex không trả doc_id cho {source}: {response!r}")

        document_map[source] = {
            "doc_id": doc_id,
            "pdf_path": str(pdf_path.relative_to(Path(__file__).parent.parent)),
            "uploaded_at": datetime.now().astimezone().isoformat(),
        }
        # Ghi ngay sau mỗi upload để không mất ID nếu lần upload sau gặp lỗi.
        _save_document_map(document_map)
        print(f"  ✓ Uploaded: {source} -> {doc_id}")

    return document_map


def _wait_for_retrieval(client, retrieval_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        response = client.get_retrieval(retrieval_id)
        status = str(response.get("status", "")).lower()
        if status == "completed":
            return response
        if status in {"failed", "error", "cancelled"}:
            raise RuntimeError(f"PageIndex retrieval thất bại: {response!r}")
        time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(
        f"PageIndex retrieval {retrieval_id} chưa xong sau {POLL_TIMEOUT_SECONDS}s"
    )


def _iter_relevant_items(value):
    """Duyệt được cả relevant_contents dạng list[dict] và list[list[dict]]."""
    if isinstance(value, list):
        for child in value:
            yield from _iter_relevant_items(child)
    elif isinstance(value, dict):
        if "relevant_content" in value:
            yield value
        else:
            for child in value.values():
                if isinstance(child, (list, dict)):
                    yield from _iter_relevant_items(child)


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")
    # PageIndex là fallback tùy chọn: thiếu key hoặc chưa upload thì trả rỗng.
    if not PAGEINDEX_API_KEY:
        return []

    document_map = _load_document_map()
    if not document_map:
        return []

    from pageindex import PageIndexClient

    client = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    parsed_results = []
    for source_file, document in document_map.items():
        doc_id = document.get("doc_id")
        if not doc_id or not client.is_retrieval_ready(doc_id):
            continue

        submitted = client.submit_query(doc_id=doc_id, query=query.strip())
        retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
        if not retrieval_id:
            raise RuntimeError(f"PageIndex không trả retrieval_id: {submitted!r}")
        retrieval = _wait_for_retrieval(client, retrieval_id)

        for node in retrieval.get("retrieved_nodes") or []:
            for item in _iter_relevant_items(node.get("relevant_contents") or []):
                content = str(item.get("relevant_content") or "").strip()
                if not content:
                    continue
                parsed_results.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": source_file,
                            "doc_id": doc_id,
                            "node_id": node.get("node_id"),
                            "section": item.get("section_title") or node.get("title"),
                            "page_index": item.get("page_index"),
                        },
                        "source": "pageindex",
                    }
                )

    # Legacy retrieval không cung cấp similarity score; dùng reciprocal rank.
    for rank, result in enumerate(parsed_results, start=1):
        result["score"] = round(1.0 / rank, 4)
    return parsed_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("danh sách sản phẩm cấm đăng bán", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
