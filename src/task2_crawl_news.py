"""
Task 2 — Crawl bài viết/hướng dẫn hỗ trợ khách hàng về thương mại điện tử.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài viết từ trung tâm trợ giúp công khai của một sàn TMĐT.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
    playwright install chromium   # bắt buộc — pip install crawl4ai KHÔNG tự tải browser binary,
                                   # thiếu bước này sẽ báo lỗi
                                   # "BrowserType.launch: Executable doesn't exist"

Gợi ý chủ đề: theo dõi đơn hàng, đổi phương thức thanh toán, bằng chứng hoàn tiền,
mua hàng xuyên biên giới.

Lưu ý: một số trang help center dùng JavaScript render (SPA) — nếu crawl về chỉ thấy
tiêu đề mà không có nội dung, đổi sang bài viết khác cùng domain thay vì cố xử lý.
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


ARTICLE_URLS = [
    "https://help.shopee.vn/portal/4/article/79072-%5BC%E1%BA%A3nh-b%C3%A1o-l%E1%BB%ABa-%C4%91%E1%BA%A3o%5D-Mua-s%E1%BA%AFm-an-to%C3%A0n-c%C3%B9ng-Shopee?previousPage=hot%20issues",
    "https://help.shopee.vn/portal/4/article/79191-%5BD%E1%BB%8Bch-v%E1%BB%A5%5D-C%C3%A1ch-li%C3%AAn-h%E1%BB%87-Ch%C4%83m-s%C3%B3c-kh%C3%A1ch-h%C3%A0ng-Shopee?previousPage=hot%20issues",
    "https://help.shopee.vn/portal/4/article/125827-%5BB%E1%BA%A3o-m%E1%BA%ADt-t%C3%A0i-kho%E1%BA%A3n%5D-T%C3%B4i-c%E1%BA%A7n-l%C3%A0m-g%C3%AC-n%E1%BA%BFu-c%C3%B3-giao-d%E1%BB%8Bch-l%E1%BA%A1-ph%C3%A1t-sinh-tr%C3%AAn-th%E1%BA%BB-t%C3%ADn-d%E1%BB%A5ng%2Ft%C3%A0i-kho%E1%BA%A3n-ng%C3%A2n-h%C3%A0ng-c%E1%BB%A7a-t%C3%B4i?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/79422-Shopee-L%C3%A0-G%C3%AC?previousPage=secondary%20category",
    "https://help.shopee.vn/portal/4/article/85565-%5BD%E1%BB%8Bch-v%E1%BB%A5%5D-T%C3%B4i-mu%E1%BB%91n-g%E1%BB%ADi-%C3%BD-ki%E1%BA%BFn-ph%E1%BA%A3n-h%E1%BB%93i-%C4%91%E1%BA%BFn-Shopee?previousPage=secondary%20category",
]


async def crawl_article(url: str) -> dict:
    """
    Crawl một bài viết và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(
            url=url,
            config=CrawlerRunConfig(
                wait_until="domcontentloaded",
                delay_before_return_html=2.0,
            ),
        )

    if not getattr(result, "success", True):
        error = getattr(result, "error_message", None) or "Unknown crawl error"
        raise RuntimeError(f"Failed to crawl {url}: {error}")

    metadata = getattr(result, "metadata", None) or {}
    markdown = getattr(result, "markdown", "") or ""

    # Crawl4AI cac phien ban moi tra ve MarkdownGenerationResult, trong khi
    # cac phien ban cu tra ve truc tiep mot chuoi Markdown.
    if not isinstance(markdown, str):
        markdown = getattr(markdown, "raw_markdown", None) or str(markdown)

    markdown = markdown.strip()
    if len(markdown) < 100:
        raise RuntimeError(
            f"Nội dung crawl từ {url} quá ngắn ({len(markdown)} ký tự). "
            "Trang có thể chưa render xong hoặc không cho phép crawler truy cập."
        )

    return {
        "url": url,
        "title": metadata.get("title") or "Unknown",
        "date_crawled": datetime.now().astimezone().isoformat(),
        "content_markdown": markdown,
    }


async def crawl_all():
    """Crawl toàn bộ bài viết trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = await crawl_article(url)
        except Exception as exc:
            print(f"  Crawl failed: {exc}")
            continue

        # Lưu file JSON
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(
            json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  ✓ Saved: {filepath}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm trang hướng dẫn/hỗ trợ khách hàng trên help center của sàn TMĐT")
    else:
        asyncio.run(crawl_all())
