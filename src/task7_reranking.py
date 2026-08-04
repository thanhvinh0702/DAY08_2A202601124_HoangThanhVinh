"""
Task 7 — Reranking Module.

Phương pháp đã chọn: RRF (Reciprocal Rank Fusion) — tự implement, không cần API key.

RRF(d) = Σ 1 / (k + r(d))

Trong đó:
    - d: một document (candidate)
    - r(d): thứ hạng (rank, bắt đầu từ 1) của d trong 1 ranked list cụ thể
      (nếu d không xuất hiện trong ranked list đó thì bỏ qua, không cộng gì)
    - k: hằng số smoothing (mặc định 60, theo paper Cormack et al. 2009)
    - Tổng Σ được lấy trên tất cả các ranked list (ở đây là Semantic Search và BM25)

Cơ chế: RRF gộp thứ hạng từ nhiều hệ thống retrieval khác nhau (semantic search
dùng cosine similarity, BM25 dùng lexical score) mà không cần chuẩn hoá /
so sánh trực tiếp thang điểm của chúng — vốn không cùng đơn vị nên không thể
cộng/so sánh thẳng. RRF chỉ quan tâm THỨ HẠNG: document đứng càng cao (rank càng
nhỏ) ở càng nhiều ranker thì điểm RRF càng lớn, nên được đẩy lên đầu danh sách
sau khi fuse.

Lưu ý quan trọng (sẽ dùng lại ở Task 9): điểm RRF fused CHỈ phụ thuộc thứ hạng,
không phải độ tương đồng thật. Top-1 sau khi fuse luôn xấp xỉ 1/(k+1) ≈ 0.0164
(k=60), bất kể nội dung đó có thật sự liên quan đến câu hỏi hay không. Đừng dùng
điểm RRF để quyết định fallback ở Task 9 — xem ghi chú ở đó.
"""


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker (vd: Semantic Search + BM25).

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists, mỗi list là kết quả (đã sort
            theo score giảm dần) từ 1 ranker, dạng [{'content': str, 'score': float,
            'metadata': dict}, ...]
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending. Mỗi item giữ
        nguyên 'content'/'metadata' của lần xuất hiện đầu tiên, 'score' được
        thay bằng điểm RRF đã fuse.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in content_map:
                content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Unified reranking interface — chỉ dùng RRF để gộp Semantic Search + BM25.

    Args:
        ranked_lists: List các ranked list cần gộp, vd:
            [semantic_search(query, top_k=20), lexical_search(query, top_k=20)]
        top_k: Số lượng kết quả sau rerank
        k: Smoothing constant cho RRF

    Returns:
        List of top_k reranked candidates.
    """
    return rerank_rrf(ranked_lists, top_k=top_k, k=k)


if __name__ == "__main__":
    # Test with dummy data: giả lập kết quả từ 2 ranker khác nhau
    # (Semantic Search và BM25) cho cùng 1 câu query, mỗi list đã sort sẵn.
    semantic_results = [
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 0.82, "metadata": {}},
        {"content": "Quy định đăng bán sản phẩm dành cho người bán", "score": 0.71, "metadata": {}},
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 0.65, "metadata": {}},
    ]
    bm25_results = [
        {"content": "Các phương thức thanh toán hỗ trợ trên Shopee Vietnam", "score": 9.4, "metadata": {}},
        {"content": "Chính sách trả hàng và hoàn tiền Shopee trong 15 ngày", "score": 8.1, "metadata": {}},
        {"content": "Điều khoản sử dụng dịch vụ vận chuyển Shopee Express", "score": 6.7, "metadata": {}},
    ]

    results = rerank([semantic_results, bm25_results], top_k=4)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
