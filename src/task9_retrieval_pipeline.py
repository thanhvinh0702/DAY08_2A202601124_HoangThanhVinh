"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Kết hợp semantic search + lexical search + reranking + PageIndex fallback
thành một pipeline thống nhất.

Logic:
    1. Chạy semantic_search + lexical_search song song
    2. Merge kết quả (RRF hoặc weighted fusion)
    3. Rerank
    4. Nếu top result score < threshold → fallback sang PageIndex
    5. Return top_k results

⚠️ BẪY THƯỜNG GẶP — đọc kỹ trước khi code:
    Nếu bạn dùng điểm RRF đã fuse (Task 7) để so với score_threshold, bạn sẽ gặp bug
    thật: RRF max score luôn ≈ 1/(k+1) ≈ 0.0164 (k=60) BẤT KỂ nội dung có liên quan
    hay không. Nếu đặt threshold thấp (như 0.005) để "hợp" với thang điểm RRF, thực
    chất KHÔNG câu hỏi nào đủ thấp để trigger fallback nữa — kể cả query hoàn toàn vô
    nghĩa vẫn trả về kết quả "hybrid" (rác) thay vì fallback đúng như thiết kế.

    Cách sửa đúng: giữ điểm cosine similarity GỐC của semantic_search (trước khi qua
    RRF) làm căn cứ quyết định fallback, tách biệt khỏi điểm RRF dùng để sắp xếp kết
    quả cuối cùng. Calibrate threshold bằng cách tự đo: chạy vài câu hỏi chắc chắn
    liên quan và vài câu chắc chắn lạc đề/rác qua semantic_search, xem khoảng cách
    điểm số giữa hai nhóm rồi chọn ngưỡng nằm giữa.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from .retrieval_utils import detect_customer_role, normalize_customer_role
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD = 0.3   # Nếu best score (cosine gốc) < threshold → fallback PageIndex
DEFAULT_TOP_K = 5
LOGGER = logging.getLogger(__name__)



def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    customer_role: str | None = None,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh với fallback logic.

    Pipeline:
        Query
          ├→ Semantic Search → dense_results (giữ điểm cosine gốc)
          ├→ Lexical Search  → sparse_results
          │
          ├→ Merge + Rerank (RRF) → reranked_results
          │
          └→ If dense_results[0]["score"] < threshold:
                └→ PageIndex Vectorless → fallback_results

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả cuối cùng
        score_threshold: Ngưỡng điểm cosine gốc tối thiểu (KHÔNG phải điểm RRF)
        use_reranking: Có gộp bằng RRF (Task 7) hay giữ nguyên thứ tự dense_results

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip():
        return []
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")

    query = query.strip()
    requested_role = normalize_customer_role(customer_role)
    effective_role = requested_role or detect_customer_role(query)

    LOGGER.info(
        "Retrieve start | query=%r | top_k=%d | threshold=%.3f | rerank=%s | role=%s",
        query,
        top_k,
        score_threshold,
        use_reranking,
        effective_role or "all",
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        dense_future = executor.submit(
            semantic_search, query, top_k * 2, effective_role
        )
        sparse_future = executor.submit(
            lexical_search, query, top_k * 2, effective_role
        )
        try:
            dense_results = dense_future.result()
        except Exception:
            LOGGER.exception("Semantic search failed")
            dense_results = []
        try:
            sparse_results = sparse_future.result()
        except Exception:
            LOGGER.exception("Lexical search failed")
            sparse_results = []

    LOGGER.info(
        "Retrieval candidates | dense=%d sparse=%d",
        len(dense_results),
        len(sparse_results),
    )

    if use_reranking and (dense_results or sparse_results):
        merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 2)
    else:
        merged = (dense_results or sparse_results)[: top_k * 2]

    retrieval_method = "hybrid" if dense_results and sparse_results else (
        "semantic" if dense_results else "bm25"
    )
    for item in merged:
        # Giữ contract Task 9: source là hybrid hoặc pageindex.
        item["source"] = "hybrid"
        item["retrieval_method"] = retrieval_method
        item["retrieval_role"] = effective_role or "all"

    LOGGER.info("Merged candidates=%d", len(merged))

    final_results = merged[:top_k]

    for item in final_results:
        item.setdefault("source", "hybrid")

    best_score = dense_results[0]["score"] if dense_results else None
    LOGGER.info(
        "Best dense score=%s | threshold=%.4f",
        f"{best_score:.4f}" if best_score is not None else "unavailable",
        score_threshold,
    )
    should_fallback = not final_results or (
        best_score is not None and best_score < score_threshold
    )
    if should_fallback:
        LOGGER.warning(
            "Fallback triggered | reason=%s",
            "no_results" if not final_results else "low_dense_score",
        )
        fallback = pageindex_search(query, top_k=top_k, customer_role=effective_role)
        if fallback:
            LOGGER.info("PageIndex fallback returned %d chunks", len(fallback))
            return fallback
        LOGGER.warning("PageIndex fallback returned no chunks")

    # Dense có thể tạm lỗi/không được index; BM25 thật vẫn là evidence hợp lệ.
    if best_score is None and final_results:
        LOGGER.warning("Dense unavailable; returning positive-score BM25 evidence")

    LOGGER.info("Returning %d %s chunks", len(final_results[:top_k]), retrieval_method)
    return final_results[:top_k]


if __name__ == "__main__":
    test_queries = [
        "What payment methods does Shopee support?",
        "How do I request a return or refund?",
        "What evidence do I need for a refund request?",
        "xyzabc123nonsense",  # Query không có kết quả → test fallback
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] [{r['source']}] {r['content'][:80]}...")
