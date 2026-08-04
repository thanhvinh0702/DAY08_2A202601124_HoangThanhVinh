"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "I cannot verify this information"

Gợi ý LLM: OpenRouter có nhiều model gắn hậu tố ":free" không tính phí — xem
https://openrouter.ai/models?max_price=0 — phù hợp nếu chưa có credit trả phí.
Base URL: "https://openrouter.ai/api/v1", dùng chung interface với OpenAI SDK.
"""

import os
import logging
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

from .task9_retrieval_pipeline import retrieve
LOGGER = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): đủ linh hoạt nhưng vẫn ưu tiên câu trả lời ổn định.
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# RAG chính sách cần factual và ít sáng tạo.
TEMPERATURE = 0.1

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Bạn là Mầm, trợ lý trả lời câu hỏi về chính sách thương mại điện tử và
hỗ trợ khách hàng (thanh toán, đổi trả, giao hàng, quyền riêng tư, quy định người bán).

Quy tắc bắt buộc:
1. Chỉ sử dụng thông tin từ context được cung cấp — KHÔNG bịa đặt
2. Trích dẫn bằng đúng nhãn [Nguồn N] được cung cấp; không tự tạo tên hoặc năm
3. Chỉ nói không thể xác minh khi không nguồn nào trực tiếp hỗ trợ câu trả lời
4. Trả lời bằng tiếng Việt, đi thẳng vào câu hỏi và trình bày rõ ràng
5. Có thể tổng hợp nhiều câu nằm trong context nhưng không suy đoán ngoài context"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks phải là list")
    if len(chunks) <= 2:
        return list(chunks)
    # Giữ item điểm cao nhất ở đầu; các item tiếp theo được phân bố về hai đầu.
    return list(chunks[::2]) + list(chunks[1::2])[::-1]


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string.
    """
    if not isinstance(chunks, list):
        raise TypeError("chunks phải là list")
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or metadata.get("file") or f"Source {i}"
        doc_type = metadata.get("type") or metadata.get("doc_type") or "unknown"
        section = metadata.get("section") or metadata.get("section_title")
        location = f" | Section: {section}" if section else ""
        role = metadata.get("customer_role") or "both"
        context_parts.append(
            f"[Nguồn {i}] Source: {source} | Type: {doc_type} | Role: {role}{location}\n"
            f"{str(chunk.get('content') or '').strip()}"
        )
    return "\n\n---\n\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(
    query: str,
    top_k: int = TOP_K,
    use_reranking: bool = True,
    customer_role: str | None = None,
) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số lượng tài liệu tham khảo
        use_reranking: Có áp dụng RRF reranking hay không

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")

    LOGGER.info(
        "Generation start | query=%r | top_k=%d | rerank=%s",
        query.strip(),
        top_k,
        use_reranking,
    )
    try:
        chunks = retrieve(
            query.strip(),
            top_k=top_k,
            use_reranking=use_reranking,
            customer_role=customer_role,
        )
        retrieval_error = None
    except Exception as exc:
        LOGGER.exception("Retrieval failed inside generation")
        chunks = []
        retrieval_error = str(exc)
    chunks = list(chunks or [])[:top_k]
    reordered = reorder_for_llm(chunks)
    retrieval_source = chunks[0].get("source", "hybrid") if chunks else "none"
    LOGGER.info("Retrieved %d chunks | source=%s", len(chunks), retrieval_source)
    no_evidence = "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not chunks or not api_key:
        if not chunks:
            LOGGER.warning("No chunks available, returning no-evidence answer")
        if not api_key:
            LOGGER.warning("No API key available for generation")
        status = "retrieval_error" if retrieval_error else (
            "no_evidence" if not chunks else "configuration_error"
        )
        return {
            "answer": no_evidence,
            "sources": chunks,
            "retrieval_source": retrieval_source,
            "status": status,
            "error": retrieval_error,
        }

    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if os.getenv("OPENROUTER_API_KEY"):
        client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
    client = OpenAI(**client_kwargs)
    user_message = (
        "Context (chỉ dùng thông tin trong các tài liệu dưới đây):\n"
        f"{format_context(reordered)}\n\n---\n\n"
        f"Question: {query.strip()}\n"
        "Trả lời bằng tiếng Việt; đặt citation [Nguồn N] ngay sau thông tin được trích dẫn."
    )
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=TEMPERATURE,
            top_p=TOP_P,
        )
        answer = (response.choices[0].message.content or "").strip() or no_evidence
    except Exception as exc:
        LOGGER.exception("LLM generation failed")
        return {
            "answer": "Mầm chưa thể tạo câu trả lời lúc này. Vui lòng thử lại sau.",
            "sources": chunks,
            "retrieval_source": retrieval_source,
            "status": "generation_error",
            "error": str(exc),
        }
    LOGGER.info("Generation complete | answer_len=%d", len(answer))
    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": retrieval_source,
        "status": "ok" if answer != no_evidence else "no_evidence",
        "error": None,
    }


if __name__ == "__main__":
    test_queries = [
        "Shopee hỗ trợ những phương thức thanh toán nào?",
        "Làm sao để yêu cầu đổi trả hay hoàn tiền?",
        "Cần chuẩn bị bằng chứng gì khi yêu cầu hoàn tiền?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Sources: {len(result['sources'])} chunks | via {result['retrieval_source']}]")
