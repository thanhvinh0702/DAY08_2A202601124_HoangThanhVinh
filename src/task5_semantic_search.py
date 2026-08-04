"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import logging

try:
    from .task4_chunking_indexing import (
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        EMBEDDING_PROVIDER,
        embed_texts,
    )
    from .retrieval_utils import normalize_customer_role
except ImportError:  # Hỗ trợ chạy trực tiếp: python src/task5_semantic_search.py
    from task4_chunking_indexing import (  # type: ignore
        CHROMA_DIR,
        COLLECTION_NAME,
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        EMBEDDING_PROVIDER,
        embed_texts,
    )
    from retrieval_utils import normalize_customer_role  # type: ignore


LOGGER = logging.getLogger(__name__)


def _get_collection():
    """Mở Chroma collection đã được Task 4 tạo và index."""
    import chromadb

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        # Chroma dùng các exception khác nhau giữa các phiên bản. Chỉ coi đây
        # là collection chưa được index nếu tên collection thực sự không tồn tại.
        existing_names = {
            getattr(collection, "name", str(collection))
            for collection in client.list_collections()
        }
        if COLLECTION_NAME not in existing_names:
            return None
        raise RuntimeError(f"Không thể mở Chroma collection: {exc}") from exc


def semantic_search(
    query: str, top_k: int = 10, customer_role: str | None = None
) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")
    collection = _get_collection()
    if collection is None:
        LOGGER.warning("Chroma collection '%s' does not exist", COLLECTION_NAME)
        return []

    collection_count = collection.count()
    if collection_count == 0:
        LOGGER.warning("Chroma collection '%s' is empty", COLLECTION_NAME)
        return []

    metadata = collection.metadata or {}
    indexed_model = metadata.get("embedding_model")
    indexed_provider = metadata.get("embedding_provider")
    indexed_dim = metadata.get("embedding_dim")
    mismatch_reasons = []
    if indexed_model and str(indexed_model) != EMBEDDING_MODEL:
        mismatch_reasons.append(f"model={indexed_model!r} (runtime={EMBEDDING_MODEL!r})")
    if indexed_provider and str(indexed_provider) != EMBEDDING_PROVIDER:
        mismatch_reasons.append(
            f"provider={indexed_provider!r} (runtime={EMBEDDING_PROVIDER!r})"
        )
    if indexed_dim and int(indexed_dim) != EMBEDDING_DIM:
        mismatch_reasons.append(f"dimension={indexed_dim} (runtime={EMBEDDING_DIM})")

    sample = collection.peek(limit=1)
    stored_vectors = sample.get("embeddings")
    if stored_vectors is not None and len(stored_vectors):
        stored_dim = len(stored_vectors[0])
        if stored_dim != EMBEDDING_DIM:
            mismatch_reasons.append(
                f"stored_dimension={stored_dim} (runtime={EMBEDDING_DIM})"
            )
    if mismatch_reasons:
        LOGGER.error(
            "Chroma index is incompatible: %s. Run `python src/task4_chunking_indexing.py`.",
            "; ".join(mismatch_reasons),
        )
        return []

    query_vector = embed_texts([query.strip()])[0]
    role = normalize_customer_role(customer_role)
    query_kwargs = dict(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection_count),
        include=["documents", "metadatas", "distances"],
    )
    if role:
        query_kwargs["where"] = {"customer_role": {"$in": [role, "both"]}}

    results = collection.query(**query_kwargs)

    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    output = []
    for document, metadata, distance in zip(documents, metadatas, distances):
        # Collection ở Task 4 dùng hnsw:space="cosine", do đó similarity = 1 - distance.
        # Clamp để tránh sai số dấu phẩy động vượt nhẹ khỏi miền cosine [-1, 1].
        score = max(-1.0, min(1.0, 1.0 - float(distance)))
        output.append(
            {
                "content": document or "",
                "score": round(score, 4),
                "metadata": metadata or {},
            }
        )

    output.sort(key=lambda item: item["score"], reverse=True)
    LOGGER.info(
        "Semantic search returned %d/%d chunks | collection=%d role=%s",
        len(output[:top_k]),
        top_k,
        collection_count,
        role or "all",
    )
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
