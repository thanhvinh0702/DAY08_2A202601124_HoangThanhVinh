"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_embedding_model():
    """Load một lần đúng embedding model được cấu hình ở Task 4."""
    from sentence_transformers import SentenceTransformer

    try:
        from .task4_chunking_indexing import EMBEDDING_MODEL
    except ImportError:  # Hỗ trợ chạy trực tiếp: python src/task5_semantic_search.py
        from task4_chunking_indexing import EMBEDDING_MODEL

    return SentenceTransformer(EMBEDDING_MODEL)


def _get_collection():
    """Mở Chroma collection đã được Task 4 tạo và index."""
    import chromadb

    try:
        from .task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME
    except ImportError:  # Hỗ trợ chạy trực tiếp: python src/task5_semantic_search.py
        from task4_chunking_indexing import CHROMA_DIR, COLLECTION_NAME

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


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
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
    if collection is None or collection.count() == 0:
        return []

    model = _get_embedding_model()
    query_vector = model.encode(query.strip(), convert_to_numpy=True).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

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
    return output[:top_k]


if __name__ == "__main__":
    # Test
    results = semantic_search("quy định trả hàng hoàn tiền shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
