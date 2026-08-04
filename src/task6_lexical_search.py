"""Task 6 - BM25 lexical retrieval over the same chunks used by Chroma."""

from __future__ import annotations

import logging
import re
import unicodedata

from rank_bm25 import BM25Okapi

try:
    from .retrieval_utils import normalize_customer_role, role_matches
    from .task4_chunking_indexing import chunk_documents, load_documents
except ImportError:  # Support: python src/task6_lexical_search.py
    from retrieval_utils import normalize_customer_role, role_matches  # type: ignore
    from task4_chunking_indexing import chunk_documents, load_documents  # type: ignore


LOGGER = logging.getLogger(__name__)
CORPUS: list[dict] = []
_BM25_INDEX: BM25Okapi | None = None


def remove_accents(input_str: str) -> str:
    """Remove Vietnamese accents while keeping the original token variant too."""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join(c for c in nfkd_form if not unicodedata.combining(c))


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization with accented and unaccented variants."""
    tokens: list[str] = []
    for word in re.findall(r"\w+", str(text or "").lower(), flags=re.UNICODE):
        tokens.append(word)
        unaccented = remove_accents(word)
        if unaccented != word:
            tokens.append(unaccented)
    return tokens


def load_corpus_from_standardized() -> list[dict]:
    """Build BM25 input from Task 4 chunks so both retrievers share metadata."""
    chunks = chunk_documents(load_documents())
    corpus = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        title = metadata.get("title") or metadata.get("source") or ""
        corpus.append(
            {
                "content": chunk.get("content") or "",
                "indexed_text": f"{title}\n{chunk.get('content') or ''}",
                "metadata": metadata,
            }
        )
    LOGGER.info("Built BM25 corpus with %d chunks", len(corpus))
    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi | None:
    if not corpus:
        return None
    return BM25Okapi(
        [tokenize(doc.get("indexed_text", doc.get("content", ""))) for doc in corpus]
    )


def _ensure_corpus_and_index() -> None:
    global CORPUS, _BM25_INDEX
    if not CORPUS:
        CORPUS = load_corpus_from_standardized()
    if _BM25_INDEX is None and CORPUS:
        _BM25_INDEX = build_bm25_index(CORPUS)


def reset_bm25_cache() -> None:
    """Clear the process cache after standardized documents change."""
    global CORPUS, _BM25_INDEX
    CORPUS = []
    _BM25_INDEX = None


def lexical_search(
    query: str, top_k: int = 10, customer_role: str | None = None
) -> list[dict]:
    """Return positive-score BM25 chunks, optionally scoped by customer role."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query phải là chuỗi không rỗng")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k phải là số nguyên dương")

    _ensure_corpus_and_index()
    if not CORPUS or _BM25_INDEX is None:
        return []

    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = _BM25_INDEX.get_scores(query_tokens)
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
    role = normalize_customer_role(customer_role)
    results = []
    for index in ranked_indices:
        score = float(scores[index])
        if score <= 0:
            break
        metadata = CORPUS[index]["metadata"]
        if not role_matches(metadata.get("customer_role"), role):
            continue
        results.append(
            {
                "content": CORPUS[index]["content"],
                "score": round(score, 4),
                "metadata": metadata,
            }
        )
        if len(results) >= top_k:
            break

    LOGGER.info("BM25 returned %d/%d chunks | role=%s", len(results), top_k, role or "all")
    return results


if __name__ == "__main__":
    for result in lexical_search("phương thức thanh toán shopee", top_k=5):
        print(f"[{result['score']:.3f}] {result['content'][:100]}...")
