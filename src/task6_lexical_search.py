import re
import unicodedata
from pathlib import Path
from rank_bm25 import BM25Okapi
import numpy as np

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}
_BM25_INDEX: BM25Okapi | None = None


def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để tăng khả năng matching từ khóa."""
    nfkd_form = unicodedata.normalize("NFKD", input_str)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def tokenize(text: str) -> list[str]:
    """Tokenize văn bản: tách từ, chuyển chữ thường và tạo biến thể không dấu."""
    text_lower = text.lower()
    words = re.findall(r"\w+", text_lower)
    tokens = []
    for w in words:
        tokens.append(w)
        unaccented = remove_accents(w)
        if unaccented != w:
            tokens.append(unaccented)
    return tokens


def load_corpus_from_standardized() -> list[dict]:
    """Tải và chia đoạn (chunking) toàn bộ tài liệu .md từ data/standardized/."""
    corpus = []
    if not STANDARDIZED_DIR.exists():
        return corpus

    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        doc_title = md_file.stem.replace("-", " ").replace("_", " ")

        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        for i, para in enumerate(paragraphs):
            indexed_text = f"[{doc_title}] {para}"
            corpus.append({
                "content": para,
                "indexed_text": indexed_text,
                "metadata": {
                    "source": md_file.name,
                    "type": doc_type,
                    "chunk_index": i
                }
            })
    return corpus


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    if not corpus:
        return None
    tokenized_corpus = [tokenize(doc.get("indexed_text", doc["content"])) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


def _ensure_corpus_and_index():
    global CORPUS, _BM25_INDEX
    if not CORPUS:
        CORPUS = load_corpus_from_standardized()
    if _BM25_INDEX is None and CORPUS:
        _BM25_INDEX = build_bm25_index(CORPUS)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    _ensure_corpus_and_index()

    if not CORPUS or _BM25_INDEX is None:
        return []

    tokenized_query = tokenize(query)
    if not tokenized_query:
        return []

    scores = _BM25_INDEX.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        results.append({
            "content": CORPUS[idx]["content"],
            "score": score,
            "metadata": CORPUS[idx]["metadata"]
        })
        if len(results) >= top_k:
            break

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("phương thức thanh toán shopee", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")

