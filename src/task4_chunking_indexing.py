"""
Task 4 — Chunking & Indexing vào Vector Store.

Thiết kế hiện tại:
    - Chunk Markdown bằng RecursiveCharacterTextSplitter
    - chunk_size = 800, chunk_overlap = 100
    - Gắn metadata customer_role = buyer | seller | both
    - Embed bằng BAAI/bge-m3 (1024 chiều)
    - Lưu vào ChromaDB với cosine distance
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import gc
from functools import lru_cache
from pathlib import Path
from typing import Iterable
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
METADATA_MANIFEST_PATH = Path(__file__).parent.parent / "data" / "metadata" / "document_metadata.json"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
LOGGER = logging.getLogger(__name__)

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


# =============================================================================
# CONFIGURATION
# =============================================================================

# 800/100 cân bằng giữa giữ ngữ cảnh và tránh làm chunk quá dài.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# BGE-M3 là mặc định của bài: multilingual và trả vector dense 1024 chiều.
# OpenAI/OpenRouter vẫn được hỗ trợ để chạy nhẹ hơn khi máy không có model local.
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "bge_m3").strip().lower()
LOCAL_EMBEDDING_MODEL = os.getenv("BGE_M3_MODEL", "BAAI/bge-m3").strip()
OPENAI_EMBEDDING_MODEL = os.getenv(
    "OPENAI_EMBEDDING_MODEL",
    "openai/text-embedding-3-small" if os.getenv("OPENROUTER_API_KEY") else "text-embedding-3-small",
).strip()
EMBEDDING_MODEL = (
    OPENAI_EMBEDDING_MODEL if EMBEDDING_PROVIDER == "openai" else LOCAL_EMBEDDING_MODEL
)
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))

VECTOR_STORE = "chromadb"  # "chromadb" | "weaviate" | "faiss"
COLLECTION_NAME = "ecommerce_support_docs"

VALID_CUSTOMER_ROLES = {"buyer", "seller", "both"}
DEFAULT_CUSTOMER_ROLE = "both"


# =============================================================================
# METADATA
# =============================================================================

def _load_metadata_manifest() -> dict[str, dict]:
    if not METADATA_MANIFEST_PATH.exists():
        return {}
    try:
        data = json.loads(METADATA_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Không đọc được manifest metadata: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Manifest metadata phải là JSON object")
    return data


def _normalize_customer_role(value: str | None) -> str:
    role = (value or DEFAULT_CUSTOMER_ROLE).strip().lower()
    if role not in VALID_CUSTOMER_ROLES:
        return DEFAULT_CUSTOMER_ROLE
    return role


def _infer_customer_role(text: str, relative_path: str) -> str:
    source = f"{relative_path}\n{text}".lower()
    seller_terms = [
        "người bán",
        "seller",
        "đăng bán",
        "bán hàng",
        "phí sàn",
        "quản lý shop",
        "nhà bán",
        "người gửi hàng",
        "kho hàng",
    ]
    buyer_terms = [
        "người mua",
        "buyer",
        "mua hàng",
        "trả hàng",
        "hoàn tiền",
        "đổi trả",
        "thanh toán",
        "đơn hàng",
        "giao hàng",
        "chăm sóc khách hàng",
        "bảo mật",
        "giao dịch",
    ]
    seller_hits = sum(term in source for term in seller_terms)
    buyer_hits = sum(term in source for term in buyer_terms)

    if seller_hits and buyer_hits:
        return "both"
    if seller_hits:
        return "seller"
    if buyer_hits:
        return "buyer"
    return DEFAULT_CUSTOMER_ROLE


def _get_manifest_entry(manifest: dict[str, dict], relative_path: str) -> dict:
    """Support both paths relative to data/standardized and data/."""
    return manifest.get(relative_path) or manifest.get(f"standardized/{relative_path}") or {}


def _resolve_chunk_customer_role(chunk_text: str, doc_metadata: dict) -> str:
    chunk_role = _infer_customer_role(chunk_text, doc_metadata.get("path", ""))
    doc_role = _normalize_customer_role(doc_metadata.get("customer_role"))
    if chunk_role != DEFAULT_CUSTOMER_ROLE:
        return chunk_role
    return doc_role


def _resolve_document_metadata(
    md_file: Path, content: str, manifest: dict[str, dict] | None = None
) -> dict:
    relative_path = md_file.relative_to(STANDARDIZED_DIR).as_posix()
    manifest = manifest if manifest is not None else _load_metadata_manifest()
    manifest_entry = _get_manifest_entry(manifest, relative_path)
    doc_type = manifest_entry.get("doc_type") or ("legal" if "legal" in md_file.parts else "news")
    customer_role = _normalize_customer_role(
        manifest_entry.get("customer_role") or _infer_customer_role(content, relative_path)
    )

    return {
        "source": md_file.name,
        "path": relative_path,
        "type": doc_type,
        "title": manifest_entry.get("title") or md_file.stem.replace("-", " ").replace("_", " "),
        "topic": manifest_entry.get("topic") or "general",
        "customer_role": customer_role,
    }


# =============================================================================
# DOCUMENT LOADING
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {..., 'customer_role': str}}
    """
    documents = []
    if not STANDARDIZED_DIR.exists():
        LOGGER.warning("Standardized directory does not exist: %s", STANDARDIZED_DIR)
        return documents

    manifest = _load_metadata_manifest()
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if not md_file.is_file():
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        documents.append(
            {
                "content": content,
                "metadata": _resolve_document_metadata(md_file, content, manifest),
            }
        )
    LOGGER.info("Loaded %d markdown documents from %s", len(documents), STANDARDIZED_DIR)
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    if CHUNKING_METHOD != "recursive":
        raise NotImplementedError(
            f"CHUNKING_METHOD='{CHUNKING_METHOD}' chưa được implement trong repo này"
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        LOGGER.debug(
            "Chunking %s -> %d chunks | role=%s | type=%s",
            doc["metadata"].get("path"),
            len(splits),
            doc["metadata"].get("customer_role"),
            doc["metadata"].get("type"),
        )
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append(
                {
                    "content": chunk_text,
                    "metadata": {
                        **doc["metadata"],
                        "document_customer_role": doc["metadata"].get("customer_role", DEFAULT_CUSTOMER_ROLE),
                        "customer_role": _resolve_chunk_customer_role(chunk_text, doc["metadata"]),
                        "chunk_index": i,
                        "chunk_size": CHUNK_SIZE,
                        "chunk_overlap": CHUNK_OVERLAP,
                    },
                }
            )
    LOGGER.info(
        "Created %d chunks from %d documents (chunk_size=%d, overlap=%d)",
        len(chunks),
        len(documents),
        CHUNK_SIZE,
        CHUNK_OVERLAP,
    )
    return chunks


# =============================================================================
# EMBEDDING
# =============================================================================

@lru_cache(maxsize=1)
def _get_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def _get_openai_client():
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    api_key = openrouter_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Thiếu OPENROUTER_API_KEY hoặc OPENAI_API_KEY để tạo embeddings")

    client_kwargs = {"api_key": api_key}
    if openrouter_key:
        client_kwargs["base_url"] = "https://openrouter.ai/api/v1"
    return OpenAI(**client_kwargs)


def _batched(items: list[str], batch_size: int = 32) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts bằng provider được cấu hình.
    Mặc định dùng BAAI/bge-m3 để ra vector 1024 chiều.
    """
    if not texts:
        return []

    LOGGER.info(
        "Embedding %d texts | provider=%s model=%s expected_dim=%d",
        len(texts),
        EMBEDDING_PROVIDER,
        EMBEDDING_MODEL,
        EMBEDDING_DIM,
    )
    if EMBEDDING_PROVIDER in {"bge_m3", "sentence_transformers", "local"}:
        model = _get_sentence_transformer()
        embeddings = model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if hasattr(embeddings, "tolist"):
            embeddings = embeddings.tolist()
        vectors = embeddings

    elif EMBEDDING_PROVIDER == "openai":
        client = _get_openai_client()
        vectors: list[list[float]] = []
        for batch in _batched(texts):
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
            )
            vectors.extend(item.embedding for item in response.data)

    else:
        raise NotImplementedError(
            f"EMBEDDING_PROVIDER='{EMBEDDING_PROVIDER}' chưa được hỗ trợ"
        )

    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding API trả {len(vectors)} vectors cho {len(texts)} texts")
    invalid_dimensions = {len(vector) for vector in vectors if len(vector) != EMBEDDING_DIM}
    if invalid_dimensions:
        raise RuntimeError(
            "Embedding dimension không khớp: "
            f"expected={EMBEDDING_DIM}, received={sorted(invalid_dimensions)}. "
            "Hãy reindex bằng cùng provider/model/dimension."
        )
    return vectors


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embed_texts(texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError("Số embeddings không khớp số chunks")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


# =============================================================================
# VECTOR STORE
# =============================================================================

def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb

    if not chunks:
        raise RuntimeError("Không có chunks để index")
    if any("embedding" not in chunk for chunk in chunks):
        raise RuntimeError("Chunks chưa có embedding; hãy gọi embed_chunks() trước")

    staging_dir = CHROMA_DIR.with_name(f"{CHROMA_DIR.name}_staging")
    backup_dir = CHROMA_DIR.with_name(f"{CHROMA_DIR.name}_backup")
    shutil.rmtree(staging_dir, ignore_errors=True)
    shutil.rmtree(backup_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(staging_dir))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": EMBEDDING_PROVIDER,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
        },
    )
    LOGGER.info(
        "Writing %d chunks into Chroma collection=%s at %s",
        len(chunks),
        COLLECTION_NAME,
        staging_dir,
    )

    ids = [
        f"{chunk['metadata']['path']}__{chunk['metadata']['chunk_index']}"
        for chunk in chunks
    ]
    collection.upsert(
        ids=ids,
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    LOGGER.info("Index complete in staging dir | collection count=%d", collection.count())

    # Đóng các handle SQLite trước khi đổi thư mục và giữ bản cũ để rollback.
    del collection
    del client
    gc.collect()
    try:
        if CHROMA_DIR.exists():
            shutil.move(str(CHROMA_DIR), str(backup_dir))
        shutil.move(str(staging_dir), str(CHROMA_DIR))
    except Exception:
        if not CHROMA_DIR.exists() and backup_dir.exists():
            shutil.move(str(backup_dir), str(CHROMA_DIR))
        raise
    shutil.rmtree(backup_dir, ignore_errors=True)

    final_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    final_collection = final_client.get_collection(name=COLLECTION_NAME)
    LOGGER.info("Index published | final collection count=%d", final_collection.count())
    return final_collection


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print(f"  Metadata manifest: {METADATA_MANIFEST_PATH}")
    print("=" * 60)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    role_counts: dict[str, int] = {"buyer": 0, "seller": 0, "both": 0}
    for doc in docs:
        role = doc["metadata"].get("customer_role", DEFAULT_CUSTOMER_ROLE)
        role_counts[role] = role_counts.get(role, 0) + 1
    print(f"✓ Customer roles: {role_counts}")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
