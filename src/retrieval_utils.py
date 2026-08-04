"""Shared validation and audience helpers for retrieval modules."""

from __future__ import annotations

VALID_QUERY_ROLES = {"buyer", "seller"}

_SELLER_TERMS = (
    "người bán",
    "seller",
    "nhà bán",
    "bán hàng",
    "đăng bán",
    "shop của tôi",
    "kênh người bán",
    "phí sàn",
    "doanh thu",
    "quản lý shop",
)
_BUYER_TERMS = (
    "người mua",
    "buyer",
    "mua hàng",
    "đặt hàng",
    "đơn hàng",
    "thanh toán",
    "trả hàng",
    "hoàn tiền",
    "đổi trả",
    "nhận hàng",
    "giao hàng",
)


def normalize_customer_role(customer_role: str | None) -> str | None:
    """Return a retrieval role or None when the query should search all roles."""
    if customer_role is None:
        return None
    role = str(customer_role).strip().lower()
    return role if role in VALID_QUERY_ROLES else None


def detect_customer_role(query: str) -> str | None:
    """Infer buyer/seller from explicit intent; ambiguous queries stay unfiltered."""
    normalized = str(query or "").lower()
    seller_hits = sum(term in normalized for term in _SELLER_TERMS)
    buyer_hits = sum(term in normalized for term in _BUYER_TERMS)
    if seller_hits > buyer_hits:
        return "seller"
    if buyer_hits > seller_hits:
        return "buyer"
    return None


def role_matches(metadata_role: str | None, customer_role: str | None) -> bool:
    """A role-specific query may use documents for that role or for both roles."""
    requested = normalize_customer_role(customer_role)
    if requested is None:
        return True
    return str(metadata_role or "both").strip().lower() in {requested, "both"}

