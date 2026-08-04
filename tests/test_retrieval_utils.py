from src.retrieval_utils import (
    detect_customer_role,
    normalize_customer_role,
    role_matches,
)


def test_detects_buyer_intent():
    assert detect_customer_role("Thời hạn trả hàng hoàn tiền là bao lâu?") == "buyer"


def test_detects_seller_intent():
    assert detect_customer_role("Người bán đăng bán sản phẩm thế nào?") == "seller"


def test_ambiguous_intent_searches_all_roles():
    assert detect_customer_role("Chính sách bảo mật của Shopee") is None


def test_both_metadata_matches_specific_roles():
    assert role_matches("both", "buyer")
    assert role_matches("both", "seller")


def test_specific_metadata_does_not_cross_roles():
    assert not role_matches("buyer", "seller")
    assert not role_matches("seller", "buyer")


def test_invalid_requested_role_becomes_unfiltered():
    assert normalize_customer_role("admin") is None
    assert role_matches("buyer", "admin")
