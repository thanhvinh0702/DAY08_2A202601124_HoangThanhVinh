"""Streamlit chat UI for the Mầm e-commerce RAG assistant."""

from __future__ import annotations

import html
import logging
import os
import sys
from pathlib import Path

import streamlit as st

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False


ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
load_dotenv()

if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="Mầm | Trợ lý Shopee",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --surface: #ffffff;
        --surface-soft: #f5faf5;
        --surface-muted: #edf6ee;
        --line: #d8e8da;
        --text: #17251a;
        --text-muted: #607363;
        --forest: #1b5e20;
        --leaf: #43a047;
        --sage: #a5d6a7;
        --warm: #f4a261;
    }
    [data-testid="stHeader"] {
        height: 3rem;
        background: rgba(251,253,251,.94);
        border-bottom: 1px solid #edf2ed;
    }
    [data-testid="stToolbar"] { display: none; }
    [data-testid="stAppViewContainer"] { background: #fbfdfb; }
    .main .block-container {
        max-width: 960px;
        padding-top: 4.1rem;
        padding-bottom: 7rem;
    }
    [data-testid="stSidebar"] {
        background: var(--forest);
        border-right: 0;
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
    [data-testid="stSidebar"] * { color: #f4fff5; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { opacity: .78; }
    [data-testid="stSidebar"] .stButton button {
        min-height: 2.6rem;
        border: 1px solid rgba(255,255,255,.18);
        border-radius: 8px;
        background: rgba(255,255,255,.07);
        text-align: left;
        justify-content: flex-start;
        transition: background .16s ease, border-color .16s ease;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,.15);
        border-color: rgba(255,255,255,.38);
    }
    [data-testid="stSidebar"] [role="radiogroup"] {
        background: rgba(255,255,255,.08);
        border-radius: 8px;
        padding: .35rem .55rem;
    }
    .brand-row {
        display: flex;
        align-items: center;
        gap: .75rem;
        padding: .2rem 0 1rem;
    }
    .brand-mark {
        width: 38px;
        height: 38px;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: var(--leaf);
        color: white;
        font-size: 1.25rem;
        font-weight: 800;
    }
    .brand-name { color: var(--text); font-size: 1.05rem; font-weight: 750; }
    .brand-subtitle { color: var(--text-muted); font-size: .8rem; margin-top: .05rem; }
    .welcome {
        padding: 4rem 1rem 2rem;
        text-align: center;
    }
    .welcome-mark {
        width: 48px;
        height: 48px;
        margin: 0 auto 1rem;
        display: grid;
        place-items: center;
        border-radius: 8px;
        background: var(--forest);
        color: white;
        font-size: 1.4rem;
    }
    .welcome h1 {
        margin: 0;
        color: var(--text);
        font-size: 1.75rem;
        letter-spacing: 0;
    }
    .welcome p { margin: .6rem 0 0; color: var(--text-muted); }
    [data-testid="stChatMessage"] {
        padding: 1rem 1.1rem;
        margin: .55rem 0;
        border-radius: 8px;
        border: 1px solid transparent;
        background: transparent;
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background: var(--surface-muted);
        border-color: var(--line);
    }
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        background: var(--surface);
        border-color: #e8eee9;
    }
    [data-testid="stChatInput"] {
        border: 1px solid #bdd8c0;
        border-radius: 8px;
        background: var(--surface);
        box-shadow: 0 8px 28px rgba(27,94,32,.09);
    }
    [data-testid="stChatInput"]:focus-within {
        border-color: var(--leaf);
        box-shadow: 0 0 0 3px rgba(67,160,71,.12), 0 8px 28px rgba(27,94,32,.09);
    }
    .source-card {
        margin: .55rem 0;
        padding: .8rem .9rem;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface-soft);
    }
    .source-title { color: var(--forest); font-weight: 700; }
    .source-meta { margin: .18rem 0 .5rem; color: var(--text-muted); font-size: .78rem; }
    .source-content { color: #344a38; font-size: .88rem; line-height: 1.55; }
    div[data-testid="stExpander"] {
        border: 1px solid var(--line);
        border-radius: 8px;
        background: var(--surface-soft);
    }
    @media (max-width: 700px) {
        .main .block-container { padding: 3.8rem .8rem 6.5rem; }
        .welcome { padding-top: 2.5rem; }
        .welcome h1 { font-size: 1.45rem; }
        [data-testid="stChatMessage"] { padding: .8rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

SUGGESTIONS = [
    "Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?",
    "Shopee hỗ trợ những phương thức thanh toán nào?",
    "Làm sao để đổi phương thức thanh toán đơn hàng?",
    "Quy định về đăng bán sản phẩm cho người bán?",
    "Cách mua hàng trên Shopee của quốc gia khác?",
]
ROLE_OPTIONS = {
    "Tự động": None,
    "Người mua": "buyer",
    "Người bán": "seller",
}


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander(f"Nguồn tham khảo · {len(sources)} đoạn", expanded=False):
        for index, source in enumerate(sources, 1):
            metadata = source.get("metadata") or {}
            name = metadata.get("title") or metadata.get("source") or "Nguồn không xác định"
            kind = metadata.get("type") or metadata.get("doc_type") or "tài liệu"
            topic = metadata.get("topic")
            role = metadata.get("customer_role")
            method = source.get("retrieval_method") or source.get("source")
            score = source.get("score")
            details = [str(kind)]
            if topic:
                details.append(str(topic))
            if role:
                details.append(str(role))
            if method:
                details.append(str(method))
            if score is not None:
                details.append(f"score {float(score):.4f}")
            content = str(source.get("content") or "").strip()
            if len(content) > 480:
                content = f"{content[:480].rstrip()}..."
            st.markdown(
                '<div class="source-card">'
                f'<div class="source-title">[{index}] {html.escape(str(name))}</div>'
                f'<div class="source-meta">{html.escape(" · ".join(details))}</div>'
                f'<div class="source-content">{html.escape(content)}</div>'
                "</div>",
                unsafe_allow_html=True,
            )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

with st.sidebar:
    st.markdown("## Mầm")
    st.caption("Trợ lý chính sách Shopee")
    st.divider()
    st.caption("CÂU HỎI GỢI Ý")
    for index, suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion, use_container_width=True, key=f"suggestion_{index}"):
            st.session_state.pending_query = suggestion
    st.divider()
    st.caption("NGỮ CẢNH")
    selected_role_label = st.radio(
        "Vai trò khách hàng",
        options=list(ROLE_OPTIONS),
        horizontal=False,
        label_visibility="collapsed",
    )
    customer_role = ROLE_OPTIONS[selected_role_label]
    top_k = st.slider("Số nguồn", min_value=3, max_value=10, value=5)
    st.divider()
    if st.button("Xóa hội thoại", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pending_query = None
        st.rerun()

st.markdown(
    '<div class="brand-row"><div class="brand-mark">M</div>'
    '<div><div class="brand-name">Mầm</div>'
    '<div class="brand-subtitle">Hỗ trợ chính sách Shopee</div></div></div>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="welcome"><div class="welcome-mark">M</div>'
        '<h1>Mình có thể giúp gì cho bạn?</h1>'
        '<p>Hỏi về thanh toán, đổi trả, giao hàng hoặc chính sách tài khoản.</p></div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    avatar = "🌱" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

typed_query = st.chat_input("Hỏi Mầm về chính sách Shopee...")
query = typed_query or st.session_state.pending_query
if query:
    query = str(query).strip()
    st.session_state.pending_query = None
    LOGGER.info("UI query=%r | top_k=%d role=%s", query, top_k, customer_role or "auto")
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user", avatar="👤"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🌱"):
        with st.spinner("Mầm đang tìm trong tài liệu..."):
            try:
                from src.task10_generation import generate_with_citation

                response = generate_with_citation(
                    query,
                    top_k=top_k,
                    customer_role=customer_role,
                )
                answer = response.get("answer") or "Chưa thể tạo câu trả lời."
                sources = response.get("sources") or []
                status = response.get("status", "ok")
                LOGGER.info(
                    "UI response | status=%s sources=%d retrieval=%s",
                    status,
                    len(sources),
                    response.get("retrieval_source"),
                )
            except Exception:
                LOGGER.exception("UI pipeline failed")
                answer = "Mầm chưa thể kết nối pipeline lúc này. Vui lòng kiểm tra log và thử lại."
                sources = []
                status = "pipeline_error"

        st.markdown(answer)
        if status in {"retrieval_error", "configuration_error", "generation_error", "pipeline_error"}:
            st.caption("Hệ thống chưa sẵn sàng đầy đủ. Chi tiết đã được ghi trong terminal.")
        render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "status": status,
        }
    )
