"""Cute Streamlit UI for the e-commerce RAG assistant."""
import sys
from pathlib import Path
import streamlit as st
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
st.set_page_config(page_title="Mầm · E-commerce Assistant", page_icon="🌱", layout="wide")
st.markdown("""
<style>
:root{--mint:#E8F5E9;--sage:#A5D6A7;--leaf:#66BB6A;--forest:#1B5E20}
.stApp{background:linear-gradient(135deg,#E8F5E9,#F8FFF8 50%,#E8F5E9)}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#1B5E20,#2E7D32)}
[data-testid="stSidebar"] *{color:#F1FFF2!important}
[data-testid="stSidebar"] .stButton>button{border:1px solid #A5D6A7;border-radius:14px;background:#ffffff18;text-align:left;font-size:.86rem}
[data-testid="stSidebar"] .stButton>button:hover{background:#66BB6A;transform:translateY(-1px)}
.hero{background:#fff;border:2px solid #A5D6A7;border-radius:28px;padding:24px 30px;box-shadow:0 10px 30px #1b5e2018;margin-bottom:16px}
.hero h1{color:#1B5E20;margin:0;font-size:2rem}.hero p{color:#558B5A;margin:.4rem 0 0}
.welcome{background:#ffffffb8;border:1px dashed #66BB6A;border-radius:22px;padding:25px;text-align:center;color:#356B3A;margin:18px 0}
[data-testid="stChatMessage"]{border-radius:22px;padding:10px 15px;margin:10px 0}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]){background:#DDF2DF}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]){background:#fff;border:1px solid #C8E8CA;box-shadow:0 4px 14px #1b5e2010}
[data-testid="stChatInput"]{background:#fff;border:2px solid #A5D6A7;border-radius:20px}
[data-testid="stChatInput"]:focus-within{border-color:#66BB6A;box-shadow:0 0 0 3px #66bb6a2e}
.source-card{background:#F5FFF5;border:1px solid #A5D6A7;border-radius:14px;padding:10px 13px;margin:7px 0}.source-title{color:#1B5E20;font-weight:700}.source-meta{color:#67946B;font-size:.78rem}
div[data-testid="stExpander"]{border-color:#A5D6A7;border-radius:15px;background:#ffffff80}
</style>
""", unsafe_allow_html=True)

SUGGESTIONS=["Thời hạn yêu cầu trả hàng/hoàn tiền là bao lâu?","Shopee hỗ trợ những phương thức thanh toán nào?","Làm sao để đổi phương thức thanh toán đơn hàng?","Quy định về đăng bán sản phẩm cho người bán?","Cách mua hàng trên Shopee của quốc gia khác?"]
with st.sidebar:
    st.markdown("# 🌿 Mầm trợ lý")
    st.caption("Hỏi đáp chính sách e-commerce thật nhẹ nhàng và dễ hiểu.")
    st.divider(); st.subheader("💡 Câu hỏi gợi ý")
    for i,suggestion in enumerate(SUGGESTIONS):
        if st.button(suggestion,use_container_width=True,key=f"suggestion_{i}"): st.session_state.pending_query=suggestion
    st.divider(); st.subheader("⚙️ Thiết lập")
    top_k=st.slider("Số tài liệu tham khảo (top_k)",3,10,5,help="Số chunks đưa vào ngữ cảnh.")
    st.caption(f"Đang dùng **{top_k}** nguồn gần nhất")
    st.divider(); st.caption("**Pipeline:** Semantic + BM25 → RRF → PageIndex → LLM citation")

if "messages" not in st.session_state: st.session_state.messages=[]
if "pending_query" not in st.session_state: st.session_state.pending_query=None
st.markdown('<div class="hero"><h1>🌱 Trợ lý Shopee thân thiện</h1><p>Hỏi mình về thanh toán, đổi trả, giao hàng và chính sách mua sắm nhé ✨</p></div>',unsafe_allow_html=True)

def render_sources(sources):
    if not sources:return
    with st.expander(f"📚 Nguồn tham khảo · {len(sources)} tài liệu"):
        for i,source in enumerate(sources,1):
            meta=source.get("metadata") or {}; name=meta.get("source") or meta.get("file") or "Nguồn không xác định"; kind=meta.get("type") or meta.get("doc_type") or "tài liệu"; section=meta.get("section") or meta.get("section_title"); score=source.get("score"); details=kind+(f" · {section}" if section else "")
            if score is not None: details+=f" · độ phù hợp {float(score):.3f}"
            content=str(source.get("content") or "").strip(); content=content[:360].rstrip()+"…" if len(content)>360 else content
            st.markdown(f'<div class="source-card"><div class="source-title">[{i}] {name}</div><div class="source-meta">{details}</div><div>{content}</div></div>',unsafe_allow_html=True)

if not st.session_state.messages: st.markdown('<div class="welcome"><div style="font-size:2.2rem">🪴</div><h3 style="color:#1B5E20;margin:.3rem 0">Xin chào, mình là Mầm!</h3><p>Chọn một câu hỏi bên trái hoặc gõ câu hỏi bên dưới để bắt đầu nhé.</p></div>',unsafe_allow_html=True)
for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar="🌱" if message["role"]=="assistant" else "🧑‍💻"):
        st.markdown(message["content"])
        if message["role"]=="assistant": render_sources(message.get("sources",[]))

typed_query=st.chat_input("Nhập câu hỏi về chính sách/hỗ trợ e-commerce…"); query=typed_query or st.session_state.pending_query
if query:
    st.session_state.pending_query=None; st.session_state.messages.append({"role":"user","content":query})
    with st.chat_message("user",avatar="🧑‍💻"): st.markdown(query)
    with st.chat_message("assistant",avatar="🌱"):
        with st.spinner("Mầm đang tìm tài liệu phù hợp…"):
            try:
                from src.task10_generation import generate_with_citation
                response=generate_with_citation(query,top_k=top_k); answer=response.get("answer","Chưa thể trả lời."); sources=response.get("sources",[])
            except Exception as exc:
                answer=f"⚠️ Chưa thể kết nối pipeline: `{exc}`"; sources=[]
        st.markdown(answer); render_sources(sources)
    st.session_state.messages.append({"role":"assistant","content":answer,"sources":sources})
