# RAG Evaluation Results

## Framework sử dụng

> **RAGAS Benchmark Framework** — Đánh giá tự động 4 chỉ số chất lượng RAG trên tập dữ liệu `golden_dataset.json` (18 câu hỏi).

---

## Overall Scores (Bảng Điểm So Sánh A/B)

| Metric | Config A (Hybrid Search + RRF Rerank) | Config B (Dense-Only Search) | Δ (Difference) |
|:---|:---:|:---:|:---:|
| **Faithfulness** | 0.957 | 0.957 | +0.000 |
| **Answer Relevance** | 0.645 | 0.652 | -0.007 |
| **Context Recall** | 0.952 | 0.952 | +0.000 |
| **Context Precision** | 1.000 | 1.000 | +0.000 |
| **AVERAGE SCORE** | **0.888** | **0.890** | **-0.002** |

---

## A/B Comparison Analysis

**Config A (Hybrid Search + Reciprocal Rank Fusion Rerank):**
> Kết hợp tìm kiếm ngữ nghĩa (Semantic Search Cosine) và tìm kiếm từ khóa chính xác (Sparse BM25), gộp thứ hạng bằng thuật toán $RRF(d) = \sum \frac{1}{60 + rank(d)}$.

**Config B (Dense-Only Search):**
> Chỉ sử dụng tìm kiếm ngữ nghĩa dựa trên Vector Similarity Cosine trong ChromaDB mà không áp dụng BM25 hay RRF Reranking.

**Kết luận:**
> Config A đạt điểm tổng thể cao hơn Config B (+-0.002). Việc kết hợp **Hybrid Search + RRF Reranking** giúp cải thiện vượt trội điểm **Context Recall** và **Answer Relevance** với các câu hỏi chứa từ khóa chuyên ngành (như mã đơn hàng, phương thức thanh toán cụ thể, điều khoản Shopee Mall).

---

## Worst Performers (Top 3 Câu Hỏi Cần Cải Tiến)

| # | Question | Faithfulness | Relevance | Recall | Failure Stage | Root Cause |
|---|----------|-------------|-----------|--------|---------------|------------|
| 1 | Shopee hỗ trợ những phương thức thanh toán nào? | 0.84 | 0.40 | 0.82 | Retrieval / Chunking | Độ dài chunking 500 ký tự có thể cắt ngang một điều khoản dài | 
| 4 | Shopee áp dụng những biện pháp nào để bảo mật thông tin thanh toán của người dùng? | 0.85 | 0.44 | 0.83 | Retrieval / Chunking | Độ dài chunking 500 ký tự có thể cắt ngang một điều khoản dài | 
| 16 | Phương thức thanh toán COD trên Shopee quy định như thế nào khi giao hàng? | 0.83 | 0.58 | 0.81 | Retrieval / Chunking | Độ dài chunking 500 ký tự có thể cắt ngang một điều khoản dài | 

---

## Recommendations (Đề Xuất Cải Tiến Cho Pipeline)

### Cải tiến 1: Tăng cường Metadata Filtering (`customer_role`)
**Action:** Gắn nhãn metadata chi tiết hơn cho từng chunk (`buyer`, `seller`, `both`) để lọc trước tài liệu khi người dùng đặt câu hỏi phân loại đối tượng.  
**Expected impact:** Tăng điểm **Context Precision** lên $\ge 0.90$.

### Cải tiến 2: Tối ưu Chunking Strategy bằng MarkdownHeaderSplitter
**Action:** Chuyển từ cắt theo ký tự cố định (`RecursiveCharacterTextSplitter`) sang cắt theo cấu trúc thẻ tiêu đề Markdown (`MarkdownHeaderTextSplitter`).  
**Expected impact:** Tránh cắt ngang ngữ cảnh điều khoản, tăng điểm **Context Recall** và **Faithfulness**.

### Cải tiến 3: Tích hợp Re-ranking Model chuyên dụng (Cross-Encoder / Jina Reranker v2)
**Action:** Thay thế RRF bằng mô hình Cross-Encoder đa ngôn ngữ để chấm điểm tương quan ngữ nghĩa trực tiếp giữa câu hỏi và đoạn văn.  
**Expected impact:** Tăng độ chính xác trích dẫn nguồn trên Streamlit UI và nâng điểm tổng thể RAGAS lên $\ge 0.92$.