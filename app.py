import streamlit as st
import torch
import datetime
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizerFast
from model_def import NewsClassifier

REPO_ID = "Theara2/cambodian-news-roberta"
LABELS = ["Politics", "Technology", "Economics", "Health", "Sports"]
COLORS = {
    "Politics": "#8b5cf6",
    "Technology": "#22c55e",
    "Economics": "#3b82f6",
    "Health": "#ef4444",
    "Sports": "#f59e0b",
}

st.set_page_config(page_title="Cambodian News Classifier", layout="wide")

# ---------------- GLOBAL STYLE ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #f5f6f8 !important;
}

/* Hide Streamlit's default top toolbar so it doesn't overlap our navbar */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}
div[data-testid="stToolbar"] {
    display: none;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container { padding-top: 2rem !important; max-width: 1200px; }

/* Make Streamlit's native buttons in the navbar look like white pill tabs */
div[data-testid="column"] .stButton > button {
    background-color: white;
    color: #374151;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    font-weight: 600;
    font-size: 13px;
}
div[data-testid="column"] .stButton > button[kind="primary"] {
    background-color: #14306b;
    color: white;
    border: none;
}
/* Text area / inputs forced to light styling */
.stTextArea textarea, .stTextInput input {
    background-color: #f9fafb !important;
    color: #111827 !important;
    border: 1px solid #e5e7eb !important;
}
.analyze-btn-wrap .stButton > button {
    background-color: #16336e !important;
    color: white !important;
    border: none !important;
    font-weight: 700;
}
.analyze-btn-wrap .stButton > button:hover {
    background-color: #0f2554 !important;
}

.navbar {
    background: #16336e;
    color: white;
    padding: 14px 28px;
    border-radius: 10px 10px 0 0;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0;
}
.navbar-left { display: flex; align-items: center; gap: 12px; }
.navbar-logo {
    width: 34px; height: 34px; border-radius: 9px;
    background: #2563eb; display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 15px;
}
.navbar-title { font-size: 14px; font-weight: 700; line-height: 1.2; }
.navbar-sub { font-size: 10px; opacity: 0.6; letter-spacing: 0.5px; }

.navbar-wrap {
    background: #16336e;
    border-radius: 10px;
    padding: 0 10px 8px 0;
    margin-bottom: 26px;
}
div[data-testid="column"] .stButton > button {
    background-color: transparent !important;
    color: rgba(255,255,255,0.75) !important;
    border: none !important;
    border-radius: 18px;
    font-weight: 500;
    font-size: 13px;
    box-shadow: none !important;
}
div[data-testid="column"] .stButton > button[kind="primary"] {
    background-color: rgba(255,255,255,0.12) !important;
    color: white !important;
    font-weight: 600;
}
div[data-testid="column"] .stButton > button:hover {
    color: white !important;
}

.card {
    background: white;
    border-radius: 14px;
    padding: 26px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #eef0f3;
}
.section-title { font-size: 17px; font-weight: 700; color: #111827; margin-bottom: 2px; }
.section-sub { font-size: 12px; color: #6b7280; margin-bottom: 18px; }

.placeholder-box {
    text-align: center;
    padding: 70px 20px;
    color: #6b7280;
}
.placeholder-icon {
    width: 56px; height: 56px; border-radius: 14px;
    background: #f3f4f6; display: flex; align-items: center; justify-content: center;
    margin: 0 auto 18px auto; font-size: 22px; color: #9ca3af;
}
.placeholder-title { font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 8px; }
.placeholder-text { font-size: 13px; color: #6b7280; max-width: 360px; margin: 0 auto 16px auto; }
.feature-list { font-size: 13px; color: #16a34a; text-align: left; display: inline-block; }
.feature-list div { margin-bottom: 4px; }

.metric-box {
    background: white;
    border: 1px solid #eef0f3;
    border-radius: 12px;
    padding: 16px 18px;
}
.metric-label { font-size: 11px; color: #9ca3af; letter-spacing: 0.5px; font-weight: 600; }
.metric-num { font-size: 26px; font-weight: 800; color: #111827; margin-top: 4px; }
.metric-sub { font-size: 11px; color: #9ca3af; }

.top-result {
    background: linear-gradient(135deg, #ede9fe, #f5f3ff);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    position: relative;
}
.top-result .tag { font-size: 12px; color: #7c3aed; font-weight: 700; }
.top-result .name { font-size: 28px; font-weight: 800; color: #6d28d9; }
.conf-badge {
    position: absolute; top: 18px; right: 18px;
    background: #7c3aed; color: white;
    border-radius: 16px; padding: 4px 12px;
    font-size: 12px; font-weight: 600;
}
.bar-row { margin-bottom: 12px; }
.bar-label { font-size: 13px; font-weight: 600; }
.bar-bg { background: #e5e7eb; border-radius: 8px; height: 9px; overflow: hidden; margin-top: 4px; }
.bar-fill { height: 100%; border-radius: 8px; }
.bar-pct { font-size: 12px; font-weight: 700; float: right; }

.history-row {
    background: white;
    border: 1px solid #eef0f3;
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.history-cat { font-weight: 700; font-size: 13px; width: 90px; }
.history-text { font-size: 13px; color: #374151; flex: 1; padding: 0 14px; }
.history-meta { font-size: 12px; color: #9ca3af; white-space: nowrap; }
.history-conf { font-weight: 700; font-size: 13px; margin-right: 14px; }
</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "Classifier"
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""

# ---------------- NAVBAR ----------------
st.markdown('<div class="navbar-wrap">', unsafe_allow_html=True)
nav_l, nav_r = st.columns([3, 2])
with nav_l:
    st.markdown("""
    <div class="navbar">
        <div class="navbar-left">
            <div class="navbar-logo">CK</div>
            <div>
                <div class="navbar-title">Cambodian News Classifier</div>
                <div class="navbar-sub">MULTI-CLASS AI ANALYSIS</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
with nav_r:
    st.write("")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📰 Classifier ●" if st.session_state.page == "Classifier" else "📰 Classifier",
                      use_container_width=True,
                      type="primary" if st.session_state.page == "Classifier" else "secondary"):
            st.session_state.page = "Classifier"
            st.rerun()
    with b2:
        if st.button("🕐 Session History ●" if st.session_state.page == "History" else "🕐 Session History",
                      use_container_width=True,
                      type="primary" if st.session_state.page == "History" else "secondary"):
            st.session_state.page = "History"
            st.rerun()
    with b3:
        if st.button("ℹ️ About ●" if st.session_state.page == "About" else "ℹ️ About",
                      use_container_width=True,
                      type="primary" if st.session_state.page == "About" else "secondary"):
            st.session_state.page = "About"
            st.rerun()
st.markdown('</div>', unsafe_allow_html=True)
st.write("")


@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="roberta_best.pt")
    model = NewsClassifier(num_labels=len(LABELS))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    return model, tokenizer


def classify(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)[0]
    return {LABELS[i]: probs[i].item() for i in range(len(LABELS))}


def extract_pdf_text(file):
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


model, tokenizer = load_model()

# =================== PAGE: CLASSIFIER ===================
if st.session_state.page == "Classifier":
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Input Section</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Paste news text for classification</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📄 Text Input", "⬆️ PDF Upload"])

        text = ""
        with tab1:
            st.caption("**Direct Text Entry**　·　_Perfect for copied articles or short texts_")
            text = st.text_area("", height=260, placeholder="Paste your news article here...\n\nThe government announced new economic policies today...",
                                 label_visibility="collapsed", key="text_input_area")

        with tab2:
            st.caption("**PDF Upload**　·　_Upload a PDF news article_")
            pdf_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
            if pdf_file is not None:
                try:
                    text = extract_pdf_text(pdf_file)
                    st.success(f"Extracted {len(text)} characters from PDF.")
                    with st.expander("Preview extracted text"):
                        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

        n_chars = len(text)
        n_words = len(text.split()) if text else 0
        st.caption(f"{n_chars:,} chars · {n_words} words")

        st.markdown('<div class="analyze-btn-wrap">', unsafe_allow_html=True)
        analyze = st.button("✨ Analyze Text", use_container_width=True, type="primary", key="analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)
        if analyze and text.strip():
            result = classify(text)
            st.session_state.result = result
            st.session_state.last_text = text
            top_label = max(result, key=result.get)
            st.session_state.history.insert(0, {
                "text": text.strip().replace("\n", " "),
                "category": top_label,
                "confidence": result[top_label],
                "time": datetime.datetime.now().strftime("%I:%M %p"),
            })
            st.rerun()
        elif analyze:
            st.warning("Please paste text or upload a PDF first.")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if st.session_state.result:
            text_used = st.session_state.last_text
            n_chars = len(text_used)
            n_words = len(text_used.split())
            sorted_results = sorted(st.session_state.result.items(), key=lambda x: -x[1])
            top_label, top_score = sorted_results[0]

            st.markdown(f"""
            <div class="top-result">
                <span class="tag">🏆 TOP CLASSIFICATION</span>
                <span class="conf-badge">{top_score*100:.1f}% confidence</span>
                <div class="name">{top_label}</div>
            </div>
            """, unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-num">{n_chars:,}</div><div class="metric-sub">Characters</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-num">{n_words}</div><div class="metric-sub">Words</div></div>', unsafe_allow_html=True)

            st.markdown('<div style="color:#16a34a; font-size:13px; margin-top:10px;">✅ Text length is optimal for classification</div>', unsafe_allow_html=True)

            st.write("")
            st.markdown("**📊 Confidence Scores**")
            for label, score in sorted_results:
                color = COLORS.get(label, "#6b7280")
                pct = score * 100
                st.markdown(f"""
                <div class="bar-row">
                    <span class="bar-label">{label}</span>
                    <span class="bar-pct" style="color:{color}">{pct:.1f}%</span>
                    <div class="bar-bg"><div class="bar-fill" style="width:{pct}%; background:{color};"></div></div>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            b1, b2 = st.columns(2)
            with b1:
                st.download_button("⬇️ Export", data=str(st.session_state.result), file_name="result.json", use_container_width=True)
            with b2:
                if st.button("🗑️ Clear", use_container_width=True):
                    st.session_state.result = None
                    st.rerun()
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-icon">📈</div>
                <div class="placeholder-title">Results Panel</div>
                <div class="placeholder-text">Enter a news article on the left and click <b>"Analyze Text"</b> to see classification results, confidence scores, and detailed analytics.</div>
                <div class="feature-list">
                    <div>✅ Supports news text input</div>
                    <div>✅ 5-category classification model</div>
                    <div>✅ Confidence scores for all categories</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =================== PAGE: SESSION HISTORY ===================
elif st.session_state.page == "History":
    st.markdown('<div class="section-title" style="font-size:22px;">Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All classification results from this session</div>', unsafe_allow_html=True)

    hist = st.session_state.history
    total = len(hist)
    cats_used = len(set(h["category"] for h in hist)) if hist else 0
    avg_conf = (sum(h["confidence"] for h in hist) / total * 100) if total else 0.0
    top_cat = max(set(h["category"] for h in hist), key=lambda c: sum(1 for h in hist if h["category"] == c)) if hist else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">TOTAL ARTICLES</div><div class="metric-num">{total}</div><div class="metric-sub">analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">CATEGORIES USED</div><div class="metric-num">{cats_used}/{len(LABELS)}</div><div class="metric-sub">of {len(LABELS)} total</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">AVG CONFIDENCE</div><div class="metric-num">{avg_conf:.1f}%</div><div class="metric-sub">across session</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">TOP CATEGORY</div><div class="metric-num" style="color:#7c3aed; font-size:20px;">{top_cat}</div><div class="metric-sub">most frequent</div></div>', unsafe_allow_html=True)

    st.write("")
    s1, s2, s3, s4 = st.columns([3, 1, 1, 1])
    with s1:
        search = st.text_input("", placeholder="🔍 Search articles...", label_visibility="collapsed")
    with s2:
        cat_filter = st.selectbox("", ["All"] + LABELS, label_visibility="collapsed")
    with s3:
        st.selectbox("", ["All"], label_visibility="collapsed")
    with s4:
        sort_order = st.selectbox("", ["Most Recent", "Oldest First", "Highest Confidence"], label_visibility="collapsed")

    filtered = hist
    if search:
        filtered = [h for h in filtered if search.lower() in h["text"].lower()]
    if cat_filter != "All":
        filtered = [h for h in filtered if h["category"] == cat_filter]
    if sort_order == "Oldest First":
        filtered = list(reversed(filtered))
    elif sort_order == "Highest Confidence":
        filtered = sorted(filtered, key=lambda h: -h["confidence"])

    top_row = st.columns([1, 1])
    with top_row[0]:
        st.caption(f"{len(filtered)} article{'s' if len(filtered) != 1 else ''}")
    with top_row[1]:
        cc1, cc2 = st.columns(2)
        with cc1:
            st.button("⬇️ Export All", use_container_width=True)
        with cc2:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.rerun()

    if not filtered:
        st.info("No articles analyzed yet. Go to the Classifier tab to get started.")
    for h in filtered:
        color = COLORS.get(h["category"], "#6b7280")
        snippet = h["text"][:90] + ("..." if len(h["text"]) > 90 else "")
        st.markdown(f"""
        <div class="history-row">
            <div class="history-cat" style="color:{color}">{h['category']}</div>
            <div class="history-text">{snippet}</div>
            <div class="history-conf" style="color:{color}">{h['confidence']*100:.1f}%</div>
            <div class="history-meta">🕐 {h['time']}</div>
        </div>
        """, unsafe_allow_html=True)

# =================== PAGE: ABOUT ===================
elif st.session_state.page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">About this app</div>', unsafe_allow_html=True)
    st.write(
        "This tool classifies Cambodian news articles into one of 5 categories "
        f"({', '.join(LABELS)}) using a fine-tuned RoBERTa model. "
        "Paste text directly or upload a PDF, and the model returns confidence scores across all categories."
    )
    st.write("Model hosted on Hugging Face Hub:", REPO_ID)
    st.markdown('</div>', unsafe_allow_html=True)
