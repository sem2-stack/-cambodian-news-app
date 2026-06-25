import streamlit as st
import torch
import datetime
import json
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
ICONS = {
    "Politics": "🏛️",
    "Technology": "💻",
    "Economics": "📈",
    "Health": "🏥",
    "Sports": "⚽",
}

st.set_page_config(page_title="Cambodian News Classifier", layout="wide", page_icon="📰")

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #f0f2f7 !important; }
.block-container { padding-top: 0 !important; max-width: 1240px; }
header[data-testid="stHeader"] { background: transparent; height: 0; }
div[data-testid="stToolbar"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* ── Navbar ── */
.navbar {
    background: linear-gradient(135deg, #0f2554 0%, #1e4db7 100%);
    padding: 0 32px;
    display: flex;
    align-items: center;
    height: 64px;
    margin-bottom: 0;
    box-shadow: 0 2px 12px rgba(15,37,84,0.2);
}
.navbar-brand { display: flex; align-items: center; gap: 14px; }
.navbar-logo {
    width: 38px; height: 38px; border-radius: 10px;
    background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; border: 1px solid rgba(255,255,255,0.2);
}
.navbar-title { font-size: 15px; font-weight: 700; color: white; line-height: 1.2; }
.navbar-sub { font-size: 10px; color: rgba(255,255,255,0.5); letter-spacing: 1px; text-transform: uppercase; }

/* ── Nav buttons ── */
div[data-testid="column"] .stButton > button {
    background: transparent !important;
    color: rgba(255,255,255,0.65) !important;
    border: none !important;
    border-radius: 22px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 18px !important;
    box-shadow: none !important;
    transition: all 0.18s !important;
}
div[data-testid="column"] .stButton > button[kind="primary"] {
    background: rgba(255,255,255,0.15) !important;
    color: white !important;
    font-weight: 600 !important;
}
div[data-testid="column"] .stButton > button:hover {
    background: rgba(255,255,255,0.1) !important;
    color: white !important;
}

/* ── Cards ── */
.card {
    background: white;
    border-radius: 16px;
    padding: 28px;
    border: 1px solid #e8eaf0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.03);
}
.card-title { font-size: 16px; font-weight: 700; color: #111827; margin-bottom: 2px; }
.card-sub { font-size: 12px; color: #9ca3af; margin-bottom: 20px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6 !important;
    border-radius: 10px !important;
    padding: 3px !important;
    gap: 2px !important;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 6px 18px !important;
    color: #6b7280 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    background: white !important;
    color: #111827 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.stTabs [data-baseweb="tab-border"] { display: none !important; }

/* ── Inputs ── */
.stTextArea textarea {
    background: #f9fafb !important;
    color: #111827 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    resize: none !important;
}
.stTextArea textarea:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}
.stTextInput input {
    background: #f9fafb !important;
    color: #111827 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-size: 13px !important;
}

/* ── Analyze button ── */
.analyze-btn-wrap .stButton > button {
    background: linear-gradient(135deg, #1e4db7 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    font-weight: 700 !important;
    padding: 12px 0 !important;
    box-shadow: 0 4px 12px rgba(37,99,235,0.3) !important;
    transition: all 0.2s !important;
}
.analyze-btn-wrap .stButton > button:hover {
    box-shadow: 0 6px 18px rgba(37,99,235,0.4) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric boxes ── */
.metric-box {
    background: #f9fafb;
    border: 1px solid #eef0f3;
    border-radius: 12px;
    padding: 14px 18px;
}
.metric-label { font-size: 11px; color: #9ca3af; letter-spacing: 0.7px; font-weight: 600; text-transform: uppercase; }
.metric-num { font-size: 26px; font-weight: 800; color: #111827; margin-top: 4px; line-height: 1; }
.metric-sub { font-size: 11px; color: #9ca3af; margin-top: 2px; }

/* ── Top result banner ── */
.top-result {
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}
.top-result .tag { font-size: 11px; font-weight: 700; letter-spacing: 0.8px; text-transform: uppercase; color: rgba(255,255,255,0.8); }
.top-result .result-icon { font-size: 30px; margin: 6px 0 2px; display: block; }
.top-result .name { font-size: 30px; font-weight: 800; color: white; line-height: 1; }
.conf-badge {
    position: absolute; top: 20px; right: 20px;
    background: rgba(0,0,0,0.15);
    color: white;
    border-radius: 20px; padding: 5px 14px;
    font-size: 12px; font-weight: 700;
    border: 1px solid rgba(255,255,255,0.25);
}

/* ── Confidence bars ── */
.bar-row { margin-bottom: 14px; }
.bar-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }
.bar-label { font-size: 13px; font-weight: 600; color: #374151; }
.bar-pct { font-size: 12px; font-weight: 700; }
.bar-bg { background: #f3f4f6; border-radius: 8px; height: 8px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 8px; }

/* ── Placeholder ── */
.placeholder-box { text-align: center; padding: 60px 20px; }
.placeholder-icon {
    width: 64px; height: 64px; border-radius: 16px;
    background: linear-gradient(135deg, #ede9fe, #dbeafe);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 20px; font-size: 28px;
}
.placeholder-title { font-size: 17px; font-weight: 700; color: #111827; margin-bottom: 10px; }
.placeholder-text { font-size: 13px; color: #6b7280; max-width: 340px; margin: 0 auto 20px; line-height: 1.6; }
.feature-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #16a34a; font-size: 12px; font-weight: 500;
    border-radius: 20px; padding: 4px 12px; margin: 3px;
}

/* ── Section divider ── */
.section-divider {
    height: 1px;
    background: linear-gradient(to right, transparent, #e5e7eb, transparent);
    margin: 18px 0;
}

/* ── Success badge ── */
.success-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #f0fdf4; border: 1px solid #bbf7d0;
    color: #16a34a; font-size: 12px; font-weight: 600;
    border-radius: 20px; padding: 4px 12px; margin-top: 10px;
}

/* ── Action buttons ── */
.action-btn-wrap .stButton > button {
    background: white !important;
    color: #374151 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    padding: 9px 0 !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
}
.action-btn-wrap .stButton > button:hover {
    background: #f9fafb !important;
    border-color: #d1d5db !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: white !important;
    color: #374151 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 10px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    width: 100% !important;
    box-shadow: none !important;
}

/* ── History rows ── */
.history-row {
    background: white;
    border: 1px solid #eef0f3;
    border-radius: 12px;
    padding: 14px 18px;
    display: flex;
    align-items: center;
    gap: 14px;
    margin-bottom: 10px;
}
.history-cat { font-weight: 700; font-size: 13px; min-width: 96px; }
.history-text { font-size: 13px; color: #6b7280; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.history-conf { font-weight: 700; font-size: 13px; flex-shrink: 0; }
.history-time { font-size: 11px; color: #d1d5db; flex-shrink: 0; }

/* ── Page title ── */
.page-title { font-size: 22px; font-weight: 800; color: #111827; margin-bottom: 2px; }
.page-sub { font-size: 13px; color: #9ca3af; margin-bottom: 22px; }

/* ── About ── */
.about-label {
    font-size: 11px; font-weight: 700; color: #9ca3af;
    letter-spacing: 1px; text-transform: uppercase;
    margin: 20px 0 8px;
}
.cat-pill {
    display: inline-block;
    color: white; border-radius: 20px;
    padding: 4px 14px; font-size: 13px; font-weight: 600;
    margin: 3px;
}
.warn-box {
    background: #fffbeb; border: 1px solid #fde68a;
    border-radius: 10px; padding: 14px 16px;
    font-size: 13px; color: #92400e; line-height: 1.7;
    margin-top: 8px;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
for key, default in [("page", "Classifier"), ("history", []), ("result", None), ("last_text", "")]:
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <div class="navbar-logo">📰</div>
        <div>
            <div class="navbar-title">Cambodian News Classifier</div>
            <div class="navbar-sub">Multi-class AI Analysis</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

_, nav_r = st.columns([3, 2])
with nav_r:
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📰 Classifier", use_container_width=True,
                     type="primary" if st.session_state.page == "Classifier" else "secondary"):
            st.session_state.page = "Classifier"; st.rerun()
    with b2:
        if st.button("🕐 History", use_container_width=True,
                     type="primary" if st.session_state.page == "History" else "secondary"):
            st.session_state.page = "History"; st.rerun()
    with b3:
        if st.button("ℹ️ About", use_container_width=True,
                     type="primary" if st.session_state.page == "About" else "secondary"):
            st.session_state.page = "About"; st.rerun()

st.write("")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "Classifier":
    col1, col2 = st.columns(2, gap="medium")

    # ── LEFT ─────────────────────────────────────────────────────────────────
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Input Section</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Paste or upload a news article for classification</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📄  Text Input", "⬆️  PDF Upload"])
        text = ""

        with tab1:
            st.caption("**Direct Text Entry** · _Paste any news article or headline_")
            text = st.text_area(
                "", height=280,
                placeholder="Paste your news article here...\n\nThe government announced new economic policies today...",
                label_visibility="collapsed", key="text_input_area"
            )

        with tab2:
            st.caption("**PDF Upload** · _Upload a scanned or digital PDF article_")
            pdf_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
            if pdf_file is not None:
                try:
                    text = extract_pdf_text(pdf_file)
                    st.success(f"✅ Extracted {len(text):,} characters from PDF.")
                    with st.expander("Preview extracted text"):
                        st.write(text[:1000] + ("…" if len(text) > 1000 else ""))
                except Exception as e:
                    st.error(f"Could not read PDF: {e}")

        n_chars = len(text)
        n_words = len(text.split()) if text.strip() else 0
        st.caption(f"**{n_chars:,}** chars · **{n_words}** words")

        st.markdown('<div class="analyze-btn-wrap">', unsafe_allow_html=True)
        analyze = st.button("✨  Analyze Text", use_container_width=True, type="primary", key="analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)

        if analyze and text.strip():
            with st.spinner("Running classification…"):
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
            st.warning("⚠️ Please paste some text or upload a PDF first.")

        st.markdown('</div>', unsafe_allow_html=True)

    # ── RIGHT ────────────────────────────────────────────────────────────────
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        if st.session_state.result:
            text_used = st.session_state.last_text
            sorted_results = sorted(st.session_state.result.items(), key=lambda x: -x[1])
            top_label, top_score = sorted_results[0]
            top_color = COLORS[top_label]
            top_icon = ICONS[top_label]

            # Top banner
            st.markdown(f"""
            <div class="top-result" style="background:{top_color};">
                <div class="tag">🏆 Top Classification</div>
                <span class="result-icon">{top_icon}</span>
                <div class="name">{top_label}</div>
                <div class="conf-badge">{top_score*100:.1f}% confidence</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Characters</div><div class="metric-num">{len(text_used):,}</div></div>', unsafe_allow_html=True)
            with m2:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Words</div><div class="metric-num">{len(text_used.split())}</div></div>', unsafe_allow_html=True)
            with m3:
                st.markdown(f'<div class="metric-box"><div class="metric-label">Category</div><div class="metric-num" style="font-size:16px;color:{top_color};">{top_icon} {top_label}</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="success-badge">✅ Classification complete</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Confidence bars
            st.markdown('<div style="font-size:14px;font-weight:700;color:#111827;margin-bottom:14px;">📊 Confidence Scores</div>', unsafe_allow_html=True)
            bars_html = ""
            for label, score in sorted_results:
                color = COLORS[label]
                icon = ICONS[label]
                pct = score * 100
                bars_html += f"""
                <div class="bar-row">
                    <div class="bar-header">
                        <span class="bar-label">{icon} {label}</span>
                        <span class="bar-pct" style="color:{color};">{pct:.1f}%</span>
                    </div>
                    <div class="bar-bg">
                        <div class="bar-fill" style="width:{pct}%;background:{color};"></div>
                    </div>
                </div>"""
            st.markdown(bars_html, unsafe_allow_html=True)

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

            # Actions
            b1, b2 = st.columns(2)
            with b1:
                export_data = json.dumps({"label": top_label, "scores": st.session_state.result}, indent=2)
                st.download_button(
                    "⬇️ Export JSON",
                    data=export_data,
                    file_name="classification_result.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with b2:
                st.markdown('<div class="action-btn-wrap">', unsafe_allow_html=True)
                if st.button("🗑️ Clear Result", use_container_width=True):
                    st.session_state.result = None
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-icon">📊</div>
                <div class="placeholder-title">Results will appear here</div>
                <div class="placeholder-text">
                    Paste a news article on the left and click <b>Analyze Text</b>
                    to see classification results and confidence scores.
                </div>
                <div>
                    <span class="feature-pill">✅ News text input</span>
                    <span class="feature-pill">✅ PDF upload</span>
                </div>
                <div style="margin-top:6px;">
                    <span class="feature-pill">✅ 5-category model</span>
                    <span class="feature-pill">✅ Confidence scores</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: SESSION HISTORY
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "History":
    st.markdown('<div class="page-title">Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">All classification results from this session</div>', unsafe_allow_html=True)

    hist = st.session_state.history
    total = len(hist)
    cats_used = len(set(h["category"] for h in hist)) if hist else 0
    avg_conf = (sum(h["confidence"] for h in hist) / total * 100) if total else 0.0
    cat_counts = {}
    for h in hist:
        cat_counts[h["category"]] = cat_counts.get(h["category"], 0) + 1
    top_cat = max(cat_counts, key=cat_counts.get) if cat_counts else "—"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Total Articles</div><div class="metric-num">{total}</div><div class="metric-sub">analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Categories Used</div><div class="metric-num">{cats_used}/{len(LABELS)}</div><div class="metric-sub">of {len(LABELS)} total</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">Avg Confidence</div><div class="metric-num">{avg_conf:.1f}%</div><div class="metric-sub">across session</div></div>', unsafe_allow_html=True)
    with c4:
        tc = COLORS.get(top_cat, "#7c3aed")
        ti = ICONS.get(top_cat, "")
        st.markdown(f'<div class="metric-box"><div class="metric-label">Top Category</div><div class="metric-num" style="color:{tc};font-size:18px;">{ti} {top_cat}</div></div>', unsafe_allow_html=True)

    st.write("")

    s1, s2, s3, s4 = st.columns([3, 1.2, 1.5, 1.2])
    with s1:
        search = st.text_input("", placeholder="🔍 Search articles…", label_visibility="collapsed")
    with s2:
        cat_filter = st.selectbox("", ["All"] + LABELS, label_visibility="collapsed")
    with s3:
        sort_order = st.selectbox("", ["Most Recent", "Oldest First", "Highest Confidence"], label_visibility="collapsed")
    with s4:
        st.markdown('<div class="action-btn-wrap">', unsafe_allow_html=True)
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.history = []
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    filtered = hist[:]
    if search:
        filtered = [h for h in filtered if search.lower() in h["text"].lower()]
    if cat_filter != "All":
        filtered = [h for h in filtered if h["category"] == cat_filter]
    if sort_order == "Oldest First":
        filtered = list(reversed(filtered))
    elif sort_order == "Highest Confidence":
        filtered = sorted(filtered, key=lambda h: -h["confidence"])

    st.caption(f"**{len(filtered)}** article{'s' if len(filtered) != 1 else ''}")

    if not filtered:
        st.info("No articles analyzed yet. Go to the Classifier tab to get started.")
    else:
        for h in filtered:
            color = COLORS.get(h["category"], "#6b7280")
            icon = ICONS.get(h["category"], "")
            snippet = h["text"][:100] + ("…" if len(h["text"]) > 100 else "")
            conf_pct = h["confidence"] * 100
            st.markdown(f"""
            <div class="history-row">
                <span class="history-cat" style="color:{color};">{icon} {h['category']}</span>
                <span class="history-text">{snippet}</span>
                <span class="history-conf" style="color:{color};">{conf_pct:.1f}%</span>
                <span class="history-time">🕐 {h['time']}</span>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "About":
    col1, col2 = st.columns([2, 1], gap="medium")

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">About this App</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Cambodian News Classifier · Multi-class AI Analysis</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <p style="font-size:14px;color:#374151;line-height:1.8;">
            This tool classifies Cambodian news articles into one of <b>5 categories</b>
            using a fine-tuned <b>RoBERTa</b> model. Paste text directly or upload a PDF,
            and the model returns confidence scores across all categories.
        </p>
        <div class="about-label">Hugging Face Model</div>
        <code style="font-size:13px;background:#f3f4f6;padding:5px 12px;border-radius:8px;color:#374151;display:inline-block;">{REPO_ID}</code>
        <div class="about-label">Categories</div>
        <div style="margin-top:4px;">
        """, unsafe_allow_html=True)
        for label in LABELS:
            st.markdown(f'<span class="cat-pill" style="background:{COLORS[label]};">{ICONS[label]} {label}</span>', unsafe_allow_html=True)
        st.markdown("""
        </div>
        <div class="about-label">⚠️ Label Mismatch Fix</div>
        <div class="warn-box">
            If your app shows wrong categories (e.g. Economics → Health), the
            <code>LABELS</code> list order in <code>app.py</code> must exactly match
            the label order used during model training. A single position mismatch
            shifts all predictions to wrong classes.
        </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">Session Stats</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-sub">Current session</div>', unsafe_allow_html=True)
        hist = st.session_state.history
        total = len(hist)
        avg_conf = (sum(h["confidence"] for h in hist) / total * 100) if total else 0.0
        st.markdown(f"""
        <div class="metric-box" style="margin-bottom:10px;">
            <div class="metric-label">Articles Analyzed</div>
            <div class="metric-num">{total}</div>
        </div>
        <div class="metric-box" style="margin-bottom:10px;">
            <div class="metric-label">Avg Confidence</div>
            <div class="metric-num">{avg_conf:.1f}%</div>
        </div>
        <div class="metric-box">
            <div class="metric-label">Model</div>
            <div style="font-size:13px;font-weight:600;color:#111827;margin-top:4px;">RoBERTa</div>
            <div class="metric-sub">Fine-tuned · HF Hub</div>
        </div>
        </div>
        """, unsafe_allow_html=True)
