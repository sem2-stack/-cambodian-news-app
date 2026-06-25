import streamlit as st
import torch
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

# ---------- styling ----------
st.markdown("""
<style>
.header-bar {
    background: #1e3a8a;
    color: white;
    padding: 18px 28px;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 24px;
}
.header-bar h1 { font-size: 18px; margin: 0; }
.header-bar p { font-size: 12px; margin: 0; opacity: 0.75; }
.card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}
.metric-box {
    background: #f3f4f6;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.metric-box .num { font-size: 22px; font-weight: 700; color: #111827; }
.metric-box .label { font-size: 12px; color: #6b7280; }
.top-result {
    background: linear-gradient(135deg, #ede9fe, #f5f3ff);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}
.top-result .tag { font-size: 12px; color: #7c3aed; font-weight: 600; }
.top-result .name { font-size: 28px; font-weight: 800; color: #6d28d9; }
.conf-badge {
    background: #7c3aed;
    color: white;
    border-radius: 16px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 600;
    float: right;
}
.bar-row { margin-bottom: 10px; }
.bar-label { font-size: 13px; font-weight: 600; margin-bottom: 4px; }
.bar-bg { background: #e5e7eb; border-radius: 8px; height: 10px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 8px; }
.bar-pct { font-size: 12px; font-weight: 600; float: right; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="header-bar">
    <div>
        <h1>📰 Cambodian News Classifier</h1>
        <p>MULTI-CLASS AI ANALYSIS</p>
    </div>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="roberta_best.pt")
    model = NewsClassifier(num_labels=len(LABELS))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    return model, tokenizer


model, tokenizer = load_model()

if "result" not in st.session_state:
    st.session_state.result = None

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("##### 📥 Input Section")
    st.caption("Paste news text for classification")
    text = st.text_area("Direct Text Entry", height=260, label_visibility="visible",
                         placeholder="Paste your Khmer or English news text here...")
    n_chars = len(text)
    n_words = len(text.split())
    st.caption(f"{n_chars:,} chars · {n_words} words")

    analyze = st.button("✨ Analyze Text", use_container_width=True, type="primary")
    if analyze and text.strip():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs)
            probs = torch.softmax(logits, dim=-1)[0]
        st.session_state.result = {LABELS[i]: probs[i].item() for i in range(len(LABELS))}
    elif analyze:
        st.warning("Please paste some text first.")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.result:
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
            st.markdown(f'<div class="metric-box"><div class="num">{n_chars:,}</div><div class="label">Characters</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-box"><div class="num">{n_words}</div><div class="label">Words</div></div>', unsafe_allow_html=True)

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
            st.button("⬇️ Export", use_container_width=True)
        with b2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.result = None
                st.rerun()
    else:
        st.info("Paste text and click **Analyze Text** to see classification results here.")
    st.markdown('</div>', unsafe_allow_html=True)
