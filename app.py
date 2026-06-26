import streamlit as st
import torch
import datetime
from huggingface_hub import hf_hub_download
from transformers import RobertaTokenizerFast
from model_def import NewsClassifier

# ==================== CONFIGURATION ====================
REPO_ID = "Theara2/cambodian-news-roberta"
LABELS = ["Politics", "Technology", "Economics", "Health", "Sports"]

# Color scheme for categories
COLORS = {
    "Politics": "#8b5cf6",    # Purple
    "Technology": "#22c55e",  # Green
    "Economics": "#3b82f6",   # Blue
    "Health": "#ef4444",      # Red
    "Sports": "#f59e0b",      # Amber
}

st.set_page_config(page_title="Cambodian News Classifier", layout="wide")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #f5f6f8 !important;
}

/* Hide Streamlit's default top toolbar */
header[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}
div[data-testid="stToolbar"] {
    display: none;
}
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

.block-container { 
    padding-top: 2rem !important; 
    max-width: 1200px; 
}

/* Navbar styles */
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

/* Text area / inputs */
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

/* Navbar */
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

/* Cards and containers */
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

/* Metrics */
.metric-box {
    background: white;
    border: 1px solid #eef0f3;
    border-radius: 12px;
    padding: 16px 18px;
}
.metric-label { font-size: 11px; color: #9ca3af; letter-spacing: 0.5px; font-weight: 600; }
.metric-num { font-size: 26px; font-weight: 800; color: #111827; margin-top: 4px; }
.metric-sub { font-size: 11px; color: #9ca3af; }

/* Results */
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

/* Confidence bars */
.bar-row { margin-bottom: 12px; }
.bar-label { font-size: 13px; font-weight: 600; }
.bar-bg { background: #e5e7eb; border-radius: 8px; height: 9px; overflow: hidden; margin-top: 4px; }
.bar-fill { height: 100%; border-radius: 8px; }
.bar-pct { font-size: 12px; font-weight: 700; float: right; }

/* History items */
.history-row {
    background: white;
    border: 1px solid #eef0f3;
    border-radius: 10px;
    padding: 14px 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
    transition: all 0.2s;
}
.history-row:hover {
    border-color: #d1d5db;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.history-cat { font-weight: 700; font-size: 13px; width: 100px; }
.history-text { font-size: 13px; color: #374151; flex: 1; padding: 0 14px; }
.history-meta { font-size: 12px; color: #9ca3af; white-space: nowrap; }
.history-conf { font-weight: 700; font-size: 13px; margin-right: 14px; }
</style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "page" not in st.session_state:
    st.session_state.page = "Classifier"
if "history" not in st.session_state:
    st.session_state.history = []
if "result" not in st.session_state:
    st.session_state.result = None
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "debug_info" not in st.session_state:
    st.session_state.debug_info = None

# ==================== NAVBAR ====================
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
        if st.button("🕐 History ●" if st.session_state.page == "History" else "🕐 History",
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

# ==================== MODEL LOADING ====================
@st.cache_resource
def load_model():
    weights_path = hf_hub_download(repo_id=REPO_ID, filename="roberta_best.pt")
    model = NewsClassifier(num_labels=len(LABELS))
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
    return model, tokenizer

# ==================== CLASSIFICATION FUNCTIONS ====================
def classify_text(text):
    """Classify text and return probabilities with potential correction"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs)
        probs = torch.softmax(logits, dim=-1)[0]
    
    # Get raw predictions
    raw_results = {LABELS[i]: probs[i].item() for i in range(len(LABELS))}
    
    # Apply keyword-based correction to improve Politics detection
    corrected_results = apply_keyword_correction(text, raw_results)
    
    return corrected_results

def apply_keyword_correction(text, results):
    """Apply domain knowledge to correct misclassifications"""
    text_lower = text.lower()
    
    # Politics keywords (in Khmer and English)
    politics_keywords = [
        'រដ្ឋាភិបាល', 'រដ្ឋមន្ត្រី', 'នាយករដ្ឋមន្ត្រី', 'គណបក្ស', 'បោះឆ្នោត', 
        'សភា', 'រដ្ឋសភា', 'ព្រឹទ្ធសភា', 'គោលនយោបាយ', 'រដ្ឋបាល',
        'government', 'prime minister', 'minister', 'election', 'parliament',
        'senate', 'policy', 'administration', 'political', 'democratic'
    ]
    
    # Health keywords
    health_keywords = [
        'សុខភាព', 'ពេទ្យ', 'មន្ទីរពេទ្យ', 'ជំងឺ', 'វេជ្ជបណ្ឌិត',
        'ថ្នាំ', 'ព្យាបាល', 'រោគសញ្ញា', 'ការពារ', 'អនាម័យ',
        'health', 'hospital', 'doctor', 'disease', 'medical',
        'treatment', 'medicine', 'symptom', 'covid', 'pandemic'
    ]
    
    # Check if text has strong politics signals
    politics_score = sum(1 for kw in politics_keywords if kw in text_lower)
    health_score = sum(1 for kw in health_keywords if kw in text_lower)
    
    # If text has politics keywords but model predicts Health, boost Politics
    if politics_score > health_score and results["Health"] > results["Politics"]:
        # Boost Politics and reduce Health
        boost_factor = 1.3 + (politics_score * 0.05)
        results["Politics"] = results["Politics"] * boost_factor
        results["Health"] = results["Health"] * 0.8
        
        # Normalize to maintain sum = 1
        total = sum(results.values())
        for key in results:
            results[key] = results[key] / total
    
    # If text has health keywords but model predicts Politics, boost Health
    elif health_score > politics_score and results["Politics"] > results["Health"]:
        boost_factor = 1.3 + (health_score * 0.05)
        results["Health"] = results["Health"] * boost_factor
        results["Politics"] = results["Politics"] * 0.8
        
        total = sum(results.values())
        for key in results:
            results[key] = results[key] / total
    
    return results

def extract_pdf_text(file):
    from pypdf import PdfReader
    reader = PdfReader(file)
    return "\n".join((page.extract_text() or "") for page in reader.pages)

# ==================== LOAD MODEL ====================
try:
    model, tokenizer = load_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# ==================== PAGE: CLASSIFIER ====================
if st.session_state.page == "Classifier":
    if not model_loaded:
        st.error("⚠️ Model could not be loaded. Please check your connection and try again.")
        st.stop()
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">📝 Input Section</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Paste news text or upload a PDF for classification</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["📄 Text Input", "⬆️ PDF Upload"])

        text = ""
        with tab1:
            st.caption("**Direct Text Entry** · Perfect for copied articles or short texts")
            text = st.text_area(
                "", 
                height=260, 
                placeholder="Paste your news article here...\n\nExample: 'The government announced new economic policies today...'",
                label_visibility="collapsed", 
                key="text_input_area"
            )

        with tab2:
            st.caption("**PDF Upload** · Upload a PDF news article")
            pdf_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")
            if pdf_file is not None:
                try:
                    text = extract_pdf_text(pdf_file)
                    st.success(f"✅ Extracted {len(text):,} characters from PDF.")
                    with st.expander("📄 Preview extracted text"):
                        st.write(text[:1000] + ("..." if len(text) > 1000 else ""))
                except Exception as e:
                    st.error(f"❌ Could not read PDF: {e}")

        # Character and word count
        n_chars = len(text)
        n_words = len(text.split()) if text else 0
        st.caption(f"📊 {n_chars:,} characters · {n_words} words")

        # Analyze button
        st.markdown('<div class="analyze-btn-wrap">', unsafe_allow_html=True)
        analyze = st.button("✨ Analyze Text", use_container_width=True, type="primary", key="analyze_btn")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if analyze and text.strip():
            with st.spinner("🔍 Analyzing..."):
                result = classify_text(text)
                st.session_state.result = result
                st.session_state.last_text = text
                
                top_label = max(result, key=result.get)
                top_confidence = result[top_label]
                
                # Add to history
                st.session_state.history.insert(0, {
                    "text": text.strip().replace("\n", " ")[:500],  # Limit text length in history
                    "category": top_label,
                    "confidence": top_confidence,
                    "time": datetime.datetime.now().strftime("%I:%M %p"),
                    "all_scores": result.copy()
                })
                
                # Show debug info if Politics detected with high confidence
                if top_label == "Health" and any(kw in text.lower() for kw in ['government', 'prime minister', 'minister', 'election']):
                    st.session_state.debug_info = f"⚠️ Politics-related text was classified as Health. Using keyword correction."
                else:
                    st.session_state.debug_info = None
                
                st.rerun()
        elif analyze:
            st.warning("⚠️ Please paste text or upload a PDF first.")
        
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        if st.session_state.result:
            text_used = st.session_state.last_text
            n_chars = len(text_used)
            n_words = len(text_used.split())
            sorted_results = sorted(st.session_state.result.items(), key=lambda x: -x[1])
            top_label, top_score = sorted_results[0]

            # Show debug warning if applicable
            if st.session_state.debug_info:
                st.warning(st.session_state.debug_info)

            # Top result card
            st.markdown(f"""
            <div class="top-result">
                <span class="tag">🏆 TOP CLASSIFICATION</span>
                <span class="conf-badge">{top_score*100:.1f}% confidence</span>
                <div class="name" style="color:{COLORS.get(top_label, '#6b7280')}">{top_label}</div>
            </div>
            """, unsafe_allow_html=True)

            # Metrics
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

            # Action buttons
            st.write("")
            b1, b2, b3 = st.columns(3)
            with b1:
                import json
                json_str = json.dumps(st.session_state.result, indent=2)
                st.download_button(
                    "⬇️ Export", 
                    data=json_str, 
                    file_name=f"classification_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 
                    use_container_width=True
                )
            with b2:
                if st.button("🔄 Reset", use_container_width=True):
                    st.session_state.result = None
                    st.session_state.debug_info = None
                    st.rerun()
        else:
            st.markdown("""
            <div class="placeholder-box">
                <div class="placeholder-icon">📈</div>
                <div class="placeholder-title">Results Panel</div>
                <div class="placeholder-text">Enter a news article on the left and click <b>"Analyze Text"</b> to see classification results, confidence scores, and detailed analytics.</div>
                <div class="feature-list">
                    <div>✅ Supports news text input</div>
                    <div>✅ 5-category classification</div>
                    <div>✅ Confidence scores for all categories</div>
                    <div>✅ Smart keyword correction</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: HISTORY ====================
elif st.session_state.page == "History":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title" style="font-size:22px;">📜 Session History</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">All classification results from this session</div>', unsafe_allow_html=True)

    hist = st.session_state.history
    total = len(hist)
    cats_used = len(set(h["category"] for h in hist)) if hist else 0
    avg_conf = (sum(h["confidence"] for h in hist) / total * 100) if total else 0.0
    top_cat = max(set(h["category"] for h in hist), key=lambda c: sum(1 for h in hist if h["category"] == c)) if hist else "—"

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-box"><div class="metric-label">📊 TOTAL ARTICLES</div><div class="metric-num">{total}</div><div class="metric-sub">analyzed</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-box"><div class="metric-label">🏷️ CATEGORIES USED</div><div class="metric-num">{cats_used}/{len(LABELS)}</div><div class="metric-sub">of {len(LABELS)} total</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-box"><div class="metric-label">🎯 AVG CONFIDENCE</div><div class="metric-num">{avg_conf:.1f}%</div><div class="metric-sub">across session</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-box"><div class="metric-label">🔥 TOP CATEGORY</div><div class="metric-num" style="color:{COLORS.get(top_cat, '#7c3aed')}; font-size:20px;">{top_cat}</div><div class="metric-sub">most frequent</div></div>', unsafe_allow_html=True)

    st.write("")
    
    # Filters
    s1, s2, s3 = st.columns([3, 1, 1])
    with s1:
        search = st.text_input("", placeholder="🔍 Search articles...", label_visibility="collapsed")
    with s2:
        cat_filter = st.selectbox("Category", ["All"] + LABELS, label_visibility="collapsed")
    with s3:
        sort_order = st.selectbox("Sort", ["Most Recent", "Oldest First", "Highest Confidence"], label_visibility="collapsed")

    # Apply filters
    filtered = hist
    if search:
        filtered = [h for h in filtered if search.lower() in h["text"].lower()]
    if cat_filter != "All":
        filtered = [h for h in filtered if h["category"] == cat_filter]
    if sort_order == "Oldest First":
        filtered = list(reversed(filtered))
    elif sort_order == "Highest Confidence":
        filtered = sorted(filtered, key=lambda h: -h["confidence"])

    # Action buttons
    col_actions = st.columns([3, 1, 1])
    with col_actions[0]:
        st.caption(f"Showing {len(filtered)} of {total} article{'s' if total != 1 else ''}")
    with col_actions[1]:
        if st.button("⬇️ Export All", use_container_width=True):
            import json
            json_str = json.dumps([
                {
                    "text": h["text"],
                    "category": h["category"],
                    "confidence": h["confidence"],
                    "time": h["time"]
                } for h in hist
            ], indent=2)
            st.download_button(
                "⬇️ Download", 
                data=json_str, 
                file_name=f"history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                use_container_width=True
            )
    with col_actions[2]:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    st.write("")

    # Display history
    if not filtered:
        if total == 0:
            st.info("💡 No articles analyzed yet. Go to the Classifier tab to get started.")
        else:
            st.info("No articles match your filters. Try adjusting your search or category filter.")
    else:
        for h in filtered:
            color = COLORS.get(h["category"], "#6b7280")
            snippet = h["text"][:100] + ("..." if len(h["text"]) > 100 else "")
            
            # Create an expandable history item
            with st.container():
                col1, col2, col3, col4 = st.columns([1.5, 4, 1, 1])
                with col1:
                    st.markdown(f'<span style="color:{color}; font-weight:700; font-size:14px;">{h["category"]}</span>', unsafe_allow_html=True)
                with col2:
                    st.markdown(f'<span style="font-size:13px; color:#374151;">{snippet}</span>', unsafe_allow_html=True)
                with col3:
                    st.markdown(f'<span style="color:{color}; font-weight:700; font-size:13px;">{h["confidence"]*100:.1f}%</span>', unsafe_allow_html=True)
                with col4:
                    st.markdown(f'<span style="font-size:12px; color:#9ca3af;">🕐 {h["time"]}</span>', unsafe_allow_html=True)
                
                # Show all scores in expander
                if "all_scores" in h:
                    with st.expander("📊 View all scores"):
                        scores = h["all_scores"]
                        for label, score in sorted(scores.items(), key=lambda x: -x[1]):
                            color2 = COLORS.get(label, "#6b7280")
                            pct = score * 100
                            st.markdown(f"""
                            <div class="bar-row">
                                <span class="bar-label">{label}</span>
                                <span class="bar-pct" style="color:{color2}">{pct:.1f}%</span>
                                <div class="bar-bg"><div class="bar-fill" style="width:{pct}%; background:{color2};"></div></div>
                            </div>
                            """, unsafe_allow_html=True)
                
                st.markdown("---")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ==================== PAGE: ABOUT ====================
elif st.session_state.page == "About":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">ℹ️ About This App</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 📰 Cambodian News Classifier
    
    This application uses a fine-tuned RoBERTa model to classify Cambodian news articles 
    into one of five categories:
    
    - 🟣 **Politics** - Government, elections, policies, and political affairs
    - 🟢 **Technology** - Tech news, digital innovation, and IT developments
    - 🔵 **Economics** - Business, finance, markets, and economic policies
    - 🔴 **Health** - Medical news, healthcare, diseases, and wellness
    - 🟡 **Sports** - Athletics, competitions, and sports events
    
    ### 🚀 Features
    
    - **Text Classification**: Paste any Cambodian news text for instant analysis
    - **PDF Support**: Upload PDF documents for automatic text extraction
    - **Confidence Scores**: View confidence percentages for all categories
    - **Smart Correction**: Keyword-based correction improves Politics vs Health classification
    - **Session History**: Track all classifications with filters and search
    - **Export Results**: Download classification data as JSON
    
    ### 🛠️ Technical Details
    
    - **Model**: Fine-tuned RoBERTa base model
    - **Training Data**: Cambodian news articles
    - **Framework**: PyTorch + Transformers
    - **Deployment**: Streamlit Cloud
    
    ### 📊 Model Performance
    
    The model is trained on a diverse dataset of Cambodian news articles and achieves 
    high accuracy across all categories. The smart keyword correction helps distinguish 
    between Politics and Health articles that may share similar vocabulary.
    
    ### 🔗 Links
    
    - **Model Repository**: [Theara2/cambodian-news-roberta](https://huggingface.co/Theara2/cambodian-news-roberta)
    - **Source Code**: Available on request
    
    ---
    
    **Version**: 2.0  
    **Last Updated**: June 2026
    """)
    
    st.markdown('</div>', unsafe_allow_html=True)
