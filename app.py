import os
import streamlit as st
import numpy as np
import pickle
import re
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import onnxruntime as ort

# ============================================================
# KONFIGURASI PATH
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_GRU_cfg1_seq100.h5")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_gru.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN = 100

st.set_page_config(page_title="IMDb: Ratings, Reviews, and Where to Watch", page_icon="🎬", layout="wide")

# ============================================================
# CUSTOM CSS: IMDB THEME
# =============================================
st.markdown("""
<style>
    /* Reset & General Body */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
        font-family: Roboto, Helvetica, Arial, sans-serif;
    }

    /* Navbar Mockup */
    .imdb-navbar {
        background-color: #121212;
        padding: 10px 20px;
        display: flex;
        align-items: center;
        border-bottom: 1px solid #333;
        margin-bottom: 20px;
    }
    .imdb-logo {
        background-color: #F5C518;
        color: #000000;
        font-weight: 900;
        font-size: 24px;
        padding: 2px 8px;
        border-radius: 4px;
        letter-spacing: -1px;
        margin-right: 20px;
    }
    .imdb-menu-icon {
        color: #FFFFFF;
        font-weight: bold;
        margin-right: 20px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    .imdb-search-bar {
        flex-grow: 1;
        background-color: #FFFFFF;
        border-radius: 4px;
        display: flex;
        align-items: center;
        padding: 4px 10px;
        color: #000;
        margin-right: 20px;
    }
    .imdb-search-input {
        color: #777;
        font-size: 14px;
    }
    .imdb-nav-links {
        display: flex;
        gap: 20px;
        font-weight: bold;
        font-size: 14px;
        align-items: center;
    }

    /* Headers */
    h1, h2, h3 { color: #F5C518 !important; }
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 4px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #F5C518 !important;
        color: #000000 !important;
        font-weight: bold;
        border-radius: 4px;
        border: none;
        padding: 8px 20px;
    }
    .stButton>button:hover {
        background-color: #E2B616 !important;
    }

    /* Radio buttons */
    div[role="radiogroup"] label {
        border: 1px solid #333;
        border-radius: 4px;
        padding: 5px 10px;
        margin-right: 10px;
        background-color: #121212;
    }

    /* Info Boxes & Metric Cards */
    .metric-container {
        background-color: #121212;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #333;
    }
    [data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 24px !important; }
    [data-testid="stMetricLabel"] { color: #AAAAAA !important; }
    [data-testid="stMetricDelta"] { color: #F5C518 !important; }

    /* Results */
    .result-score {
        font-size: 48px;
        font-weight: bold;
        color: #F5C518;
    }
    .result-star {
        color: #F5C518;
        font-size: 48px;
    }
    .result-out-of {
        color: #AAAAAA;
        font-size: 24px;
    }
    .result-box {
        background-color: #121212;
        padding: 20px;
        border-radius: 8px;
        border-left: 5px solid #F5C518;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# NAVBAR MOCKUP
# ============================================================
st.markdown("""
<div class="imdb-navbar">
    <div class="imdb-logo">IMDb</div>
    <div class="imdb-menu-icon">☰ Menu</div>
    <div class="imdb-search-bar">
        <span class="imdb-search-input">All ▾ Search IMDb...</span>
    </div>
    <div class="imdb-nav-links">
        <span>IMDbPro</span>
        <span>+ Watchlist</span>
        <span>👤 Sign In</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ============================================================
# LOAD MODEL & UTILS
# ============================================================
@st.cache_resource
def load_keras_model():
    return load_model(H5_MODEL_PATH)

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession(ONNX_MODEL_PATH)

@st.cache_resource
def get_tokenizer():
    with open(TOKENIZER_PATH, 'rb') as f:
        return pickle.load(f)

def preprocess_text(text, tokenizer):
    text = text.lower()
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

def predict_onnx(session, processed):
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: processed.astype(np.float32)})
    return result[0][0][0]

if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# MAIN LAYOUT
# ============================================================
st.markdown("<h1>Up next: AI Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #AAAAAA;'>V3 Final Project • Powered by GRU Deep Learning</p>", unsafe_allow_html=True)

# Layout with Columns to mimic the "Up next" right sidebar look
col_main, col_side = st.columns([7, 3])

with col_main:
    # -------------------------
    # INPUT SECTION
    # -------------------------
    st.markdown("<h3>Submit your review</h3>", unsafe_allow_html=True)
    
    model_choice = st.radio(
        "Select Engine",
        ["Model asli (.h5)", "Model teroptimasi (.onnx)"],
        horizontal=True,
    )
    
    user_input = st.text_area(
        "",
        "The cinematography was breathtaking, and the acting was top-notch. Easily one of the best movies of the year!",
        height=120
    )

    if st.button("Rate Review"):
        if user_input.strip() == "":
            st.warning("Review cannot be empty.")
        else:
            with st.spinner("Analyzing semantics..."):
                tokenizer = get_tokenizer()
                processed = preprocess_text(user_input, tokenizer)
                start = time.time()

                if model_choice.startswith("Model asli"):
                    keras_model = load_keras_model()
                    pred = keras_model.predict(processed, verbose=0)[0][0]
                    model_label = "H5"
                else:
                    onnx_session = load_onnx_model()
                    pred = predict_onnx(onnx_session, processed)
                    model_label = "ONNX"

                elapsed = time.time() - start

            prob_positive = float(pred)
            prob_negative = 1 - prob_positive
            
            # IMDB Style Score (1-10)
            score_10 = round(prob_positive * 10, 1) if prob_positive > 0.5 else round((prob_negative * 10), 1)
            pred_class = "Positive" if prob_positive > 0.5 else "Negative"

            st.markdown("<br><h3>Review Verdict</h3>", unsafe_allow_html=True)
            
            res_col1, res_col2 = st.columns([1, 3])
            with res_col1:
                st.markdown(f"""
                <div>
                    <span class="result-star">★</span>
                    <span class="result-score">{score_10}</span><span class="result-out-of">/10</span>
                </div>
                """, unsafe_allow_html=True)
                
            with res_col2:
                border_color = "#F5C518" if pred_class == "Positive" else "#E50914"
                st.markdown(f"""
                <div style="background-color: #121212; padding: 15px; border-radius: 4px; border-left: 4px solid {border_color};">
                    <h4 style="color: #FFF; margin-top:0;">{pred_class.upper()} SENTIMENT</h4>
                    <p style="color: #AAA; margin-bottom: 0;">Inference Time: {elapsed * 1000:.1f}ms via {model_label} engine</p>
                </div>
                """, unsafe_allow_html=True)

            st.write("")
            st.write(f"**Confidence Breakdown:**")
            st.progress(prob_positive)
            st.caption(f"Positive: {prob_positive * 100:.1f}%")
            
            st.progress(prob_negative)
            st.caption(f"Negative: {prob_negative * 100:.1f}%")

            st.session_state.history.append({
                "Review (Snippet)": user_input[:40] + "...",
                "Rating": f"★ {score_10}/10",
                "Sentiment": pred_class,
                "Model": model_label
            })

with col_side:
    # -------------------------
    # RIGHT SIDEBAR (Info & History)
    # -------------------------
    st.markdown("<h3>More to explore</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #F5C518; margin-top:0;'>Optimization Data</h4>", unsafe_allow_html=True)
    
    if os.path.exists(H5_MODEL_PATH) and os.path.exists(ONNX_MODEL_PATH):
        h5_mb = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
        onnx_mb = os.path.getsize(ONNX_MODEL_PATH) / (1024 * 1024)
        saving = (1 - onnx_mb / h5_mb) * 100
        
        st.metric("Base (.h5)", f"{h5_mb:.2f} MB")
        st.metric("Optimized (.onnx)", f"{onnx_mb:.2f} MB")
        st.metric("Size Reduction", f"{saving:.1f}%")
    else:
        st.info("Metrics will appear once models are loaded.")
    
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><h3>Recent Reviews</h3>", unsafe_allow_html=True)
    if st.session_state.history:
        for idx, item in enumerate(reversed(st.session_state.history[-5:])): # Show last 5
            color = "#F5C518" if item['Sentiment'] == "Positive" else "#E50914"
            st.markdown(f"""
            <div style="margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                <span style="color: {color}; font-weight: bold;">{item['Rating']}</span> - {item['Sentiment']}
                <br>
                <span style="color: #AAA; font-size: 12px;">{item['Review (Snippet)']}</span>
                <br>
                <span style="color: #777; font-size: 10px;">Engine: {item['Model']}</span>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("<p style='color: #777;'>No reviews processed yet.</p>", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #777; font-size: 12px;">
    An IMDb clone built with Streamlit • Week 5 Deployment Project
</div>
""", unsafe_allow_html=True)
