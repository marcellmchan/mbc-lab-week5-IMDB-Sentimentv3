import os
import time
import numpy as np
import pickle
import re
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import onnxruntime as ort

# ============================================================
# KONFIGURASI
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_GRU_cfg1_seq100.h5")
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "sentiment_gru.onnx")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")
MAX_LEN = 100

VERSION_HISTORY = [
    {"versi": "v1", "perubahan": "Model dasar GRU (.h5) untuk klasifikasi sentimen review film. Antarmuka standar: input teks, prediksi sentimen + confidence score."},
    {"versi": "v2", "perubahan": "Optimasi model (ONNX Runtime) untuk inferensi lebih cepat, ditambah pilihan mesin inferensi, probabilitas breakdown dua kelas, dan riwayat prediksi."},
    {"versi": "v3 (Final)", "perubahan": "Perombakan UI total menyerupai website IMDb asli, navigasi multi-halaman, fitur Model Deep Dive (arsitektur GRU, tokenizer, top keywords), dan halaman Trivia & Versions."},
]

st.set_page_config(page_title="IMDb: Ratings, Reviews, and Where to Watch", page_icon="🎬", layout="wide")

# ============================================================
# GAYA (CSS) - IMDB CLONE THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');
    
    /* Reset & General Body */
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
        font-family: 'Roboto', sans-serif;
    }
    
    /* Hide top padding */
    .block-container { padding-top: 3rem; }

    /* Navbar Mockup IMDb */
    .imdb-navbar {
        background-color: #121212;
        padding: 15px 25px;
        display: flex;
        align-items: center;
        margin-bottom: 30px;
        border-radius: 8px;
        border: 1px solid #333;
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
        height: 32px;
    }
    .imdb-search-input {
        color: #777;
        font-size: 14px;
        font-weight: 500;
    }
    .imdb-nav-links {
        display: flex;
        gap: 20px;
        font-weight: bold;
        font-size: 14px;
        align-items: center;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid #333; }
    [data-testid="stSidebar"] * { color: #CCCCCC !important; }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        background: transparent; border: none; padding: 0.4rem 0; margin-right: 0;
    }
    [data-testid="stSidebar"] input[type="radio"] { accent-color: #F5C518; }
    [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: #F5C518 !important; font-weight: 700; }

    /* Headers */
    h1, h2, h3 { color: #F5C518 !important; }
    .page-title { color: #FFF !important; font-size: 2.5rem; margin-bottom: 0.2rem;}
    
    /* Text Area */
    .stTextArea textarea {
        background-color: #1A1A1A !important;
        color: #FFFFFF !important;
        border: 1px solid #333333 !important;
        border-radius: 4px;
    }
    .stTextArea textarea:focus {
        border-color: #F5C518 !important;
        box-shadow: 0 0 0 1px #F5C518 !important;
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
    .stButton>button:hover { background-color: #E2B616 !important; }

    /* Radio buttons */
    div[role="radiogroup"] label {
        border: 1px solid #333;
        border-radius: 4px;
        padding: 5px 10px;
        margin-right: 10px;
        background-color: #121212;
    }
    .main input[type="radio"] { accent-color: #F5C518; }
    
    /* Metric Cards */
    .spec-card {
        background: #121212; border: 1px solid #333; border-radius: 4px; padding: 1rem 1.2rem; margin-bottom: 15px;
    }
    .spec-card .label { color: #AAA; font-size: 0.85rem; font-weight:bold; }
    .spec-card .value { color: #F5C518; font-size: 1.6rem; font-weight: 900; margin-top: 0.1rem; margin-bottom: 0.2rem;}

    /* Progress */
    .stProgress > div > div { background: #F5C518; }
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
        <span style="color: #57B5F9;">IMDbPro</span>
        <span><b style="font-size:18px;">+</b> Watchlist</span>
        <span>Sign In</span>
        <span>EN ▾</span>
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

def predict_keras(processed):
    return float(load_keras_model().predict(processed, verbose=0)[0][0])

def predict_onnx(processed):
    session = load_onnx_model()
    input_name = session.get_inputs()[0].name
    return float(session.run(None, {input_name: processed.astype(np.float32)})[0][0][0])

def classify(prob_positive):
    prob_negative = 1 - prob_positive
    label = "Positive" if prob_positive > 0.5 else "Negative"
    confidence = max(prob_positive, prob_negative)
    return label, confidence, prob_positive, prob_negative

def file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0

# ============================================================
# STATE
# ============================================================
if "current_text" not in st.session_state:
    st.session_state.current_text = "The cinematography was breathtaking, and the acting was top-notch. Easily one of the best movies of the year!"
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# SIDEBAR — NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("<h2 style='color:#F5C518;'>Navigation</h2>", unsafe_allow_html=True)
    page = st.radio(
        "Menu",
        ["Home: Review Analysis", "About the Model", "Trivia & Versions"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("IMDb Review Sentiment by marcellmchan")

# ============================================================
# HALAMAN 1 — ANALISIS ULASAN (IMDB LAYOUT)
# ============================================================
if page == "Home: Review Analysis":
    
    # Layout 70-30 seperti "Main Trailer" dan "Up Next" di IMDb
    col_main, col_side = st.columns([7, 3], gap="large")

    with col_main:
        st.markdown("<h1 class='page-title'>Submit your review</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color:#AAA; font-size:16px;'>Test our Deep Learning AI sentiment predictor.</p>", unsafe_allow_html=True)
        
        model_choice = st.radio(
            "Engine Select",
            ["Model asli (.h5)", "Model teroptimasi (.onnx)"],
            horizontal=True,
        )
        
        user_input = st.text_area(
            "Write your review",
            value=st.session_state.current_text,
            height=150,
            label_visibility="collapsed"
        )

        if st.button("Rate This Review"):
            if user_input.strip() == "":
                st.warning("Review cannot be empty.")
            else:
                st.session_state.current_text = user_input
                with st.spinner("Analyzing semantic structure..."):
                    tokenizer = get_tokenizer()
                    processed = preprocess_text(user_input, tokenizer)
                    start = time.time()

                    if model_choice.startswith("Model asli"):
                        prob_positive = predict_keras(processed)
                        model_label = "H5 Engine"
                    else:
                        prob_positive = predict_onnx(processed)
                        model_label = "ONNX Engine"

                    elapsed = time.time() - start
                    label, confidence, prob_positive, prob_negative = classify(prob_positive)
                    score_10 = max(1.0, round(prob_positive * 10, 1))

                st.markdown("### User Ratings")
                
                res_col1, res_col2 = st.columns([1, 4])
                with res_col1:
                    st.markdown(f"""
                    <div style="text-align:center;">
                        <div style="color:#F5C518; font-size:40px; line-height:1;">★</div>
                        <div style="font-size:24px; font-weight:bold; color:#FFF;">{score_10}<span style="color:#AAA; font-size:16px; font-weight:normal;">/10</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with res_col2:
                    border_color = "#F5C518" if label == "Positive" else "#E50914"
                    st.markdown(f"""
                    <div style="background-color: #121212; padding: 15px; border-radius: 4px; border-left: 4px solid {border_color};">
                        <h4 style="color: #FFF; margin-top:0;">{label.upper()} SENTIMENT</h4>
                        <p style="color: #AAA; margin-bottom: 0;">Inference Time: {elapsed * 1000:.1f}ms via {model_label}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                c1, c2 = st.columns(2)
                with c1:
                    st.caption(f"Positive: {prob_positive*100:.1f}%")
                    st.progress(prob_positive)
                with c2:
                    st.caption(f"Negative: {prob_negative*100:.1f}%")
                    st.progress(prob_negative)

                st.session_state.history.append({
                    "snippet": user_input[:40] + "...",
                    "score": f"★ {score_10}/10",
                    "label": label,
                    "engine": model_label
                })

    with col_side:
        st.markdown("<h2 style='color:#F5C518; margin-bottom:15px;'>Up next</h2>", unsafe_allow_html=True)
        
        # Optimization Info box
        st.markdown("""
        <div style="background-color:#121212; padding:15px; margin-bottom:20px;">
            <p style="color:#FFF; font-weight:bold; margin-bottom:5px;">Model Specs</p>
            <p style="color:#AAA; font-size:12px; margin-top:0;">Compare sizes of inference engines.</p>
        """, unsafe_allow_html=True)
        
        size_h5 = file_size_mb(H5_MODEL_PATH)
        size_onnx = file_size_mb(ONNX_MODEL_PATH)
        st.markdown(f"<p style='color:#FFF; font-size:14px;'>Original (.h5): <span style='color:#F5C518; float:right;'>{size_h5:.2f} MB</span></p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#FFF; font-size:14px;'>Optimized (.onnx): <span style='color:#F5C518; float:right;'>{size_onnx:.2f} MB</span></p>", unsafe_allow_html=True)
        
        if size_h5 > 0:
            st.markdown(f"<p style='color:#AAA; font-size:12px;'>Saving: {(1 - size_onnx/size_h5)*100:.1f}%</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<h3 style='color:#FFF; font-size:16px;'>Recent Reviews</h3>", unsafe_allow_html=True)
        if st.session_state.history:
            for item in reversed(st.session_state.history[-4:]):
                color = "#F5C518" if item['label'] == "Positive" else "#E50914"
                st.markdown(f"""
                <div style="margin-bottom: 15px; border-bottom: 1px solid #333; padding-bottom: 10px;">
                    <span style="color: {color}; font-weight: bold; font-size:14px;">{item['score']}</span> 
                    <span style="color: #FFF; font-size:14px;">- {item['label']}</span>
                    <br>
                    <span style="color: #AAA; font-size: 13px;">"{item['snippet']}"</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #777; font-size:13px;'>No reviews submitted yet.</p>", unsafe_allow_html=True)

# ============================================================
# HALAMAN 2 — MODEL DEEP DIVE
# ============================================================
elif page == "About the Model":
    st.markdown("<h1 class='page-title'>Model Deep Dive</h1>", unsafe_allow_html=True)
    st.write("Pelajari cara model AI ini bekerja di balik layar — dari teks mentah hingga prediksi sentimen.")

    st.markdown("---")

    # ---- ARSITEKTUR GRU ----
    st.markdown("<h2>🧠 Arsitektur Model GRU</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown("""
    <div class="spec-card" style="text-align:center;">
        <p class="label">Layer Input</p>
        <p class="value">Embedding</p>
        <p style="color:#AAA; font-size:12px;">Vocab 10.000 kata<br>Seq. Length 100</p>
    </div>""", unsafe_allow_html=True)
    col2.markdown("""
    <div class="spec-card" style="text-align:center;">
        <p class="label">Hidden Layer</p>
        <p class="value">GRU</p>
        <p style="color:#AAA; font-size:12px;">Gated Recurrent Unit<br>Menangkap konteks urutan kata</p>
    </div>""", unsafe_allow_html=True)
    col3.markdown("""
    <div class="spec-card" style="text-align:center;">
        <p class="label">Output Layer</p>
        <p class="value">Dense</p>
        <p style="color:#AAA; font-size:12px;">Aktivasi Sigmoid<br>Output 0.0 – 1.0</p>
    </div>""", unsafe_allow_html=True)
    col4.markdown("""
    <div class="spec-card" style="text-align:center;">
        <p class="label">Dataset Pelatihan</p>
        <p class="value">IMDB</p>
        <p style="color:#AAA; font-size:12px;">50.000 review film<br>Label Positif / Negatif</p>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- CARA TOKENIZER BEKERJA ----
    st.markdown("<h2>🔤 Cara Tokenizer Bekerja</h2>", unsafe_allow_html=True)
    
    tokenizer = get_tokenizer()
    example_sentence = st.text_input(
        "Ketik kalimat untuk melihat proses tokenisasi:",
        value="This movie was amazing and brilliant!"
    )

    if example_sentence.strip():
        cleaned = example_sentence.lower()
        cleaned = re.sub(r'<[^>]*>', '', cleaned)
        cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', cleaned)
        words = cleaned.split()

        sequences = tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequences, maxlen=MAX_LEN, padding='post', truncating='post')

        st.markdown(f"**1. Teks asli:** `{example_sentence}`")
        st.markdown(f"**2. Setelah cleaning (lowercase, hapus karakter khusus):** `{cleaned}`")
        
        word_token_map = []
        for word in words:
            idx = tokenizer.word_index.get(word, None)
            word_token_map.append({"Kata": word, "Token ID": idx if idx else "⚠️ Tidak dikenali (OOV)"})
        
        import pandas as pd
        st.markdown("**3. Konversi kata → token ID:**")
        st.dataframe(pd.DataFrame(word_token_map), use_container_width=True, hide_index=True)
        st.markdown(f"**4. Padding ke panjang 100:** `{list(padded[0][:15])}... (+ {MAX_LEN - min(len(words), MAX_LEN)} padding 0)`")

    st.markdown("<br>", unsafe_allow_html=True)

    # ---- TOP KEYWORDS ----
    st.markdown("<h2>📊 Top Keywords Paling Berpengaruh</h2>", unsafe_allow_html=True)
    st.write("Kata-kata dengan frekuensi tertinggi dalam dataset IMDB yang dikenali model:")
    
    # Ambil top 20 kata dari tokenizer (skip stopwords umum)
    stopwords = {"the", "a", "and", "of", "to", "is", "in", "it", "i", "this", "that", "was", "for", "on", "are", "with", "as", "be", "by", "at"}
    top_words = [(word, idx) for word, idx in sorted(tokenizer.word_index.items(), key=lambda x: x[1]) if word not in stopwords][:30]
    
    # Kata berkonotasi positif & negatif
    positive_keywords = ["great", "good", "excellent", "wonderful", "best", "amazing", "love", "brilliant", "perfect", "enjoyed", "fantastic", "beautiful", "powerful", "outstanding"]
    negative_keywords = ["bad", "worst", "terrible", "awful", "boring", "waste", "disappointing", "horrible", "poor", "stupid", "pointless", "dull"]
    
    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("<h3 style='color:#F5C518;'>🌟 Sinyal Positif</h3>", unsafe_allow_html=True)
        for kw in positive_keywords:
            idx = tokenizer.word_index.get(kw, None)
            if idx:
                st.markdown(f"<div style='background:#121212; padding:6px 12px; border-radius:4px; margin-bottom:5px; border-left:3px solid #F5C518;'><span style='color:#FFF;'>`{kw}`</span> <span style='color:#888; float:right; font-size:12px;'>Token #{idx}</span></div>", unsafe_allow_html=True)
    with col_neg:
        st.markdown("<h3 style='color:#E50914;'>💔 Sinyal Negatif</h3>", unsafe_allow_html=True)
        for kw in negative_keywords:
            idx = tokenizer.word_index.get(kw, None)
            if idx:
                st.markdown(f"<div style='background:#121212; padding:6px 12px; border-radius:4px; margin-bottom:5px; border-left:3px solid #E50914;'><span style='color:#FFF;'>`{kw}`</span> <span style='color:#888; float:right; font-size:12px;'>Token #{idx}</span></div>", unsafe_allow_html=True)


# ============================================================
# HALAMAN 3 — TENTANG & VERSI
# ============================================================
else:
    st.markdown("<h1 class='page-title'>Trivia & Versions</h1>", unsafe_allow_html=True)
    st.write(
        "Aplikasi ini mengklasifikasikan sentimen review film berbahasa Inggris ke dalam sentimen "
        "Positif atau Negatif. Menggunakan arsitektur Gated Recurrent Unit (GRU) yang dilatih "
        "pada dataset IMDB 50K."
    )

    st.markdown("### Release History")
    for v in VERSION_HISTORY:
        st.markdown(f"""
        <div class="spec-card" style="margin-bottom:10px;">
            <p class="label" style="color:#F5C518; font-size:16px;">{v['versi'].upper()}</p>
            <p style="margin-top:0.3rem; color:#E0E0E0;">{v['perubahan']}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<div style='text-align: center; color: #555; font-size: 12px;'>An IMDb sentiment built with Streamlit</div>", unsafe_allow_html=True)
