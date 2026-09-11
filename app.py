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
    {
        "versi": "v1",
        "perubahan": "Implementasi dasar model GRU (.h5) untuk klasifikasi sentimen review film berbahasa Inggris. Antarmuka satu halaman: input teks, lihat sentimen.",
    },
    {
        "versi": "v2",
        "perubahan": "Menambahkan opsi model teroptimasi (ONNX Runtime) untuk inferensi yang lebih cepat, breakdown probabilitas kelas, dan info perbandingan ukuran file.",
    },
    {
        "versi": "v3 (Final)",
        "perubahan": "Desain ulang antarmuka menjadi tema Cinematic/IMDb gelap, penambahan navigasi multi-halaman, dan fitur Uji Performa Benchmark secara langsung.",
    },
]

st.set_page_config(page_title="Cinematic Sentiment Analysis v3", page_icon="🎬", layout="wide")

# ============================================================
# GAYA (CSS) - Cinematic Dark Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700;900&display=swap');

:root {
    --bg: #0E0E10; --panel: #1A1A1D; --text-main: #E0E0E0; --text-muted: #888888;
    --imdb-yellow: #F5C518; --imdb-yellow-hover: #E2B616;
    --positive: #F5C518; --negative: #E50914;
    --border: #333333;
}

.stApp { background: var(--bg); color: var(--text-main); font-family: 'Roboto', sans-serif; }
h1, h2, h3 { color: var(--imdb-yellow) !important; font-weight: 700 !important; }
p, span, label, .stMarkdown { color: var(--text-main); }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #121212; border-right: 1px solid var(--border); }
[data-testid="stSidebar"] * { color: #CCCCCC !important; }
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent; border: none; padding: 0.4rem 0; margin-right: 0;
}
[data-testid="stSidebar"] input[type="radio"] { accent-color: var(--imdb-yellow); }
[data-testid="stSidebar"] div[role="radiogroup"] label p { font-weight: 400; }
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p { color: var(--imdb-yellow) !important; font-weight: 700; }

/* Main Area Radio */
div[role="radiogroup"] label {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.3rem 0.8rem; margin-right: 0.4rem;
}
.main input[type="radio"] { accent-color: var(--imdb-yellow); }

/* Buttons */
.stButton>button {
    background: var(--imdb-yellow); color: #000000 !important; border: none;
    border-radius: 4px; padding: 0.55rem 1.4rem; font-weight: 700;
}
.stButton>button:hover { background: var(--imdb-yellow-hover); color: #000000 !important; }
.stButton>button p, .stButton>button span, .stButton>button div { color: #000000 !important; }

/* Text Area */
.stTextArea textarea {
    background-color: var(--panel) !important; color: #FFF !important;
    border: 1px solid var(--border) !important; border-radius: 6px !important;
}
.stTextArea textarea:focus { border-color: var(--imdb-yellow) !important; box-shadow: 0 0 0 1px var(--imdb-yellow) !important; }

/* Progress Bar */
.stProgress > div > div { background: var(--imdb-yellow); }

/* Cards */
.result-card {
    background: var(--panel); border: 1px solid var(--border); border-left: 5px solid var(--imdb-yellow);
    border-radius: 6px; padding: 1.1rem 1.4rem; margin-top: 0.8rem;
}
.result-card.negative { border-left-color: var(--negative); }
.result-title { font-size: 1.4rem; font-weight: 700; margin: 0 0 0.2rem 0; color: #FFF; }
.result-meta { color: var(--text-muted); font-size: 0.9rem; }

.spec-card {
    background: var(--panel); border: 1px solid var(--border); border-radius: 6px; padding: 1rem 1.2rem;
}
.spec-card .label { color: var(--text-muted); font-size: 0.85rem; }
.spec-card .value { color: var(--imdb-yellow); font-size: 1.6rem; font-weight: 700; margin-top: 0.1rem; }
</style>
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
    st.session_state.current_text = "The cinematography was absolutely breathtaking. Every scene felt like a painting. Highly recommended!"


# ============================================================
# SIDEBAR — NAVIGASI
# ============================================================
with st.sidebar:
    st.markdown("### 🎬 IMDb Sentiment AI")
    st.caption("managed by marcellmchan")
    page = st.radio(
        "Navigasi",
        ["Analisis Ulasan", "Uji Performa Model", "Tentang & Versi"],
        label_visibility="collapsed",
    )
    st.markdown("---")


# ============================================================
# HALAMAN 1 — ANALISIS ULASAN
# ============================================================
if page == "Analisis Ulasan":
    st.markdown('<p style="color:#888; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:0;">AI Review Analysis</p>', unsafe_allow_html=True)
    st.title("Rate Your Movie Review")
    st.write("Tulis ulasan film Anda, pilih mesin inferensi, lalu jalankan klasifikasi sentimen.")

    model_choice = st.radio(
        "Pilih model inferensi",
        ["Model asli (.h5)", "Model teroptimasi (.onnx)"],
        horizontal=True,
        help="Model teroptimasi (ONNX) dirancang agar inferensi berjalan lebih cepat. Bandingkan performanya di menu 'Uji Performa Model'."
    )

    user_input = st.text_area(
        "Tulis ulasan (Bahasa Inggris)", 
        value=st.session_state.current_text,
        height=130
    )

    if st.button("Analisis Sentimen"):
        if user_input.strip() == "":
            st.warning("Ulasan tidak boleh kosong.")
        else:
            st.session_state.current_text = user_input
            with st.spinner("Memproses semantik bahasa..."):
                tokenizer = get_tokenizer()
                processed = preprocess_text(user_input, tokenizer)
                
                if model_choice.startswith("Model asli"):
                    prob_positive = predict_keras(processed)
                else:
                    prob_positive = predict_onnx(processed)
                
                label, confidence, prob_positive, prob_negative = classify(prob_positive)

            css_class = "positive" if label == "Positive" else "negative"
            emoji = "🌟" if label == "Positive" else "💔"
            rating = round(prob_positive * 10, 1) if label == "Positive" else round(prob_negative * 10, 1)

            st.markdown(f"""
            <div class="result-card {css_class}">
                <p class="result-title">{emoji} {label.upper()} REVIEW</p>
                <p class="result-meta">Tingkat keyakinan: {confidence*100:.1f}% — {model_choice} | Setara IMDb Rating: ★ {rating}/10</p>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"🌟 Positive Sentiment — {prob_positive*100:.1f}%")
                st.progress(prob_positive)
            with c2:
                st.write(f"💔 Negative Sentiment — {prob_negative*100:.1f}%")
                st.progress(prob_negative)


# ============================================================
# HALAMAN 2 — UJI PERFORMA MODEL
# ============================================================
elif page == "Uji Performa Model":
    st.markdown('<p style="color:#888; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:0;">Perbandingan Model</p>', unsafe_allow_html=True)
    st.title("Keras (.h5) vs ONNX Runtime")
    st.write(
        "Model ONNX dibuat untuk mempercepat inferensi. Halaman ini mengukur langsung selisih "
        "ukuran berkas dan kecepatan klasifikasi memakai teks ulasan terakhir yang Anda masukkan di halaman utama."
    )

    st.markdown(f"**Ulasan yang diuji:**\n> *\"{st.session_state.current_text[:100]}...\"*")
    
    n_runs = st.slider("Jumlah pengulangan inferensi", min_value=3, max_value=50, value=10)

    if st.button("Jalankan Benchmark"):
        tokenizer = get_tokenizer()
        processed = preprocess_text(st.session_state.current_text, tokenizer)
        
        with st.spinner("Menjalankan kedua model secara berulang..."):
            predict_keras(processed)   # warm-up
            predict_onnx(processed)    # warm-up

            keras_times, onnx_times = [], []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                prob_h5 = predict_keras(processed)
                keras_times.append((time.perf_counter() - t0) * 1000)

                t0 = time.perf_counter()
                prob_onnx = predict_onnx(processed)
                onnx_times.append((time.perf_counter() - t0) * 1000)

        label_h5, conf_h5, _, _ = classify(prob_h5)
        label_onnx, conf_onnx, _, _ = classify(prob_onnx)
        avg_h5 = sum(keras_times) / len(keras_times)
        avg_onnx = sum(onnx_times) / len(onnx_times)
        size_h5 = file_size_mb(H5_MODEL_PATH)
        size_onnx = file_size_mb(ONNX_MODEL_PATH)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
            <div class="spec-card">
                <p class="label">Model Keras Asli (.h5)</p>
                <p class="value">{size_h5:.1f} MB</p>
                <p class="result-meta">⏱ {avg_h5:.2f} ms/inferensi<br>🧠 Prediksi: {label_h5} ({conf_h5*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="spec-card">
                <p class="label">Model Teroptimasi (.onnx)</p>
                <p class="value">{size_onnx:.1f} MB</p>
                <p class="result-meta">⏱ {avg_onnx:.2f} ms/inferensi<br>🧠 Prediksi: {label_onnx} ({conf_onnx*100:.1f}%)</p>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        reduction = (1 - size_onnx / size_h5) * 100 if size_h5 > 0 else 0
        speed_note = "lebih cepat" if avg_onnx < avg_h5 else "lebih lambat"
        speed_diff = abs(avg_h5 - avg_onnx)
        speed_multiplier = avg_h5 / avg_onnx if avg_onnx > 0 else 0
        agree = "sama" if label_h5 == label_onnx else "berbeda"
        
        st.success(
            f"**Kesimpulan Benchmark:**\n\n"
            f"Model teroptimasi **{reduction:.0f}% lebih kecil** dan rata-rata **{speed_diff:.2f} ms {speed_note}** "
            f"({speed_multiplier:.1f}x lebih efisien) dibanding model asli. \n\n"
            f"Hasil klasifikasi kedua model **{agree}** untuk ulasan ini."
        )


# ============================================================
# HALAMAN 3 — TENTANG & VERSI
# ============================================================
else:
    st.markdown('<p style="color:#888; font-size:0.9rem; text-transform:uppercase; letter-spacing:1px; margin-bottom:0;">Dokumentasi Proyek</p>', unsafe_allow_html=True)
    st.title("Tentang aplikasi ini")
    st.write(
        "Aplikasi ini mengklasifikasikan sentimen review film berbahasa Inggris ke dalam sentimen "
        "Positif atau Negatif. Menggunakan arsitektur Gated Recurrent Unit (GRU) yang dilatih "
        "pada dataset IMDB 50K."
    )

    with st.expander("Cara pakai"):
        st.markdown("""
        1. Buka halaman **Analisis Ulasan**, lalu tulis atau paste ulasan film.
        2. Pilih model yang ingin dipakai (Keras standar atau ONNX Runtime yang lebih cepat).
        3. Klik **Analisis Sentimen** untuk melihat hasilnya.
        4. Untuk melihat perbandingan kecepatan inferensi yang nyata, buka halaman **Uji Performa Model**.
        """)

    st.subheader("Riwayat versi")
    for v in VERSION_HISTORY:
        st.markdown(f"""
        <div class="spec-card" style="margin-bottom:0.6rem;">
            <p class="label">{v['versi']}</p>
            <p style="margin-top:0.3rem; color:#E0E0E0;">{v['perubahan']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.caption(
        "Arsitektur: GRU, Vocab 10.000 kata, Sequence Length 100, klasifikasi biner (sigmoid). "
        "Model teroptimasi dikonversi ke format ONNX menggunakan tf2onnx untuk performa CPU yang lebih baik."
    )
