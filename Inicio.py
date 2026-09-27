import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(
    page_title="TF-IDF · Español",
    page_icon="📐",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# ESTILOS — BLUEPRINT SIMPLE
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

    :root {
        --paper:      #f4efe4;
        --paper-2:    #ebe5d5;
        --ink:        #1a1c1a;
        --muted:      #7a7a70;
        --blueprint:  #1e40af;
        --rule-soft:  #c7d0e8;
        --red:        #c1272d;
        --green:      #0f6b3a;
    }

    html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--ink) !important;
    }

    /* ═══ PAPEL + REJILLA ═══ */
    .stApp {
        background-color: var(--paper) !important;
        background-image:
            linear-gradient(rgba(30, 64, 175, 0.06) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 64, 175, 0.06) 1px, transparent 1px);
        background-size: 28px 28px;
        background-attachment: fixed;
    }

    /* ═══ SIDEBAR (por si acaso) ═══ */
    [data-testid="stSidebar"] {
        background: var(--paper-2) !important;
        border-right: 1px solid var(--rule-soft) !important;
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }

    /* ═══ ENCABEZADO ═══ */
    .header {
        border-bottom: 1.5px solid var(--blueprint);
        padding-bottom: 1.25rem;
        margin-bottom: 2rem;
    }
    .header h1 {
        font-family: 'Instrument Serif', serif !important;
        font-size: 3rem !important;
        font-weight: 400 !important;
        line-height: 1 !important;
        letter-spacing: -0.02em !important;
        color: var(--ink) !important;
        margin: 0 0 0.6rem 0 !important;
    }
    .header h1 em {
        font-style: italic;
        color: var(--red);
    }
    .header p {
        font-size: 1rem;
        color: var(--ink) !important;
        opacity: 0.75;
        margin: 0 !important;
        line-height: 1.55 !important;
        max-width: 620px;
    }

    /* ═══ ETIQUETA DE SECCIÓN ═══ */
    .label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--ink) !important;
        padding-bottom: 0.55rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid var(--rule-soft);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .label .hint {
        font-size: 0.68rem;
        color: var(--muted) !important;
        letter-spacing: 0.06em;
        text-transform: none;
    }

    /* ═══ INPUTS ═══ */
    .stTextArea label,
    .stTextInput label,
    .stTextArea [data-testid="stWidgetLabel"] p,
    .stTextInput [data-testid="stWidgetLabel"] p {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.08em !important;
        color: var(--muted) !important;
        margin-bottom: 0.35rem !important;
    }
    .stTextArea textarea,
    .stTextInput div[data-baseweb="input"] > div,
    .stTextInput input {
        background: #ffffff !important;
        border: 1px solid var(--rule-soft) !important;
        border-radius: 2px !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.9rem !important;
        line-height: 1.7 !important;
        padding: 0.85rem 1rem !important;
        transition: border-color 0.15s ease !important;
    }
    .stTextArea textarea:focus,
    .stTextInput div[data-baseweb="input"] > div:focus-within,
    .stTextInput input:focus {
        border-color: var(--blueprint) !important;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.10) !important;
        outline: none !important;
    }

    /* ═══ BOTONES SECUNDARIOS (preguntas sugeridas) ═══ */
    .stButton > button {
        background: #ffffff !important;
        border: 1px solid var(--rule-soft) !important;
        border-radius: 2px !important;
        color: var(--ink) !important;
        font-family: 'IBM Plex Sans', sans-serif !important;
        font-weight: 400 !important;
        font-size: 0.86rem !important;
        padding: 0.55rem 0.9rem !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
        justify-content: flex-start !important;
    }
    .stButton > button p {
        color: var(--ink) !important;
        font-size: 0.86rem !important;
        font-weight: 400 !important;
    }
    .stButton > button:hover {
        background: var(--blueprint) !important;
        border-color: var(--blueprint) !important;
    }
    .stButton > button:hover p { color: #ffffff !important; }

    /* ═══ BOTÓN PRIMARIO (Analizar) ═══ */
    .stButton > button[kind="primary"] {
        background: var(--red) !important;
        border: 1px solid var(--red) !important;
        color: #ffffff !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        padding: 0.8rem 1.4rem !important;
        justify-content: center !important;
        text-align: center !important;
    }
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.85rem !important;
        letter-spacing: 0.14em !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #9b1f24 !important;
        border-color: #9b1f24 !important;
    }

    /* ═══ TARJETA DE RESULTADO ═══ */
    .result {
        background: #ffffff;
        border: 1.5px solid var(--blueprint);
        border-radius: 2px;
        padding: 1.6rem 1.85rem;
        margin-top: 0.5rem;
        position: relative;
    }
    .result::before {
        content: '';
        position: absolute;
        top: -1.5px; left: -1.5px;
        width: 0; height: 0;
        border-top: 16px solid var(--red);
        border-right: 16px solid transparent;
    }
    .result .lbl {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.2em;
        text-transform: uppercase;
        color: var(--muted) !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .result .q {
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-size: 1.2rem;
        color: var(--ink) !important;
        line-height: 1.4;
        margin-bottom: 1.4rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px dashed var(--rule-soft);
    }
    .result .a {
        font-size: 1.15rem;
        line-height: 1.55;
        color: var(--ink) !important;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .result .score-head {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        margin-bottom: 0.5rem;
    }
    .result .score-head .n {
        font-family: 'Instrument Serif', serif;
        font-size: 2rem;
        line-height: 1;
        color: var(--blueprint) !important;
        font-variant-numeric: tabular-nums;
    }
    .result .score-head .t {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted) !important;
        font-weight: 600;
    }
    .result .bar {
        width: 100%;
        height: 6px;
        background: var(--paper-2);
        border-radius: 2px;
        overflow: hidden;
    }
    .result .bar > div {
        height: 100%;
        transition: width 0.4s ease;
    }

    /* ═══ ALERTAS ═══ */
    [data-testid="stAlert"] {
        border-radius: 2px !important;
        border: 1px solid var(--rule-soft) !important;
        background: var(--paper) !important;
    }
    [data-testid="stAlert"] * { color: var(--ink) !important; }

    /* ═══ DATAFRAME ═══ */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--rule-soft) !important;
        border-radius: 2px !important;
        overflow: hidden;
        background: #ffffff !important;
    }

    /* ═══ EXPANDER ═══ */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid var(--rule-soft) !important;
        border-radius: 2px !important;
        overflow: hidden;
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.76rem !important;
        letter-spacing: 0.08em !important;
        color: var(--ink) !important;
        font-weight: 500 !important;
        padding: 0.8rem 1rem !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--blueprint) !important; }

    /* ═══ SEPARADOR ═══ */
    hr {
        border: none !important;
        border-top: 1px dashed var(--rule-soft) !important;
        margin: 1.75rem 0 !important;
    }

    /* ═══ CAPTION ═══ */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        color: var(--muted) !important;
    }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--paper-2); }
    ::-webkit-scrollbar-thumb {
        background: var(--blueprint);
        border: 2px solid var(--paper-2);
    }

    /* ═══ RESPONSIVE ═══ */
    @media (max-width: 820px) {
        .header h1 { font-size: 2.2rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# ENCABEZADO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="header">
    <h1>TF-IDF <em>en español</em></h1>
    <p>Escribe varios documentos y una consulta. El motor selecciona el documento con mayor similitud semántica.</p>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════
default_docs = """El perro ladra fuerte en el parque.
El gato maúlla suavemente durante la noche.
El perro y el gato juegan juntos en el jardín.
Los niños corren y se divierten en el parque.
La música suena muy alta en la fiesta.
Los pájaros cantan hermosas melodías al amanecer."""

stemmer = SnowballStemmer("spanish")


def tokenize_and_stem(text):
    text = text.lower()
    text = re.sub(r'[^a-záéíóúüñ\s]', ' ', text)
    tokens = [t for t in text.split() if len(t) > 1]
    return [stemmer.stem(t) for t in tokens]


if "question_value" not in st.session_state:
    st.session_state.question_value = "¿Dónde juegan el perro y el gato?"


# ═══════════════════════════════════════════════════════════════
# LAYOUT
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown(
        '<div class="label">Documentos <span class="hint">uno por línea</span></div>',
        unsafe_allow_html=True,
    )
    text_input = st.text_area(
        "Documentos",
        default_docs,
        height=170,
        label_visibility="collapsed",
    )

with col2:
    st.markdown(
        '<div class="label">Consultas rápidas</div>',
        unsafe_allow_html=True,
    )
    suggested = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?",
    ]
    for i, q in enumerate(suggested):
        if st.button(q, use_container_width=True, key=f"btn_{i}"):
            st.session_state.question_value = q
            st.rerun()


st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)
st.markdown('<div class="label">Consulta</div>', unsafe_allow_html=True)
question = st.text_input(
    "Consulta",
    key="question_value",
    label_visibility="collapsed",
)

st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
with col_btn_c:
    analyze = st.button("▶  ANALIZAR", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# RESULTADOS
# ═══════════════════════════════════════════════════════════════
if analyze:
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una consulta.")
    else:
        vectorizer = TfidfVectorizer(tokenizer=tokenize_and_stem, min_df=1)
        X = vectorizer.fit_transform(documents)

        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = float(similarities[best_idx])

        # Color e interpretación según confianza
        if best_score > 0.3:
            bar_color = "#0f6b3a"
            level = "Alta relevancia"
        elif best_score > 0.01:
            bar_color = "#1e40af"
            level = "Relevancia moderada"
        else:
            bar_color = "#c1272d"
            level = "Baja relevancia"

        pct = min(100, int(best_score * 100))

        st.markdown(f"""
            <div class="result">
                <div class="lbl">Consulta</div>
                <div class="q">{question}</div>
                <div class="lbl">Documento más relevante</div>
                <div class="a">{best_doc}</div>
                <div class="score-head">
                    <div class="n">{best_score:.3f}</div>
                    <div class="t">{level}</div>
                </div>
                <div class="bar">
                    <div style="width: {pct}%; background: {bar_color};"></div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # Detalles opcionales
        with st.expander("Ver matriz TF-IDF y similitudes"):
            tab1, tab2 = st.tabs(["Similitudes", "Matriz TF-IDF"])

            with tab1:
                sim_df = pd.DataFrame({
                    "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                    "Texto": documents,
                    "Similitud": [round(float(s), 4) for s in similarities],
                }).sort_values("Similitud", ascending=False)
                st.dataframe(sim_df, use_container_width=True, hide_index=True)

            with tab2:
                df_tfidf = pd.DataFrame(
                    X.toarray(),
                    columns=vectorizer.get_feature_names_out(),
                    index=[f"Doc {i+1}" for i in range(len(documents))],
                )
                st.dataframe(df_tfidf.round(3), use_container_width=True)


st.markdown("---")
st.caption("TF-IDF · scikit-learn · SnowballStemmer (español) · similitud de coseno")
