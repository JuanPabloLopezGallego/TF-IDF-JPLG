import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
from nltk.stem import SnowballStemmer

st.set_page_config(
    page_title="TF-IDF · Motor de similitud",
    page_icon="📐",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════
# ESTILOS — BLUEPRINT / ESQUEMA TÉCNICO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

    :root {
        --paper:      #f4efe4;
        --paper-2:    #ebe5d5;
        --paper-3:    #e2dcc9;
        --ink:        #1a1c1a;
        --ink-2:      #3a3d38;
        --muted:      #7a7a70;
        --muted-2:    #9a9a8c;
        --blueprint:  #1e40af;
        --blueprint-2:#3b5bdb;
        --rule:       #1e40af;
        --rule-soft:  #c7d0e8;
        --red:        #c1272d;
        --red-soft:   #fdecea;
        --amber:      #b7791f;
        --amber-soft: #fdf6e3;
        --green:      #0f6b3a;
        --green-soft: #e6f4ec;
    }

    html, body, [class*="css"], .stApp, button, input, textarea, select {
        font-family: 'IBM Plex Sans', sans-serif !important;
        color: var(--ink) !important;
    }

    /* ═══ PAPEL + REJILLA DE PLANO ═══ */
    .stApp {
        background-color: var(--paper) !important;
        background-image:
            linear-gradient(rgba(30, 64, 175, 0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(30, 64, 175, 0.07) 1px, transparent 1px),
            radial-gradient(circle at 15% 10%, rgba(30, 64, 175, 0.05), transparent 40%),
            radial-gradient(circle at 85% 90%, rgba(193, 39, 45, 0.04), transparent 40%);
        background-size: 28px 28px, 28px 28px, 100% 100%, 100% 100%;
        background-attachment: fixed;
    }

    /* ═══ SIDEBAR ═══ */
    [data-testid="stSidebar"] {
        background: var(--paper-2) !important;
        border-right: 1px solid var(--rule-soft) !important;
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }

    /* ═══ HEADER TIPO PLANO ═══ */
    .blueprint-header {
        border: 1.5px solid var(--rule);
        background: linear-gradient(180deg, #ffffff 0%, var(--paper) 100%);
        border-radius: 2px;
        padding: 1.75rem 2rem 1.6rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        box-shadow: 6px 6px 0 rgba(30, 64, 175, 0.08);
    }
    .blueprint-header::before,
    .blueprint-header::after {
        content: '';
        position: absolute;
        width: 14px; height: 14px;
        border: 1.5px solid var(--rule);
    }
    .blueprint-header::before { top: -1.5px;  left: -1.5px;  border-right: none; border-bottom: none; }
    .blueprint-header::after  { bottom: -1.5px; right: -1.5px; border-left: none; border-top: none; }

    .bp-topline {
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted) !important;
        padding-bottom: 0.85rem;
        margin-bottom: 1rem;
        border-bottom: 1px dashed var(--rule-soft);
    }
    .bp-topline .dwg { color: var(--red) !important; font-weight: 700; }
    .bp-topline .rev { color: var(--blueprint) !important; font-weight: 700; }

    .bp-main {
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 2rem;
        align-items: end;
    }
    .bp-title-block .kicker {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 0.24em;
        text-transform: uppercase;
        color: var(--blueprint) !important;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.55rem;
        margin-bottom: 0.65rem;
    }
    .bp-title-block .kicker::before {
        content: '';
        width: 22px; height: 1.5px;
        background: var(--blueprint);
    }
    .bp-title-block h1 {
        font-family: 'Instrument Serif', serif !important;
        font-size: 3.4rem !important;
        font-weight: 400 !important;
        line-height: 1 !important;
        letter-spacing: -0.02em !important;
        color: var(--ink) !important;
        margin: 0 0 0.85rem 0 !important;
    }
    .bp-title-block h1 em {
        font-style: italic;
        color: var(--red);
    }
    .bp-title-block p {
        font-size: 0.98rem;
        color: var(--ink-2) !important;
        margin: 0 !important;
        line-height: 1.6 !important;
        max-width: 640px;
    }
    .bp-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--muted) !important;
        line-height: 1.9;
        text-align: right;
        border-left: 1px dashed var(--rule-soft);
        padding-left: 1.4rem;
    }
    .bp-meta b {
        color: var(--ink) !important;
        font-weight: 600;
        letter-spacing: 0.08em;
    }
    .bp-meta .val { color: var(--blueprint) !important; }

    /* ═══ ETIQUETA DE SECCIÓN ═══ */
    .section-label {
        display: flex;
        align-items: baseline;
        gap: 0.85rem;
        margin: 0 0 1.1rem 0;
        padding-bottom: 0.65rem;
        border-bottom: 1px solid var(--rule);
    }
    .section-label .num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--red) !important;
        letter-spacing: 0.08em;
    }
    .section-label .ttl {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--ink) !important;
        letter-spacing: 0.18em;
        text-transform: uppercase;
    }
    .section-label .rule {
        flex: 1;
        height: 1px;
        background: repeating-linear-gradient(
            90deg,
            var(--rule) 0 6px,
            transparent 6px 12px
        );
        align-self: center;
    }

    /* ═══ INPUTS ═══ */
    .stTextArea label,
    .stTextInput label,
    .stTextArea [data-testid="stWidgetLabel"] p,
    .stTextInput [data-testid="stWidgetLabel"] p {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.14em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
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
        transition: all 0.15s ease !important;
    }
    .stTextArea textarea:focus,
    .stTextInput div[data-baseweb="input"] > div:focus-within,
    .stTextInput input:focus {
        border-color: var(--blueprint) !important;
        box-shadow: 0 0 0 3px rgba(30, 64, 175, 0.12) !important;
        outline: none !important;
    }

    /* ═══ BOTONES ═══ */
    .stButton > button {
        background: #ffffff !important;
        border: 1.5px solid var(--rule) !important;
        border-radius: 2px !important;
        color: var(--blueprint) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em !important;
        padding: 0.55rem 1rem !important;
        transition: all 0.15s ease !important;
        text-align: left !important;
        justify-content: flex-start !important;
        box-shadow: 3px 3px 0 rgba(30, 64, 175, 0.10);
    }
    .stButton > button p {
        color: inherit !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        background: var(--blueprint) !important;
        color: #ffffff !important;
        border-color: var(--blueprint) !important;
        box-shadow: 3px 3px 0 rgba(30, 64, 175, 0.35);
        transform: translate(-1px, -1px);
    }
    .stButton > button:hover p { color: #ffffff !important; }

    /* Botón primario (Analizar) - más grande, tipo "commit" */
    .stButton > button[kind="primary"] {
        background: var(--red) !important;
        border: 1.5px solid var(--red) !important;
        color: #ffffff !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.16em !important;
        text-transform: uppercase !important;
        padding: 0.75rem 1.4rem !important;
        justify-content: center !important;
        text-align: center !important;
        box-shadow: 4px 4px 0 rgba(193, 39, 45, 0.25) !important;
    }
    .stButton > button[kind="primary"] p {
        color: #ffffff !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.16em !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #9b1f24 !important;
        border-color: #9b1f24 !important;
        transform: translate(-1px, -1px);
        box-shadow: 4px 4px 0 rgba(193, 39, 45, 0.4) !important;
    }

    /* ═══ RESPUESTA / OUTPUT ═══ */
    .output-card {
        background: #ffffff;
        border: 1.5px solid var(--rule);
        border-radius: 2px;
        padding: 1.5rem 1.75rem;
        margin: 1rem 0;
        position: relative;
        box-shadow: 5px 5px 0 rgba(30, 64, 175, 0.08);
    }
    .output-card::before {
        content: '';
        position: absolute;
        top: -1.5px; left: -1.5px;
        width: 0; height: 0;
        border-top: 16px solid var(--red);
        border-right: 16px solid transparent;
    }
    .output-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--red) !important;
        font-weight: 700;
        margin-bottom: 0.7rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
    }
    .output-question {
        font-family: 'Instrument Serif', serif;
        font-style: italic;
        font-size: 1.15rem;
        color: var(--ink-2) !important;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
        border-bottom: 1px dashed var(--rule-soft);
        line-height: 1.4;
    }
    .output-answer {
        font-size: 1.1rem;
        line-height: 1.6;
        color: var(--ink) !important;
        font-weight: 500;
    }

    /* Score gauge */
    .score-block {
        display: flex;
        align-items: center;
        gap: 1.25rem;
        margin-top: 1.2rem;
        padding-top: 1.2rem;
        border-top: 1px solid var(--rule-soft);
    }
    .score-num {
        font-family: 'Instrument Serif', serif;
        font-size: 2.75rem;
        font-weight: 400;
        line-height: 1;
        color: var(--blueprint) !important;
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }
    .score-track {
        flex: 1;
        height: 8px;
        background: var(--paper-3);
        border: 1px solid var(--rule-soft);
        border-radius: 2px;
        overflow: hidden;
        position: relative;
    }
    .score-fill {
        height: 100%;
        background: repeating-linear-gradient(
            135deg,
            var(--blueprint) 0 6px,
            var(--blueprint-2) 6px 12px
        );
        transition: width 0.4s ease;
    }
    .score-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: var(--muted) !important;
        font-weight: 600;
    }

    /* ═══ ALERTAS NATIVAS REESTILIZADAS ═══ */
    [data-testid="stAlert"] {
        border-radius: 2px !important;
        border: 1.5px solid var(--rule) !important;
        background: var(--paper) !important;
        box-shadow: 4px 4px 0 rgba(30, 64, 175, 0.10);
    }
    [data-testid="stAlert"] * { color: var(--ink) !important; }

    /* ═══ DATAFRAME ═══ */
    [data-testid="stDataFrame"] {
        border: 1.5px solid var(--rule) !important;
        border-radius: 2px !important;
        overflow: hidden;
        box-shadow: 4px 4px 0 rgba(30, 64, 175, 0.10);
        background: #ffffff !important;
    }

    /* ═══ EXPANDER ═══ */
    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1.5px solid var(--rule) !important;
        border-radius: 2px !important;
        overflow: hidden;
        box-shadow: 4px 4px 0 rgba(30, 64, 175, 0.08);
    }
    [data-testid="stExpander"] summary {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        color: var(--ink) !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.1rem !important;
    }
    [data-testid="stExpander"] summary:hover { color: var(--blueprint) !important; }

    /* ═══ SEPARADOR / HR ═══ */
    hr {
        border: none !important;
        border-top: 1px dashed var(--rule-soft) !important;
        margin: 1.5rem 0 !important;
    }

    /* ═══ CAPTION ═══ */
    .stCaption, [data-testid="stCaptionContainer"] {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.72rem !important;
        color: var(--muted) !important;
        letter-spacing: 0.04em !important;
    }

    /* ═══ SCROLLBAR ═══ */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--paper-2); }
    ::-webkit-scrollbar-thumb {
        background: var(--blueprint);
        border-radius: 0;
        border: 2px solid var(--paper-2);
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--red); }

    /* ═══ RESPONSIVE ═══ */
    @media (max-width: 820px) {
        .bp-main { grid-template-columns: 1fr; }
        .bp-meta {
            text-align: left;
            border-left: none;
            border-top: 1px dashed var(--rule-soft);
            padding-left: 0;
            padding-top: 0.85rem;
        }
        .bp-title-block h1 { font-size: 2.3rem !important; }
    }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# HEADER TIPO PLANO TÉCNICO
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div class="blueprint-header">
    <div class="bp-topline">
        <span class="dwg">■ MOTOR DE RECUPERACIÓN DE INFORMACIÓN</span>
        <span class="rev">PLANO Nº TFIDF-03 · REV. A</span>
    </div>
    <div class="bp-main">
        <div class="bp-title-block">
            <div class="kicker">Vectorización y similitud · sklearn</div>
            <h1>TF-IDF <em>—</em> Similitud <em>en español</em></h1>
            <p>Escribe un conjunto de documentos, formula una consulta y observa cómo el
            algoritmo mide la relevancia mediante vectores de frecuencia invertida y
            similitud de coseno.</p>
        </div>
        <div class="bp-meta">
            <div><b>UNIDAD</b> · <span class="val">TF-IDF</span></div>
            <div><b>MÉTRICA</b> · <span class="val">COSENO</span></div>
            <div><b>IDIOMA</b> · <span class="val">ESPAÑOL</span></div>
            <div><b>STEMMER</b> · <span class="val">SNOWBALL</span></div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE DATOS
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
    stems = [stemmer.stem(t) for t in tokens]
    return stems


# Inicializamos el input de pregunta para que los botones lo actualicen correctamente
if "question_value" not in st.session_state:
    st.session_state.question_value = "¿Dónde juegan el perro y el gato?"


# ═══════════════════════════════════════════════════════════════
# LAYOUT PRINCIPAL
# ═══════════════════════════════════════════════════════════════
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("""
        <div class="section-label">
            <span class="num">§01</span>
            <span class="ttl">Documentos · Corpus</span>
            <span class="rule"></span>
        </div>
    """, unsafe_allow_html=True)

    text_input = st.text_area(
        "Documentos (uno por línea):",
        default_docs,
        height=160,
    )

    st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

    st.markdown("""
        <div class="section-label">
            <span class="num">§02</span>
            <span class="ttl">Consulta · Query</span>
            <span class="rule"></span>
        </div>
    """, unsafe_allow_html=True)

    question = st.text_input(
        "Escribe tu pregunta:",
        key="question_value",
    )

with col2:
    st.markdown("""
        <div class="section-label">
            <span class="num">§A</span>
            <span class="ttl">Índice</span>
            <span class="rule"></span>
        </div>
    """, unsafe_allow_html=True)

    # Preguntas sugeridas — estilo "índice de referencia" con numeración mono
    suggested = [
        "¿Dónde juegan el perro y el gato?",
        "¿Qué hacen los niños en el parque?",
        "¿Cuándo cantan los pájaros?",
        "¿Dónde suena la música alta?",
        "¿Qué animal maúlla durante la noche?",
    ]

    for i, q in enumerate(suggested, 1):
        label = f"{i:02d} · {q}"
        if st.button(label, use_container_width=True, key=f"btn_{i}"):
            st.session_state.question_value = q
            st.rerun()


# ═══════════════════════════════════════════════════════════════
# BOTÓN ANALIZAR
# ═══════════════════════════════════════════════════════════════
st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

col_btn_l, col_btn_c, col_btn_r = st.columns([1, 2, 1])
with col_btn_c:
    analyze = st.button("▶  EJECUTAR ANÁLISIS", type="primary", use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# RESULTADOS
# ═══════════════════════════════════════════════════════════════
if analyze:
    documents = [d.strip() for d in text_input.split("\n") if d.strip()]

    if len(documents) < 1:
        st.error("⚠️ Ingresa al menos un documento.")
    elif not question.strip():
        st.error("⚠️ Escribe una pregunta.")
    else:
        vectorizer = TfidfVectorizer(
            tokenizer=tokenize_and_stem,
            min_df=1,
        )
        X = vectorizer.fit_transform(documents)

        # ── Matriz TF-IDF ──
        st.markdown("""
            <div class="section-label">
                <span class="num">§03</span>
                <span class="ttl">Matriz TF-IDF · Documento × Término</span>
                <span class="rule"></span>
            </div>
        """, unsafe_allow_html=True)

        df_tfidf = pd.DataFrame(
            X.toarray(),
            columns=vectorizer.get_feature_names_out(),
            index=[f"Doc {i+1}" for i in range(len(documents))],
        )
        st.dataframe(df_tfidf.round(3), use_container_width=True)

        # ── Similitud ──
        question_vec = vectorizer.transform([question])
        similarities = cosine_similarity(question_vec, X).flatten()

        best_idx = similarities.argmax()
        best_doc = documents[best_idx]
        best_score = similarities[best_idx]

        # ── Respuesta ──
        st.markdown("""
            <div class="section-label" style="margin-top: 2rem;">
                <span class="num">§04</span>
                <span class="ttl">Resultado · Documento más relevante</span>
                <span class="rule"></span>
            </div>
        """, unsafe_allow_html=True)

        # Colores del score según confianza
        if best_score > 0.3:
            score_color = "#0f6b3a"
            score_label = "ALTA RELEVANCIA"
        elif best_score > 0.01:
            score_color = "#1e40af"
            score_label = "RELEVANCIA MODERADA"
        else:
            score_color = "#c1272d"
            score_label = "BAJA RELEVANCIA"

        pct = min(100, int(best_score * 100))

        st.markdown(f"""
            <div class="output-card">
                <div class="output-label">▸ CONSULTA PROCESADA</div>
                <div class="output-question">"{question}"</div>
                <div class="output-label" style="color: {score_color} !important;">▸ DOCUMENTO SELECCIONADO · DOC {best_idx + 1}</div>
                <div class="output-answer">{best_doc}</div>
                <div class="score-block">
                    <div>
                        <div class="score-label">Similitud coseno</div>
                        <div class="score-num">{best_score:.3f}</div>
                    </div>
                    <div style="flex: 1;">
                        <div class="score-label" style="margin-bottom: 0.4rem;">
                            {score_label}
                        </div>
                        <div class="score-track">
                            <div class="score-fill" style="width: {pct}%; background: {'repeating-linear-gradient(135deg, #0f6b3a 0 6px, #15803d 6px 12px)' if best_score > 0.3 else ('repeating-linear-gradient(135deg, #1e40af 0 6px, #3b5bdb 6px 12px)' if best_score > 0.01 else 'repeating-linear-gradient(135deg, #c1272d 0 6px, #dc2626 6px 12px)')};"></div>
                        </div>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        # ── Detalle de similitudes ──
        with st.expander("📊  VER SIMILITUD POR DOCUMENTO"):
            sim_df = pd.DataFrame({
                "Documento": [f"Doc {i+1}" for i in range(len(documents))],
                "Texto": documents,
                "Similitud": [round(float(s), 4) for s in similarities],
            }).sort_values("Similitud", ascending=False)
            st.dataframe(sim_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# PIE DE PLANO
# ═══════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.caption("**MOTOR TF-IDF · ESPAÑOL** — scikit-learn · NLTK SnowballStemmer · similitud de coseno · Streamlit")
