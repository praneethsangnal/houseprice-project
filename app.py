import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------
# Load Models
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "savedmodels"

model      = joblib.load(MODEL_DIR / "model.pkl")
scaler     = joblib.load(MODEL_DIR / "scaler.pkl")
medians    = joblib.load(MODEL_DIR / "medians.pkl")
categories = joblib.load(MODEL_DIR / "categories.pkl")
columns    = joblib.load(MODEL_DIR / "columns.pkl")
model_name = joblib.load(MODEL_DIR / "best_modelname.pkl")

# ----------------------------
# CSS
# ----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Instrument+Sans:wght@300;400;500&display=swap');

:root {
    --bg:       #0f1117;
    --bg2:      #161a24;
    --bg3:      #1c2130;
    --border:   rgba(255,255,255,.07);
    --text:     #e4e8f0;
    --muted:    #8a93a8;
    --faint:    #444e63;
    --blue:     #4f8ef7;
    --blue-lt:  #7aaeff;
    --blue-dim: rgba(79,142,247,.1);
    --green:    #34d399;
    --radius:   12px;
}

/* ── Base ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main .block-container {
    background: var(--bg) !important;
    color: var(--text);
    font-family: 'Instrument Sans', sans-serif;
}
.block-container { padding-top: 2rem !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text) !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: var(--blue) !important;
    box-shadow: 0 0 0 2px rgba(79,142,247,.15) !important;
}
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label {
    font-size: .73rem !important;
    letter-spacing: .07em;
    text-transform: uppercase;
    color: var(--muted) !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
}
[data-testid="stSidebar"] hr { border-color: var(--border) !important; }

/* ── Sidebar toggle button ── */
[data-testid="collapsedControl"] svg { fill: var(--muted) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }

/* ── Page title ── */
.page-title {
    text-align: center;
    padding: 1.5rem 1rem 1.2rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.page-title h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(1.6rem, 3.5vw, 2.4rem);
    font-weight: 800;
    color: var(--text);
    letter-spacing: -.02em;
    margin: 0 0 .4rem;
}
.page-title h1 span { color: var(--blue); }
.page-title .sub {
    font-size: .88rem;
    color: var(--muted);
    font-weight: 300;
}
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: .35rem;
    margin-top: .75rem;
    background: var(--blue-dim);
    border: 1px solid rgba(79,142,247,.22);
    border-radius: 99px;
    padding: .25rem .8rem;
    font-size: .72rem;
    color: var(--blue-lt);
    font-weight: 500;
    letter-spacing: .03em;
}

/* ── Section label ── */
.sec-label {
    display: flex;
    align-items: center;
    gap: .55rem;
    margin: 0 0 1rem;
}
.sec-label-txt {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .13em;
    text-transform: uppercase;
    color: var(--faint);
    white-space: nowrap;
    font-family: 'Syne', sans-serif;
}
.sec-label-line { flex: 1; height: 1px; background: var(--border); }

/* ── Snapshot cards ── */
.snap-grid-4 {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .75rem;
    margin-bottom: .75rem;
}
.snap-grid-3 {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .75rem;
    margin-bottom: 2rem;
}
.snap-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1rem .95rem;
    transition: border-color .18s;
}
.snap-card:hover { border-color: rgba(79,142,247,.28); }
.snap-icon { font-size: 1.25rem; margin-bottom: .45rem; display: block; }
.snap-lbl {
    font-size: .62rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--faint);
    font-weight: 500;
    margin-bottom: .2rem;
}
.snap-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text);
    line-height: 1;
}
.snap-val small {
    font-size: .65rem;
    color: var(--muted);
    font-family: 'Instrument Sans', sans-serif;
    font-weight: 400;
    margin-left: .18rem;
}
.qual-bar {
    height: 4px;
    border-radius: 99px;
    background: var(--bg3);
    margin-top: .65rem;
    overflow: hidden;
}
.qual-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--blue), var(--blue-lt));
}

/* ── Result card ── */
.result-card {
    background: var(--bg2);
    border: 1px solid rgba(79,142,247,.22);
    border-radius: 16px;
    padding: 2.4rem 2.2rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 2.5rem;
}
.result-card::before {
    content: '';
    position: absolute;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(79,142,247,.13) 0%, transparent 68%);
    top: -110px; right: -60px;
    pointer-events: none;
}
.result-eyebrow {
    font-size: .65rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--faint);
    font-weight: 500;
    margin-bottom: .5rem;
}
.result-price {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 5vw, 4.2rem);
    font-weight: 800;
    color: var(--text);
    line-height: 1;
    letter-spacing: -.03em;
}
.result-price sup {
    font-size: 40%;
    vertical-align: super;
    color: var(--blue);
    font-weight: 700;
}
.result-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: .6rem;
    margin-top: 1.4rem;
}
.result-stat {
    background: var(--bg3);
    border-radius: 9px;
    padding: .75rem .9rem;
    border: 1px solid var(--border);
}
.result-stat-lbl {
    font-size: .6rem;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--faint);
    margin-bottom: .2rem;
    font-weight: 500;
}
.result-stat-val {
    font-family: 'Syne', sans-serif;
    font-size: .95rem;
    font-weight: 700;
    color: var(--text);
}
.result-disclaimer {
    margin-top: .9rem;
    font-size: .72rem;
    color: var(--faint);
    line-height: 1.55;
}

/* ── Predict button ── */
div[data-testid="stButton"] > button {
    background: var(--blue) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    padding: .7rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: .85rem !important;
    letter-spacing: .06em;
    text-transform: uppercase;
    transition: background .18s, transform .12s;
}
div[data-testid="stButton"] > button:hover {
    background: var(--blue-lt) !important;
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button:active { transform: translateY(0); }

/* ── st.caption ── */
[data-testid="stCaptionContainer"] p { color: var(--faint) !important; font-size: .75rem !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("🏠 Enter House Details")

st.sidebar.subheader("📐 Size Features")
GrLivArea   = st.sidebar.number_input("Living Area (sq ft)",   min_value=100,   max_value=20000,  value=1500, step=50)
TotalBsmtSF = st.sidebar.number_input("Basement Area (sq ft)", min_value=0,     max_value=10000,  value=800,  step=50)
LotArea     = st.sidebar.number_input("Lot Area (sq ft)",      min_value=100,   max_value=500000, value=5000, step=100)

st.sidebar.subheader("🏗️ Structure Features")
YearBuilt   = st.sidebar.number_input("Year Built",             min_value=1800, max_value=2025,   value=2000, step=1)
OverallQual = st.sidebar.slider("Overall Quality (1–10)", 1, 10, 5)
GarageCars  = st.sidebar.number_input("Garage Capacity (cars)", min_value=0,    max_value=10,     value=1,    step=1)
FullBath    = st.sidebar.number_input("Number of Bathrooms",    min_value=0,    max_value=10,     value=1,    step=1)

st.sidebar.subheader("📍 Location")
Neighborhood = st.sidebar.selectbox(
    "Neighborhood",
    options=categories.get("Neighborhood", ["None"])
)

st.sidebar.divider()

# Validation
errors = []
if GrLivArea < 100:
    errors.append("Living Area must be ≥ 100 sq ft.")
if LotArea < 100:
    errors.append("Lot Area must be ≥ 100 sq ft.")
if YearBuilt > 2025:
    errors.append("Year Built cannot be in the future.")
for err in errors:
    st.sidebar.error(f"⚠ {err}")

predict_clicked = st.sidebar.button("🔍 Predict Price", disabled=bool(errors))
st.sidebar.caption("⚠ This is an AI-based estimate using historical data and may differ from real market prices.")

# ----------------------------
# Prediction Function (unchanged)
# ----------------------------
def preprocess_input(input_dict):
    df = pd.DataFrame([input_dict])
    for col, median in medians.items():
        if col not in df.columns:
            df[col] = median
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if pd.isna(df[col].iloc[0]):
            df[col] = median
    for col, allowed in categories.items():
        if col not in df.columns:
            df[col] = "None"
        elif df[col].iloc[0] not in allowed:
            df[col] = "None"
    df = pd.get_dummies(df, drop_first=True)
    df = df.reindex(columns=columns, fill_value=0)
    df = scaler.transform(df)
    return df

# ----------------------------
# Main canvas
# ----------------------------

# Title
st.markdown(f"""
<div class="page-title">
    <h1>🏠 House <span>Price Predictor</span></h1>
    <p class="sub">Enter house details in the sidebar and click <b>Predict Price</b>.</p>
    <div class="model-badge">⚙ &nbsp;<b>{model_name}</b></div>
</div>
""", unsafe_allow_html=True)

# Live snapshot
qual_pct = int(OverallQual / 10 * 100)

st.markdown(f"""
<div class="sec-label">
    <span class="sec-label-txt">Property Snapshot</span>
    <div class="sec-label-line"></div>
</div>
<div class="snap-grid-4">
    <div class="snap-card">
        <span class="snap-icon">📐</span>
        <div class="snap-lbl">Living Area</div>
        <div class="snap-val">{GrLivArea:,}<small>sqft</small></div>
    </div>
    <div class="snap-card">
        <span class="snap-icon">📅</span>
        <div class="snap-lbl">Year Built</div>
        <div class="snap-val">{YearBuilt}</div>
    </div>
    <div class="snap-card">
        <span class="snap-icon">🚗 🛁</span>
        <div class="snap-lbl">Garage · Bath</div>
        <div class="snap-val">{GarageCars}<small>car</small>&nbsp;&nbsp;{FullBath}<small>ba</small></div>
    </div>
    <div class="snap-card">
        <span class="snap-icon">⭐</span>
        <div class="snap-lbl">Overall Quality</div>
        <div class="snap-val">{OverallQual}<small>/ 10</small></div>
        <div class="qual-bar"><div class="qual-fill" style="width:{qual_pct}%"></div></div>
    </div>
</div>
<div class="snap-grid-3">
    <div class="snap-card">
        <span class="snap-icon">🏚</span>
        <div class="snap-lbl">Basement</div>
        <div class="snap-val">{TotalBsmtSF:,}<small>sqft</small></div>
    </div>
    <div class="snap-card">
        <span class="snap-icon">🌳</span>
        <div class="snap-lbl">Lot Area</div>
        <div class="snap-val">{LotArea:,}<small>sqft</small></div>
    </div>
    <div class="snap-card">
        <span class="snap-icon">📍</span>
        <div class="snap-lbl">Neighborhood</div>
        <div class="snap-val" style="font-size:1rem">{Neighborhood}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# Result
# ----------------------------
if predict_clicked:
    input_data = {
        "GrLivArea":    GrLivArea,
        "TotalBsmtSF":  TotalBsmtSF,
        "YearBuilt":    YearBuilt,
        "LotArea":      LotArea,
        "Neighborhood": Neighborhood,
        "OverallQual":  OverallQual,
        "GarageCars":   GarageCars,
        "FullBath":     FullBath,
    }

    try:
        processed  = preprocess_input(input_data)
        pred_log   = model.predict(processed)[0]
        pred_price = np.expm1(pred_log)

        price_int  = int(pred_price)
        formatted  = f"{price_int:,}"
        price_sqft = int(pred_price / GrLivArea) if GrLivArea else 0
        age        = 2025 - YearBuilt

        st.markdown(f"""
        <div class="sec-label">
            <span class="sec-label-txt">💰 Predicted Price</span>
            <div class="sec-label-line"></div>
        </div>
        <div class="result-card">
            <div class="result-eyebrow">Estimated Market Value</div>
            <div class="result-price"><sup>$</sup>{formatted}</div>
            <div class="result-grid">
                <div class="result-stat">
                    <div class="result-stat-lbl">Price per sq ft</div>
                    <div class="result-stat-val">${price_sqft:,}</div>
                </div>
                <div class="result-stat">
                    <div class="result-stat-lbl">Property Age</div>
                    <div class="result-stat-val">{age} yrs</div>
                </div>
                <div class="result-stat">
                    <div class="result-stat-lbl">Quality Score</div>
                    <div class="result-stat-val">{OverallQual} / 10</div>
                </div>
                <div class="result-stat">
                    <div class="result-stat-lbl">Neighborhood</div>
                    <div class="result-stat-val">{Neighborhood}</div>
                </div>
            </div>
            <div class="result-disclaimer">
                ⚠ AI estimate only &nbsp;·&nbsp; For indicative purposes only &nbsp;·&nbsp; Consult a licensed appraiser for official valuation
            </div>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Prediction failed. Please check inputs.\n\n`{e}`")