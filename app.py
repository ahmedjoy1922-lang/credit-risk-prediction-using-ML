import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Credit Risk Predictor",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Visual styling only — prediction logic and input values remain unchanged.
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --ink: #172033;
        --muted: #68758b;
        --accent: #5267e8;
        --accent-dark: #4053cf;
        --line: #e8ecf4;
        --surface: #ffffff;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 0%, rgba(105, 125, 255, .12), transparent 34%),
            radial-gradient(ellipse at 96% 12%, rgba(67, 193, 173, .10), transparent 28%),
            #f5f7fc;
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] { background: rgba(245,247,252,.82); }
    .block-container {
        max-width: 1120px;
        padding-top: 2.4rem;
        padding-bottom: 4rem;
    }

    h1, h2, h3, p, label, button, input, [data-baseweb="select"] {
        font-family: 'DM Sans', sans-serif !important;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 2.25rem 2.5rem;
        margin-bottom: 1.5rem;
        border-radius: 25px;
        color: white;
        background: linear-gradient(118deg, #202c55 0%, #354a9b 58%, #586ce4 100%);
        box-shadow: 0 18px 45px rgba(41, 58, 119, .18);
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 300px;
        height: 300px;
        right: -75px;
        top: -150px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 50%;
        box-shadow: 0 0 0 35px rgba(255,255,255,.035),
                    0 0 0 75px rgba(255,255,255,.025);
    }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 11px;
        border: 1px solid rgba(255,255,255,.22);
        border-radius: 30px;
        background: rgba(255,255,255,.10);
        color: #e4eaff;
        font-size: .72rem;
        font-weight: 700;
        letter-spacing: .12em;
        text-transform: uppercase;
    }
    .hero h1 {
        margin: 15px 0 8px;
        color: #fff;
        font-family: 'Manrope', sans-serif !important;
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 800;
        letter-spacing: -.045em;
        line-height: 1.12;
    }
    .hero p {
        max-width: 620px;
        margin: 0;
        color: #d9e1ff;
        font-size: 1rem;
        line-height: 1.65;
    }
    .section-heading {
        margin: .35rem 0 1rem;
        color: #26324a;
        font-family: 'Manrope', sans-serif !important;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -.02em;
    }
    .section-subtitle {
        margin-top: -.65rem;
        margin-bottom: 1.2rem;
        color: var(--muted);
        font-size: .88rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border: 1px solid rgba(226,231,242,.95);
        border-radius: 20px;
        background: rgba(255,255,255,.88);
        box-shadow: 0 8px 26px rgba(28, 42, 78, .045);
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stSelectbox"] label {
        color: #39465e !important;
        font-size: .86rem !important;
        font-weight: 600 !important;
    }
    [data-baseweb="input"], [data-baseweb="select"] > div {
        border-radius: 11px !important;
        border-color: #dfe5f0 !important;
        background: #fbfcff !important;
    }
    [data-baseweb="input"]:focus-within,
    [data-baseweb="select"]:focus-within {
        border-color: #7182ef !important;
        box-shadow: 0 0 0 3px rgba(82,103,232,.11) !important;
    }
    div.stButton > button {
        width: 100%;
        min-height: 3.15rem;
        border: 0;
        border-radius: 12px;
        background: linear-gradient(100deg, #5267e8, #6878ed);
        color: white;
        font-size: .98rem;
        font-weight: 700;
        box-shadow: 0 8px 18px rgba(82,103,232,.23);
        transition: transform .18s ease, box-shadow .18s ease;
    }
    div.stButton > button:hover {
        border: 0;
        background: linear-gradient(100deg, #4358d8, #5b6ce2);
        color: white;
        transform: translateY(-1px);
        box-shadow: 0 11px 22px rgba(82,103,232,.28);
    }
    div.stButton > button:focus { box-shadow: 0 0 0 3px rgba(82,103,232,.2); }
    [data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid var(--line);
    }
    .footnote {
        padding-top: 1rem;
        color: #8993a6;
        font-size: .78rem;
        text-align: center;
    }
    @media (max-width: 640px) {
        .block-container { padding: 1.2rem 1rem 2.5rem; }
        .hero { padding: 1.6rem 1.35rem; border-radius: 19px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

model = joblib.load("extra_trees_credit_model.pkl")
encoders = {col: joblib.load(f"{col}_encoder.pkl") for col in ["Sex", "Housing", "Saving accounts", "Checking account"]}

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">✦ Intelligent assessment</div>
        <h1>Credit Risk Predictor</h1>
        <p>Evaluate an applicant profile with a clear, simple risk prediction.
        Enter the applicant details below to get started.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-heading">Applicant information</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-subtitle">Complete each field using the applicant\'s available information.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown("**Personal profile**")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        sex = st.selectbox("Sex", ["male", "female"])
    with col2:
        job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1)
        housing = st.selectbox("Housing", ["own", "rent", "free"])

    st.divider()
    st.markdown("**Financial profile**")
    col3, col4 = st.columns(2)
    with col3:
        saving_accounts = st.selectbox("Saving Accounts", ["little", "moderate", "rich", "quite rich"])
        credit_amount = st.number_input("Credit Amount", min_value=0, value=1000)
    with col4:
        checking_account = st.selectbox("Checking Account", ["little", "moderate", "rich"])
        duration = st.number_input("Duration (months)", min_value=1, value=12)

input_df = pd.DataFrame({
    "Age": [age],
    "Sex": [encoders["Sex"].transform([sex])[0]],
    "Job": [job],
    "Housing": [encoders["Housing"].transform([housing])[0]],
    "Saving accounts": [encoders["Saving accounts"].transform([saving_accounts])[0]],
    "Checking account": [encoders["Checking account"].transform([checking_account])[0]],
    "Credit amount": [credit_amount],
    "Duration": [duration]
})

st.write("")
if st.button("✦  Predict Risk"):
    pred = model.predict(input_df)[0]

    if pred == 1:
        st.success("The predicted credit risk is: **GOOD**")
    else:
        st.error("The predicted credit risk is: **BAD**")

st.markdown('<div class="footnote">Credit Risk Prediction · Applicant assessment interface</div>', unsafe_allow_html=True)
