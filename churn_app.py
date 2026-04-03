import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📡",
    layout="wide"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background: #0d0f14;
        color: #e8e8e8;
    }

    .main-title {
        font-family: 'Space Mono', monospace;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00ffa3;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #666;
        font-size: 0.95rem;
        margin-top: 4px;
        font-family: 'Space Mono', monospace;
    }

    .risk-card {
        border-radius: 12px;
        padding: 28px 32px;
        text-align: center;
        margin: 16px 0;
    }

    .risk-high {
        background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
        border: 1px solid #8b0000;
    }

    .risk-medium {
        background: linear-gradient(135deg, #1a1400, #2d2200);
        border: 1px solid #8b6000;
    }

    .risk-low {
        background: linear-gradient(135deg, #001a0a, #002d14);
        border: 1px solid #006030;
    }

    .risk-score {
        font-family: 'Space Mono', monospace;
        font-size: 3.5rem;
        font-weight: 700;
        margin: 8px 0;
    }

    .risk-label {
        font-size: 1rem;
        font-family: 'Space Mono', monospace;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .insight-box {
        background: #13161e;
        border: 1px solid #1e2230;
        border-radius: 10px;
        padding: 16px 20px;
        margin: 8px 0;
        font-size: 0.9rem;
    }

    .insight-title {
        font-family: 'Space Mono', monospace;
        font-size: 0.75rem;
        color: #00ffa3;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 6px;
    }

    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 0.7rem;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #444;
        margin: 24px 0 12px 0;
        border-bottom: 1px solid #1e2230;
        padding-bottom: 8px;
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #888 !important;
        font-size: 0.82rem !important;
        font-family: 'Space Mono', monospace !important;
        letter-spacing: 1px;
    }

    div[data-baseweb="select"] > div {
        background-color: #13161e !important;
        border-color: #1e2230 !important;
        color: #e8e8e8 !important;
    }

    .stSlider .stMarkdown { color: #888; }

    div.stButton > button {
        width: 100%;
        background: #00ffa3;
        color: #0d0f14;
        font-family: 'Space Mono', monospace;
        font-weight: 700;
        font-size: 0.9rem;
        letter-spacing: 2px;
        border: none;
        border-radius: 8px;
        padding: 14px;
        margin-top: 20px;
        cursor: pointer;
        transition: all 0.2s;
    }

    div.stButton > button:hover {
        background: #00cc82;
        transform: translateY(-1px);
    }

    .train-badge {
        background: #001a0a;
        border: 1px solid #006030;
        color: #00ffa3;
        font-family: 'Space Mono', monospace;
        font-size: 0.72rem;
        padding: 4px 12px;
        border-radius: 20px;
        display: inline-block;
        letter-spacing: 1px;
    }

    .metric-row {
        display: flex;
        gap: 12px;
        margin: 12px 0;
    }

    .metric-pill {
        background: #13161e;
        border: 1px solid #1e2230;
        border-radius: 20px;
        padding: 6px 14px;
        font-size: 0.8rem;
        font-family: 'Space Mono', monospace;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# MODEL TRAINING — runs once, cached
# WHY @st.cache_resource: The model trains only ONCE when the app
# first loads. Every subsequent prediction reuses the cached model.
# This is the no-pickle alternative — model lives in memory.
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_model():
    """
    Load data, engineer features, preprocess, and train model.
    Returns: trained model + preprocessor + feature metadata
    """
    # --- Load data ---
    # WHY read_csv here: The app is self-contained. No pickle = no version mismatch errors.
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # --- Fix types ---
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(0, inplace=True)
    df['Churn_Binary'] = (df['Churn'] == 'Yes').astype(int)

    # --- Feature Engineering ---
    df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ChargeIncrease'] = df['MonthlyCharges'] - df['AvgMonthlyCharges']

    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies']
    for col in service_cols:
        df[col + '_Flag'] = (df[col] == 'Yes').astype(int)
    df['TotalServices'] = df[[c + '_Flag' for c in service_cols]].sum(axis=1)
    df['HasMultipleServices'] = (df['TotalServices'] >= 3).astype(int)
    df['IsHighValue'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
    df['SeniorNoSupport'] = ((df['SeniorCitizen'] == 1) & (df['TechSupport'] == 'No')).astype(int)
    df['HighRiskProfile'] = ((df['Contract'] == 'Month-to-month') &
                              (df['PaymentMethod'] == 'Electronic check')).astype(int)

    # --- Prepare X, y ---
    drop_cols = ['customerID', 'Churn', 'Churn_Binary'] + [c + '_Flag' for c in service_cols]
    feature_df = df.drop(columns=drop_cols)
    y = df['Churn_Binary']

    cat_cols = feature_df.select_dtypes(include='object').columns.tolist()
    num_cols = feature_df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # --- Preprocessing pipeline with imputers ---
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # --- Split & SMOTE ---
    X_train, X_test, y_train, y_test = train_test_split(
        feature_df, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train_proc = preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

    # --- Train model ---
    model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X_train_bal, y_train_bal)

    return model, preprocessor, cat_cols, num_cols


# ─────────────────────────────────────────────
# BUILD INPUT ROW FROM USER SELECTIONS
# WHY this function: We mirror the exact same feature engineering
# the training pipeline used, so the model sees consistent data.
# ─────────────────────────────────────────────
def build_input_row(params: dict) -> pd.DataFrame:
    """Convert UI inputs → feature-engineered DataFrame row."""
    p = params

    # Derived fields (same logic as training)
    total_charges = p['MonthlyCharges'] * p['tenure']
    avg_monthly = total_charges / (p['tenure'] + 1)
    charge_increase = p['MonthlyCharges'] - avg_monthly

    service_list = [
        p['PhoneService'], p['MultipleLines'], p['OnlineSecurity'],
        p['OnlineBackup'], p['DeviceProtection'], p['TechSupport'],
        p['StreamingTV'], p['StreamingMovies']
    ]
    total_services = sum(1 for s in service_list if s == 'Yes')
    has_multiple = 1 if total_services >= 3 else 0
    is_high_value = 1 if p['MonthlyCharges'] > 65 else 0  # ~75th percentile
    senior_no_support = 1 if (p['SeniorCitizen'] == 1 and p['TechSupport'] == 'No') else 0
    high_risk = 1 if (p['Contract'] == 'Month-to-month' and
                       p['PaymentMethod'] == 'Electronic check') else 0

    row = {
        'SeniorCitizen': p['SeniorCitizen'],
        'Partner': p['Partner'],
        'Dependents': p['Dependents'],
        'tenure': p['tenure'],
        'PhoneService': p['PhoneService'],
        'MultipleLines': p['MultipleLines'],
        'InternetService': p['InternetService'],
        'OnlineSecurity': p['OnlineSecurity'],
        'OnlineBackup': p['OnlineBackup'],
        'DeviceProtection': p['DeviceProtection'],
        'TechSupport': p['TechSupport'],
        'StreamingTV': p['StreamingTV'],
        'StreamingMovies': p['StreamingMovies'],
        'Contract': p['Contract'],
        'PaperlessBilling': p['PaperlessBilling'],
        'PaymentMethod': p['PaymentMethod'],
        'MonthlyCharges': p['MonthlyCharges'],
        'TotalCharges': total_charges,
        'gender': p['gender'],
        'AvgMonthlyCharges': avg_monthly,
        'ChargeIncrease': charge_increase,
        'TotalServices': total_services,
        'HasMultipleServices': has_multiple,
        'IsHighValue': is_high_value,
        'SeniorNoSupport': senior_no_support,
        'HighRiskProfile': high_risk,
    }
    return pd.DataFrame([row])


# ─────────────────────────────────────────────
# GENERATE BUSINESS RECOMMENDATIONS
# ─────────────────────────────────────────────
def get_recommendations(params: dict, prob: float) -> list:
    recs = []
    if params['Contract'] == 'Month-to-month':
        recs.append(("📋 Contract Risk", "Offer 1–2 months free to switch to an annual plan. M2M customers churn 3× more."))
    if params['tenure'] <= 12:
        recs.append(("🚀 New Customer", "Trigger 90-day onboarding program. Assign a success touch-point at months 1, 3, and 6."))
    if params['MonthlyCharges'] > 65 and params['OnlineSecurity'] == 'No':
        recs.append(("🔒 Security Upsell", "High-paying customer without OnlineSecurity — offer 3-month free trial to increase stickiness."))
    if params['PaymentMethod'] == 'Electronic check':
        recs.append(("💳 Payment Method", "Incentivize auto-pay setup ($5/month discount). Auto-pay customers cancel less passively."))
    if params['SeniorCitizen'] == 1 and params['TechSupport'] == 'No':
        recs.append(("👴 Senior Care", "Enroll in priority TechSupport package. Senior customers without support churn from frustration."))
    if params['TotalServices'] <= 2 if 'TotalServices' in params else True:
        recs.append(("📦 Bundle Opportunity", "Customer uses few services. Present a bundle upgrade — more services = higher switching cost."))
    if not recs:
        recs.append(("✅ Healthy Profile", "Customer shows strong retention signals. Include in loyalty rewards and referral program."))
    return recs


# ─────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">📡 ChurnIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">// customer churn intelligence system</p>', unsafe_allow_html=True)

# Train model on first load
with st.spinner("Training model on Telco dataset..."):
    model, preprocessor, cat_cols, num_cols = train_model()

st.markdown('<span class="train-badge">✓ MODEL READY</span>', unsafe_allow_html=True)
st.markdown("---")

# Two-column layout: inputs left, results right
left, right = st.columns([1.2, 1], gap="large")

with left:
    st.markdown('<p class="section-header">Customer Demographics</p>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        senior = st.selectbox("Senior Citizen", ['No', 'Yes'])
        senior_val = 1 if senior == 'Yes' else 0
    with c2:
        partner = st.selectbox("Has Partner", ['Yes', 'No'])
        dependents = st.selectbox("Has Dependents", ['Yes', 'No'])

    st.markdown('<p class="section-header">Account Info</p>', unsafe_allow_html=True)

    tenure = st.slider("Tenure (months)", 0, 72, 12,
                        help="How long the customer has been with the company")
    monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)

    c3, c4 = st.columns(2)
    with c3:
        contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
        paperless = st.selectbox("Paperless Billing", ['Yes', 'No'])
    with c4:
        payment = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check',
            'Bank transfer (automatic)', 'Credit card (automatic)'
        ])

    st.markdown('<p class="section-header">Services Subscribed</p>', unsafe_allow_html=True)

    c5, c6, c7 = st.columns(3)
    with c5:
        internet = st.selectbox("Internet", ['Fiber optic', 'DSL', 'No'])
        phone = st.selectbox("Phone", ['Yes', 'No'])
        multi_lines = st.selectbox("Multi Lines", ['Yes', 'No', 'No phone service'])
    with c6:
        online_sec = st.selectbox("Online Security", ['Yes', 'No', 'No internet service'])
        online_bkp = st.selectbox("Online Backup", ['Yes', 'No', 'No internet service'])
        device_prot = st.selectbox("Device Protection", ['Yes', 'No', 'No internet service'])
    with c7:
        tech_sup = st.selectbox("Tech Support", ['Yes', 'No', 'No internet service'])
        stream_tv = st.selectbox("Streaming TV", ['Yes', 'No', 'No internet service'])
        stream_mv = st.selectbox("Streaming Movies", ['Yes', 'No', 'No internet service'])

    predict_btn = st.button("ANALYSE CHURN RISK →")

# ─────────────────────────────────────────────
# RIGHT PANEL — results appear here
# ─────────────────────────────────────────────
with right:
    if predict_btn:
        params = {
            'gender': gender,
            'SeniorCitizen': senior_val,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone,
            'MultipleLines': multi_lines,
            'InternetService': internet,
            'OnlineSecurity': online_sec,
            'OnlineBackup': online_bkp,
            'DeviceProtection': device_prot,
            'TechSupport': tech_sup,
            'StreamingTV': stream_tv,
            'StreamingMovies': stream_mv,
            'Contract': contract,
            'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly_charges,
        }

        input_df = build_input_row(params)
        input_processed = preprocessor.transform(input_df)
        prob = model.predict_proba(input_processed)[0][1]
        pct = int(prob * 100)

        # Risk card
        if prob >= 0.65:
            card_class = "risk-high"
            score_color = "#ff4444"
            risk_label = "HIGH RISK"
            emoji = "🔴"
        elif prob >= 0.35:
            card_class = "risk-medium"
            score_color = "#ffaa00"
            risk_label = "MEDIUM RISK"
            emoji = "🟡"
        else:
            card_class = "risk-low"
            score_color = "#00ffa3"
            risk_label = "LOW RISK"
            emoji = "🟢"

        st.markdown(f"""
        <div class="risk-card {card_class}">
            <div class="risk-label" style="color:{score_color}">{emoji} {risk_label}</div>
            <div class="risk-score" style="color:{score_color}">{pct}%</div>
            <div style="color:#888;font-size:0.85rem;font-family:'Space Mono',monospace">
                churn probability
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Key metrics derived
        total_services_count = sum(1 for s in [
            phone, multi_lines, online_sec, online_bkp,
            device_prot, tech_sup, stream_tv, stream_mv
        ] if s == 'Yes')
        total_charges_est = monthly_charges * tenure

        st.markdown(f"""
        <div class="metric-row">
            <span class="metric-pill">📅 {tenure}m tenure</span>
            <span class="metric-pill">💰 ${monthly_charges}/mo</span>
            <span class="metric-pill">🔌 {total_services_count} services</span>
            <span class="metric-pill">📦 ${total_charges_est:,.0f} LTV</span>
        </div>
        """, unsafe_allow_html=True)

        # Recommendations
        st.markdown('<p class="section-header">Retention Recommendations</p>', unsafe_allow_html=True)
        recs = get_recommendations({**params, 'TotalServices': total_services_count}, prob)
        for title, body in recs:
            st.markdown(f"""
            <div class="insight-box">
                <div class="insight-title">{title}</div>
                <div style="color:#ccc;line-height:1.5">{body}</div>
            </div>
            """, unsafe_allow_html=True)

        # Churn probability gauge (visual bar)
        st.markdown('<p class="section-header">Risk Breakdown</p>', unsafe_allow_html=True)

        risk_factors = {}
        if contract == 'Month-to-month': risk_factors['Month-to-month contract'] = 35
        if tenure <= 12: risk_factors['New customer (<12 months)'] = 25
        if monthly_charges > 65: risk_factors['High monthly charges'] = 20
        if payment == 'Electronic check': risk_factors['Electronic check payment'] = 15
        if online_sec == 'No' and internet != 'No': risk_factors['No online security'] = 10
        if senior_val == 1: risk_factors['Senior citizen'] = 10

        for factor, weight in list(risk_factors.items())[:4]:
            bar_pct = min(weight * 2.5, 100)
            st.markdown(f"""
            <div style="margin:8px 0">
                <div style="display:flex;justify-content:space-between;
                            font-size:0.78rem;color:#888;margin-bottom:4px;
                            font-family:'Space Mono',monospace">
                    <span>{factor}</span>
                    <span style="color:#ff6644">+{weight}pts</span>
                </div>
                <div style="background:#13161e;border-radius:4px;height:5px">
                    <div style="background:#ff6644;width:{bar_pct}%;height:5px;
                                border-radius:4px;transition:width 0.3s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Placeholder state
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#333">
            <div style="font-size:3rem;margin-bottom:16px">📡</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.85rem;
                        color:#444;letter-spacing:2px">
                AWAITING INPUT
            </div>
            <div style="font-size:0.8rem;color:#333;margin-top:8px">
                Fill in customer details and click Analyse
            </div>
        </div>
        """, unsafe_allow_html=True)