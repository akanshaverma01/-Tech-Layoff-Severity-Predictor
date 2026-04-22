import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Tech Layoff Severity Predictor", page_icon="🔍", layout="wide")

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global background ── */
    .stApp,
    .stApp > div,
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section { background-color: #0f1117 !important; }

    /* ── Hide white top header bar ── */
    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    .stDeployButton,
    [data-testid="stToolbar"] { background-color: #0f1117 !important; border-bottom: none !important; }

    /* ── Sidebar collapse/expand arrow — make it visible ── */
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] svg,
    button[kind="header"],
    [data-testid="collapsedControl"],
    [data-testid="collapsedControl"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        background-color: #4f6ef7 !important;
        border-radius: 6px !important;
        opacity: 1 !important;
    }

    /* ── Caption text — make readable ── */
    .stCaption, .stCaption p,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p { color: #a8b4cc !important; font-size: 0.85rem !important; }

    /* ── Expander — dark themed ── */
    [data-testid="stExpander"],
    [data-testid="stExpanderDetails"] { background-color: #1a1d27 !important; border: 1px solid #2e3250 !important; border-radius: 10px !important; }
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary p,
    [data-testid="stExpander"] summary span { color: #c9d1e0 !important; font-weight: 600 !important; }
    [data-testid="stExpander"] svg { fill: #c9d1e0 !important; }
    [data-testid="stExpanderDetails"] p,
    [data-testid="stExpanderDetails"] td,
    [data-testid="stExpanderDetails"] th { color: #c9d1e0 !important; }

    /* ── Sidebar — force dark completely ── */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] > div:first-child,
    section[data-testid="stSidebar"] { background-color: #1a1d27 !important; border-right: 1px solid #2e3250 !important; }

    /* Sidebar text */
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] div { color: #c9d1e0 !important; }

    /* Sidebar input fields */
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] select,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="select"] {
        background-color: #12141f !important;
        border: 1px solid #2e3250 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] span {
        background-color: #12141f !important;
        color: #ffffff !important;
    }

    /* Sidebar number input container */
    [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
        background-color: #12141f !important;
        border: 1px solid #2e3250 !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] input {
        background-color: #12141f !important;
        color: #ffffff !important;
    }

    /* Sidebar +/- buttons on number input */
    [data-testid="stSidebar"] [data-testid="stNumberInputContainer"] button {
        background-color: #2e3250 !important;
        color: #ffffff !important;
        border: none !important;
    }

    /* Dropdown menu (opens outside sidebar, so target globally) */
    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        background-color: #1a1d27 !important;
        border: 1px solid #2e3250 !important;
    }
    [role="option"], [data-baseweb="menu"] li {
        background-color: #1a1d27 !important;
        color: #c9d1e0 !important;
    }
    [role="option"]:hover, [data-baseweb="menu"] li:hover {
        background-color: #2e3250 !important;
    }

    /* Predict button */
    [data-testid="stSidebar"] .stButton button {
        background: linear-gradient(135deg, #4f6ef7, #7c3aed) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 12px !important;
        width: 100% !important;
        margin-top: 12px !important;
    }
    [data-testid="stSidebar"] .stButton button:hover { opacity: 0.85 !important; }

    /* ── Main content ── */
    h2 { color: #ffffff !important; }
    h3 { color: #c9d1e0 !important; }
    h4 { color: #a8b4cc !important; }

    /* Metric cards */
    .metric-card {
        background: #1a1d27;
        border: 1px solid #2e3250;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
        margin: 4px 0;
    }
    .metric-card .label { color: #8b95b0; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 6px; }
    .metric-card .value { color: #ffffff; font-size: 1.4rem; font-weight: 700; }

    /* Step cards */
    .step-card {
        background: #1a1d27;
        border: 1px solid #2e3250;
        border-left: 4px solid #4f6ef7;
        border-radius: 10px;
        padding: 20px;
        height: 100%;
    }
    .step-card .step-num { color: #4f6ef7; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; }
    .step-card .step-text { color: #c9d1e0; font-size: 0.95rem; margin-top: 6px; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { background: #1a1d27; border-radius: 10px; padding: 4px; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; color: #8b95b0; font-weight: 500; }
    .stTabs [aria-selected="true"] { background: #4f6ef7 !important; color: white !important; }

    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #2e3250; border-radius: 10px; }

    /* Divider */
    hr { border-color: #2e3250 !important; }

    /* Caption */
    .stCaption { color: #6b7490 !important; }

    /* About table */
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px 14px; border-bottom: 1px solid #2e3250; color: #c9d1e0; text-align: left; }
    th { color: #8b95b0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.06em; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
YEAR = 2025   # Hardcoded — year is a training feature, not user input

TOP_15_INDUSTRIES = [
    'Finance', 'Retail', 'Healthcare', 'Transportation', 'Food',
    'Other', 'Marketing', 'Consumer', 'HR', 'Security',
    'Education', 'Real Estate', 'Crypto', 'Media', 'Data'
]
STAGES = [
    'Seed', 'Series A', 'Series B', 'Series C', 'Series D',
    'Series E', 'Series F', 'Series G', 'Series H', 'Series I',
    'Series J', 'Post-IPO', 'Private Equity', 'Acquired', 'Subsidiary', 'Unknown'
]
COUNTRIES = [
    'USA', 'India', 'Israel', 'Canada', 'Germany', 'UK', 'Brazil',
    'Australia', 'Singapore', 'Sweden', 'Nigeria', 'France',
    'Indonesia', 'Netherlands', 'Kenya', 'Ireland',
    'United Arabian Emirates', 'China', 'Japan', 'Estonia',
    'Argentina', 'Austria', 'Belgium', 'Cayman Islands', 'Chile',
    'Czech Republic', 'Denmark', 'Finland', 'Hong Kong', 'Lithuania',
    'Malaysia', 'Mexico', 'New Zealand', 'Norway', 'Pakistan',
    'Portugal', 'Romania', 'Russia', 'Senegal', 'South Korea',
    'Spain', 'Switzerland', 'Thailand', 'United Kingdom', 'Uruquay'
]
CONTINENTS = ['Africa', 'Asia', 'Europe', 'North America', 'Oceana', 'South America']

ALL_FEATURE_COLS = [
    'Company_Size_before_Layoffs', 'Year', 'Funding_per_Employee_log',
    'Money_Raised_log', 'Is_Stage_Unknown',
    'Country_Argentina', 'Country_Australia', 'Country_Austria',
    'Country_Belgium', 'Country_Brazil', 'Country_Canada',
    'Country_Cayman Islands', 'Country_Chile', 'Country_China',
    'Country_Czech Republic', 'Country_Denmark', 'Country_Estonia',
    'Country_Finland', 'Country_France', 'Country_Germany',
    'Country_Hong Kong', 'Country_India', 'Country_Indonesia',
    'Country_Ireland', 'Country_Israel', 'Country_Japan',
    'Country_Kenya', 'Country_Lithuania', 'Country_Malaysia',
    'Country_Mexico', 'Country_Netherlands', 'Country_New Zealand',
    'Country_Nigeria', 'Country_Norway', 'Country_Pakistan',
    'Country_Portugal', 'Country_Romania', 'Country_Russia',
    'Country_Senegal', 'Country_Singapore', 'Country_South Korea',
    'Country_Spain', 'Country_Sweden', 'Country_Switzerland',
    'Country_Thailand', 'Country_UK', 'Country_USA',
    'Country_United Arabian Emirates', 'Country_United Kingdom', 'Country_Uruquay',
    'Continent_Africa', 'Continent_Asia', 'Continent_Europe',
    'Continent_North America', 'Continent_Oceana', 'Continent_South America',
    'Industry_grouped_Consumer', 'Industry_grouped_Crypto',
    'Industry_grouped_Data', 'Industry_grouped_Education',
    'Industry_grouped_Finance', 'Industry_grouped_Food',
    'Industry_grouped_HR', 'Industry_grouped_Healthcare',
    'Industry_grouped_Marketing', 'Industry_grouped_Media',
    'Industry_grouped_Other', 'Industry_grouped_Real Estate',
    'Industry_grouped_Retail', 'Industry_grouped_Security',
    'Industry_grouped_Transportation',
    'Stage_Acquired', 'Stage_Post-IPO', 'Stage_Private Equity',
    'Stage_Seed', 'Stage_Series A', 'Stage_Series B', 'Stage_Series C',
    'Stage_Series D', 'Stage_Series E', 'Stage_Series F', 'Stage_Series G',
    'Stage_Series H', 'Stage_Series I', 'Stage_Series J',
    'Stage_Subsidiary', 'Stage_Unknown',
    'Size_Category_Enterprise', 'Size_Category_Large',
    'Size_Category_Mid', 'Size_Category_Small', 'Size_Category_Startup'
]

LABEL_MAP   = {0: '🟢 Low', 1: '🟡 Medium', 2: '🔴 High'}
LABEL_COLOR = {0: '#2ecc71', 1: '#f39c12', 2: '#e74c3c'}
LABEL_BG    = {0: '#0d2b1a', 1: '#2b1f0a', 2: '#2b0d0d'}
LABEL_DESC  = {
    0: 'Less than 10% of workforce affected. Minor restructuring.',
    1: '10–30% of workforce affected. Significant but partial cuts.',
    2: 'More than 30% of workforce affected. Massive restructuring.'
}

# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load('best_model.pkl')
    return pipeline, pipeline.named_steps['scaler'], pipeline.named_steps['model']

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

pipeline, scaler, model = load_model()
explainer = load_explainer(model)

# ── Feature engineering ────────────────────────────────────────────────────────
def size_category(n):
    if n < 100:     return 'Startup'
    elif n < 500:   return 'Small'
    elif n < 5000:  return 'Mid'
    elif n < 20000: return 'Large'
    else:           return 'Enterprise'

def build_feature_row(company_size, money_raised, stage, industry, country, continent, year):
    company_size_safe = max(float(company_size), 1.0)
    money_raised_safe = max(float(money_raised), 0.0)
    row = {col: 0 for col in ALL_FEATURE_COLS}
    row['Company_Size_before_Layoffs'] = company_size_safe
    row['Year']                        = year
    row['Funding_per_Employee_log']    = np.log1p(money_raised_safe / company_size_safe)
    row['Money_Raised_log']            = np.log1p(money_raised_safe)
    row['Is_Stage_Unknown']            = 1 if stage == 'Unknown' else 0
    industry_grouped = industry if industry in TOP_15_INDUSTRIES else 'Other'
    for key in [f'Country_{country}', f'Continent_{continent}',
                f'Industry_grouped_{industry_grouped}', f'Stage_{stage}',
                f'Size_Category_{size_category(company_size_safe)}']:
        if key in row:
            row[key] = 1
    return pd.DataFrame([row])[ALL_FEATURE_COLS]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 8px 0 24px 0;'>
    <h1 style='color:#ffffff; font-size:2.2rem; margin:0;'>🔍 Tech Layoff Severity Predictor</h1>
    <p style='color:#8b95b0; font-size:1rem; margin-top:8px;'>
        Enter your company's profile to estimate potential layoff severity,
        powered by a Random Forest model trained on 2,412 real tech layoff events (2020–2025).
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='color:#ffffff; margin-bottom:4px;'>🏢 Company Profile</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b95b0; font-size:0.85rem; margin-bottom:20px;'>Fill in the details below and click Predict.</p>", unsafe_allow_html=True)

    company_size = st.number_input("👥 Company Size (before layoffs)", min_value=1, max_value=500000, value=1000, step=100, help="Total headcount before layoffs")
    money_raised = st.number_input("💰 Total Funding Raised (millions USD)", min_value=0.0, max_value=200000.0, value=100.0, step=10.0, help="Total capital raised across all rounds")
    stage        = st.selectbox("📈 Funding Stage", options=STAGES, index=1, help="Current funding stage")
    industry     = st.selectbox("🏭 Industry", options=sorted(TOP_15_INDUSTRIES), help="Primary industry sector")
    country      = st.selectbox("🌍 Country (HQ)", options=sorted(COUNTRIES), index=sorted(COUNTRIES).index('USA'))
    continent    = st.selectbox("🗺️ Continent", options=CONTINENTS, index=CONTINENTS.index('North America'))

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔮 Predict Severity", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:#12141f; border:1px solid #2e3250; border-radius:10px; padding:14px;'>
        <p style='color:#6b7490; font-size:0.75rem; margin:0;'>
        ℹ️ <b style='color:#8b95b0;'>How to use:</b><br>
        This tool estimates layoff severity <i>if</i> layoffs were to occur at your company,
        based on similar historical cases. It does not predict whether layoffs will happen.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ── Landing page ───────────────────────────────────────────────────────────────
if not predict_btn:
    c1, c2, c3 = st.columns(3)
    for col, num, title, desc in [
        (c1, "01", "Fill the Profile", "Enter your company's size, funding, stage, industry and location in the sidebar."),
        (c2, "02", "Run the Model",    "Click <b>Predict Severity</b> to run the Random Forest classifier."),
        (c3, "03", "Read the Results", "See the severity class, confidence scores, and a full SHAP explanation of why."),
        
    ]:
        with col:
            st.markdown(f"""
            <div class='step-card'>
                <div class='step-num'>Step {num}</div>
                <div style='color:#ffffff; font-size:1rem; font-weight:600; margin: 8px 0 6px 0;'>{title}</div>
                <div class='step-text'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("<h3 style='color:#ffffff;'>📊 About this Model</h3>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        <table>
            <tr><th>Detail</th><th>Value</th></tr>
            <tr><td>Model</td><td>Random Forest Classifier</td></tr>
            <tr><td>Training Data</td><td>2,412 tech layoff events</td></tr>
            <tr><td>Time Period</td><td>2020 – 2025</td></tr>
            <tr><td>Test F1 Macro</td><td>0.6875</td></tr>
            <tr><td>Explainability</td><td>SHAP TreeExplainer</td></tr>
        </table>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown("""
        <div style='background:#1a1d27; border:1px solid #2e3250; border-radius:12px; padding:20px;'>
            <p style='color:#8b95b0; font-size:0.85rem; margin:0 0 10px 0; text-transform:uppercase; letter-spacing:0.08em;'>Severity Classes</p>
            <div style='margin-bottom:10px;'><span style='color:#2ecc71; font-weight:700;'>🟢 Low</span> <span style='color:#c9d1e0;'> — Less than 10% workforce affected</span></div>
            <div style='margin-bottom:10px;'><span style='color:#f39c12; font-weight:700;'>🟡 Medium</span> <span style='color:#c9d1e0;'> — 10–30% workforce affected</span></div>
            <div><span style='color:#e74c3c; font-weight:700;'>🔴 High</span> <span style='color:#c9d1e0;'> — More than 30% workforce affected</span></div>
        </div>
        """, unsafe_allow_html=True)

# ── Prediction page ────────────────────────────────────────────────────────────
else:
    # Build + scale input — YEAR is hardcoded, not from UI
    X_input = build_feature_row(company_size, money_raised, stage, industry, country, continent, YEAR)
    X_input = X_input.replace([np.inf, -np.inf], 0).fillna(0)

    feature_cols = pd.read_csv('feature_columns.csv')['feature'].tolist()
    X_input      = X_input[feature_cols]
    X_scaled     = scaler.transform(X_input)
    X_scaled_df  = pd.DataFrame(X_scaled, columns=feature_cols)

    # Predict
    pred_class = int(model.predict(X_scaled_df)[0])
    pred_proba = model.predict_proba(X_scaled_df)[0]
    pred_label = LABEL_MAP[pred_class]
    pred_color = LABEL_COLOR[pred_class]
    pred_bg    = LABEL_BG[pred_class]
    pred_desc  = LABEL_DESC[pred_class]

    # ── Result card ────────────────────────────────────────────────────────────
    st.markdown("<h2>🎯 Prediction Result</h2>", unsafe_allow_html=True)
    col_pred, col_conf = st.columns([1, 2], gap="large")

    with col_pred:
        st.markdown(f"""
        <div style='background:{pred_bg}; border: 2px solid {pred_color}; border-radius:16px;
                    padding:36px 24px; text-align:center; box-shadow: 0 0 40px {pred_color}33;'>
            <div style='font-size:3rem; margin-bottom:8px;'>
                {'🟢' if pred_class==0 else '🟡' if pred_class==1 else '🔴'}
            </div>
            <div style='color:{pred_color}; font-size:1.8rem; font-weight:800; letter-spacing:0.04em;'>
                {'LOW' if pred_class==0 else 'MEDIUM' if pred_class==1 else 'HIGH'}
            </div>
            <div style='color:#8b95b0; font-size:0.9rem; margin-top:10px; line-height:1.5;'>
                {pred_desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_conf:
        st.markdown("<h4>Confidence per Severity Class</h4>", unsafe_allow_html=True)
        colors  = ['#2ecc71', '#f39c12', '#e74c3c']
        labels  = ['🟢 Low', '🟡 Medium', '🔴 High']
        for lbl, prob, clr in zip(labels, pred_proba, colors):
            st.markdown(f"""
            <div style='margin-bottom:14px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:4px;'>
                    <span style='color:#c9d1e0; font-weight:600;'>{lbl}</span>
                    <span style='color:{clr}; font-weight:700;'>{prob*100:.1f}%</span>
                </div>
                <div style='background:#2e3250; border-radius:6px; height:10px;'>
                    <div style='background:{clr}; width:{prob*100:.1f}%; height:10px; border-radius:6px;'></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Engineered feature cards ───────────────────────────────────────────────
    st.markdown("<h2>🔧 What the Model Actually Sees</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b95b0;'>Your inputs were transformed into these key engineered features before prediction.</p>", unsafe_allow_html=True)

    size_cat        = size_category(company_size)
    funding_per_emp = money_raised / max(company_size, 1)

    e1, e2, e3, e4 = st.columns(4)
    for col, label, value in [
        (e1, "Size Category",        size_cat),
        (e2, "Funding / Employee",   f"${funding_per_emp:.2f}M"),
        (e3, "Log Funding",          f"{np.log1p(money_raised):.3f}"),
        (e4, "Stage Unknown Flag",   str(stage == 'Unknown')),
    ]:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='label'>{label}</div>
                <div class='value'>{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── SHAP section ───────────────────────────────────────────────────────────
    st.markdown("<h2>🧠 Why Did the Model Predict This?</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b95b0;'>SHAP values show which features pushed the prediction toward or away from the predicted class. <b style='color:#e74c3c;'>Red = pushed UP</b> &nbsp;|&nbsp; <b style='color:#4f6ef7;'>Blue = pulled DOWN</b></p>", unsafe_allow_html=True)

    # SHAP extraction — shape confirmed (1, 92, 3): axis2 = class
    shap_arr = np.array(explainer.shap_values(X_scaled_df))  # (1, 92, 3)
    sv       = shap_arr[0, :, pred_class].astype(float)      # (92,)

    ev_raw = explainer.expected_value
    ev = float(ev_raw[pred_class]) if hasattr(ev_raw, '__len__') else float(ev_raw)

    data_row = X_scaled_df.iloc[0].values  # (92,)

    tab1, tab2, tab3 = st.tabs(["📊 Waterfall Plot", "📈 SHAP Impact Chart", "📋 Feature Table"])

    with tab1:
        st.markdown(f"<h4>Explaining the <span style='color:{pred_color};'>{pred_label}</span> prediction</h4>", unsafe_allow_html=True)
        try:
            shap_exp = shap.Explanation(values=sv, base_values=ev, data=data_row, feature_names=feature_cols)
            shap.waterfall_plot(shap_exp, max_display=12, show=False)
            st.pyplot(plt.gcf())
        except Exception as e:
            st.error(f"Waterfall plot error: {e}")
        finally:
            plt.close('all')
        st.caption("Each bar = one feature. Bar length = how much it shifted the prediction from the baseline.")

    with tab2:
        st.markdown(f"<h4>Feature Impact Chart — <span style='color:{pred_color};'>{pred_label}</span> class</h4>", unsafe_allow_html=True)
        st.caption("Shows the top 15 features and how much each one pushed the prediction up (red) or down (blue).")
        try:
            shap_series = pd.Series(sv, index=feature_cols)
            # Top 15 by absolute value, sorted so biggest is at top
            top15 = shap_series.abs().sort_values(ascending=False).head(15)
            top15_vals = shap_series[top15.index].sort_values()

            colors = ['#e74c3c' if v > 0 else '#4f6ef7' for v in top15_vals]

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#1a1d27')
            ax.set_facecolor('#1a1d27')

            bars = ax.barh(top15_vals.index, top15_vals.values, color=colors, height=0.6, edgecolor='none')

            # Value labels on each bar
            for bar, val in zip(bars, top15_vals.values):
                x_pos = val + 0.001 if val >= 0 else val - 0.001
                ha = 'left' if val >= 0 else 'right'
                ax.text(x_pos, bar.get_y() + bar.get_height()/2,
                        f'{val:+.3f}', va='center', ha=ha,
                        color='#ffffff', fontsize=8)

            ax.axvline(0, color='#4a5070', linewidth=1, linestyle='--')
            ax.set_xlabel('SHAP Value  (positive = pushed toward this class)', color='#8b95b0', fontsize=9)
            ax.tick_params(colors='#c9d1e0', labelsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#2e3250')
            ax.spines['bottom'].set_color('#2e3250')
            for label in ax.get_yticklabels():
                label.set_color('#c9d1e0')

            plt.tight_layout()
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Feature impact chart error: {e}")
        finally:
            plt.close('all')

    with tab3:
        st.markdown("<h4>Top 10 Features Driving This Prediction</h4>", unsafe_allow_html=True)
        try:
            shap_series = pd.Series(sv, index=feature_cols)
            top10    = shap_series.abs().sort_values(ascending=False).head(10)
            top10_df = pd.DataFrame({
                'Feature':    top10.index,
                'SHAP Value': shap_series[top10.index].round(4),
                'Direction':  shap_series[top10.index].apply(
                    lambda x: '🔴 Pushed UP' if x > 0 else '🔵 Pulled DOWN'
                )
            }).reset_index(drop=True)
            st.dataframe(top10_df, use_container_width=True)
        except Exception as e:
            st.error(f"Feature table error: {e}")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # ── Input summary ──────────────────────────────────────────────────────────
    with st.expander("📋 See Your Full Input Summary"):
        st.markdown(f"""
| Field | Value |
|---|---|
| Company Size | {company_size:,} employees |
| Funding Raised | ${money_raised:.1f}M |
| Funding Stage | {stage} |
| Industry | {industry} |
| Country | {country} |
| Continent | {continent} |
| Size Category (derived) | {size_cat} |
""")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("<p style='color:#3d4460; font-size:0.8rem; text-align:center;'>Tech Layoff Severity Predictor &nbsp;·&nbsp; Random Forest + SHAP &nbsp;·&nbsp; Trained on 2,412 events (2020–2025)</p>", unsafe_allow_html=True)

