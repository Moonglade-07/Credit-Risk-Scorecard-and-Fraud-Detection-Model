import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
import json

st.set_page_config(page_title="Credit Risk Scorecard", layout="wide", page_icon="🏦")

# --- PATHS ---
project_root = Path(__file__).resolve().parent.parent
processed_data_path = project_root / 'data' / 'processed'
models_path = project_root / 'models'
reports_path = project_root / 'reports'

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    lr = joblib.load(models_path / 'logistic_regression.pkl')
    xgb = joblib.load(models_path / 'xgboost_tuned.pkl')
    iso_forest = joblib.load(models_path / 'isolation_forest.pkl')
    woe_mappings = joblib.load(processed_data_path / 'woe_mappings.pkl')
    cap_bounds = joblib.load(processed_data_path / 'cap_bounds.pkl')
    return lr, xgb, iso_forest, woe_mappings, cap_bounds

@st.cache_data
def load_data():
    test_df_raw = pd.read_csv(processed_data_path / 'test_raw.csv')
    test_df_woe = pd.read_csv(processed_data_path / 'test_woe.csv')
    test_fraud = pd.read_csv(processed_data_path / 'test_with_fraud.csv')
    with open(reports_path / 'model_comparison.json', 'r') as f:
        model_metrics = json.load(f)
    return test_df_raw, test_df_woe, test_fraud, model_metrics

# PDO=40 calibration: factor = 40/ln(2) = 57.708
# At dataset mean default rate (6.68%) -> score 660
# At best model output (~0.5% PD)      -> score ~813 (Green)
# At high risk (20% PD)                -> score ~588 (Orange)
# At subprime (80% PD)                 -> score ~428 (Red)
PDO_FACTOR = 40 / np.log(2)   # 57.708
PDO_OFFSET = 507.8

def convert_to_score(prob: float, offset: float = PDO_OFFSET, factor: float = PDO_FACTOR) -> float:
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    odds_good = (1 - prob) / prob
    score = offset + factor * np.log(odds_good)
    return np.clip(score, 300, 850)

def determine_risk_band(score: float):
    if score >= 750: return 'Approve - Green', 'green'
    elif score >= 650: return 'Approve with conditions - Yellow', 'orange'
    elif score >= 550: return 'Manual review - Orange', 'darkorange'
    else: return 'Decline - Red', 'red'

# --- MAIN APP ---
st.sidebar.title("🏦 Credit Risk System")
page = st.sidebar.radio("Navigate", ["Portfolio Overview", "Borrower Risk Assessment", "Model Performance", "Fraud Risk Monitor"])

try:
    lr, xgb, iso_forest, woe_mappings, cap_bounds = load_models()
    test_df_raw, test_df_woe, test_fraud, model_metrics = load_data()
except Exception as e:
    st.error("Data or models not found. Please run the pipeline first to generate the models and data.")
    st.stop()

if page == "Portfolio Overview":
    st.title("Portfolio Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Borrowers", f"{len(test_df_raw):,}")
    
    default_rate = test_df_raw['SeriousDlqin2yrs'].mean() * 100
    col2.metric("Overall Default Rate", f"{default_rate:.2f}%")
    
    best_model_auc = model_metrics['XGBoost Tuned']['AUC-ROC']
    col3.metric("Model AUC (Best)", f"{best_model_auc:.3f}")
    
    fraud_rate = test_fraud['is_fraud_risk'].mean() * 100
    col4.metric("Fraud Risk Rate", f"{fraud_rate:.2f}%")
    
    probs = lr.predict_proba(test_df_woe.drop('SeriousDlqin2yrs', axis=1))[:, 1]
    scores = [convert_to_score(p) for p in probs]
    test_df_raw['Score'] = scores
    
    def band(s):
        if s >= 750: return 'Green'
        elif s >= 650: return 'Yellow'
        elif s >= 550: return 'Orange'
        else: return 'Red'
        
    test_df_raw['Band'] = test_df_raw['Score'].apply(band)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Score Distribution")
        fig = px.histogram(test_df_raw, x="Score", color="Band",
                           color_discrete_map={'Green': 'green', 'Yellow': 'gold', 'Orange': 'orange', 'Red': 'red'},
                           nbins=50)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("Default Rate by Score Band")
        rate_by_band = test_df_raw.groupby('Band')['SeriousDlqin2yrs'].mean().reset_index()
        fig = px.bar(rate_by_band, x='Band', y='SeriousDlqin2yrs',
                     color="Band", color_discrete_map={'Green': 'green', 'Yellow': 'gold', 'Orange': 'orange', 'Red': 'red'})
        fig.update_layout(yaxis_title="Default Rate")
        st.plotly_chart(fig, use_container_width=True)
        
    st.subheader("Feature Importance (XGBoost)")
    st.image(str(reports_path / 'feature_importance_xgb.png'))

elif page == "Borrower Risk Assessment":
    st.title("Borrower Risk Assessment")
    
    with st.form("borrower_form"):
        col1, col2 = st.columns(2)
        with col1:
            utilization = st.number_input("Revolving Utilization", min_value=0.0, value=0.5)
            age = st.number_input("Age", min_value=18, max_value=120, value=45)
            dpd_30_59 = st.number_input("Times 30-59 Days Past Due", min_value=0, value=0)
            debt_ratio = st.number_input("Debt Ratio", min_value=0.0, value=0.3)
            monthly_income = st.number_input("Monthly Income", min_value=0.0, value=5000.0)
        with col2:
            open_credit_lines = st.number_input("Open Credit Lines/Loans", min_value=0, value=5)
            dpd_90 = st.number_input("Times 90+ Days Late", min_value=0, value=0)
            real_estate_loans = st.number_input("Real Estate Loans/Lines", min_value=0, value=1)
            dpd_60_89 = st.number_input("Times 60-89 Days Past Due", min_value=0, value=0)
            dependents = st.number_input("Number of Dependents", min_value=0, value=0)
            
        loan_amount = st.number_input("Requested Loan Amount (EAD)", min_value=1000, value=10000)
            
        submitted = st.form_submit_button("Assess Risk")
        
    if submitted:
        input_data = pd.DataFrame([{
            'RevolvingUtilizationOfUnsecuredLines': utilization,
            'age': age,
            'NumberOfTime30-59DaysPastDueNotWorse': dpd_30_59,
            'DebtRatio': debt_ratio,
            'MonthlyIncome': monthly_income,
            'NumberOfOpenCreditLinesAndLoans': open_credit_lines,
            'NumberOfTimes90DaysLate': dpd_90,
            'NumberRealEstateLoansOrLines': real_estate_loans,
            'NumberOfTime60-89DaysPastDueNotWorse': dpd_60_89,
            'NumberOfDependents': dependents
        }])

        # Step 1: Apply training-time outlier caps so inference matches training exactly
        capped_data = input_data.copy()
        for feature, bounds in cap_bounds.items():
            if bounds is not None and feature in capped_data.columns:
                lower, upper = bounds
                capped_data[feature] = np.clip(capped_data[feature], lower, upper)

        # Step 2: WoE encode using saved mappings (include_lowest=True avoids boundary NaNs)
        woe_data = capped_data.copy()
        woe_values = {}
        for feature in capped_data.columns:
            edges   = woe_mappings[feature]['edges']
            woe_map = woe_mappings[feature]['woe_map']
            binned  = pd.cut(capped_data[feature], bins=edges, include_lowest=True)
            woe_val = binned.map(woe_map).astype(float).fillna(0.0)
            woe_data[feature] = woe_val
            woe_values[feature] = float(woe_val.iloc[0])

        # Step 3: Predict
        prob  = lr.predict_proba(woe_data)[0, 1]
        score = convert_to_score(prob)
        band_text, band_color = determine_risk_band(score)

        # Step 4: Fraud check
        fraud_features = [
            'RevolvingUtilizationOfUnsecuredLines', 'NumberOfTimes90DaysLate',
            'NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTime60-89DaysPastDueNotWorse', 'DebtRatio'
        ]
        is_fraud = iso_forest.predict(input_data[fraud_features])[0] == -1

        el = prob * 0.45 * loan_amount

        st.divider()
        col1, col2, col3 = st.columns(3)
        col1.metric("Credit Score", f"{int(score)}", delta=band_text, delta_color="off")
        col2.metric("Probability of Default (PD)", f"{prob*100:.2f}%")
        col3.metric("Expected Loss (EL)", f"${el:.2f}")

        st.markdown(f"### Recommendation: <span style='color:{band_color}'>{band_text}</span>", unsafe_allow_html=True)
        if is_fraud:
            st.error("\U0001f6a8 HIGH FRAUD RISK DETECTED by Isolation Forest")

        # Step 5: Score Factor Breakdown — shows which features helped/hurt the score
        st.subheader("Score Factor Breakdown")
        lr_coef = dict(zip(woe_data.columns, lr.coef_[0]))
        factor_contributions = []
        for feat, woe_v in woe_values.items():
            coef = lr_coef.get(feat, 0)
            # Contribution to log-odds (positive = lowers default risk = raises score)
            contribution = -coef * woe_v  # negative because higher prob = lower score
            factor_contributions.append({
                'Feature': feat,
                'WoE Value': round(woe_v, 4),
                'Score Impact': round(contribution * PDO_FACTOR, 1),
                'Effect': '\u25b2 Positive' if contribution >= 0 else '\u25bc Negative'
            })
        factors_df = pd.DataFrame(factor_contributions).sort_values('Score Impact', ascending=False)
        st.dataframe(
            factors_df.style.applymap(
                lambda v: 'color: #2ecc71' if '\u25b2' in str(v) else ('color: #e74c3c' if '\u25bc' in str(v) else ''),
                subset=['Effect']
            ),
            use_container_width=True
        )

        st.subheader("SHAP Values (XGBoost — Risk Driver Analysis)")
        import shap
        explainer  = shap.TreeExplainer(xgb)
        shap_vals  = explainer.shap_values(woe_data)
        shap_array = shap_vals[0] if isinstance(shap_vals, list) or (
            isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) > 1 and shap_vals.shape[0] == 1
        ) else shap_vals
        if isinstance(shap_array, np.ndarray) and len(shap_array.shape) > 1:
            shap_array = shap_array[0]
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(woe_data.columns) + ["total"],
            y=list(woe_data.columns) + ["Total"],
            x=list(shap_array) + [float(sum(shap_array))],
            textposition="outside"
        ))
        fig.update_layout(title="SHAP Waterfall (positive = higher default risk)")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Model Performance":
    st.title("Model Performance")
    
    st.subheader("Metrics Comparison")
    df_metrics = pd.DataFrame(model_metrics).T
    st.dataframe(df_metrics.style.highlight_max(axis=0))
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("ROC Curves")
        st.image(str(reports_path / 'roc_curves.png'))
    with col2:
        st.subheader("Calibration Curve")
        st.image(str(reports_path / 'calibration_curve.png'))
        
    st.subheader("Confusion Matrix (Best Model)")
    st.image(str(reports_path / 'confusion_matrices.png'))

elif page == "Fraud Risk Monitor":
    st.title("Fraud Risk Monitor")
    
    with open(reports_path / 'fraud_analysis.json', 'r') as f:
        fraud_analysis = json.load(f)
        
    col1, col2 = st.columns(2)
    col1.metric("Overall Fraud Rate", f"{fraud_analysis['fraud_rate_pct']:.2f}%")
    col2.metric("Overlap with Actual Defaults", f"{fraud_analysis['overlap_with_default_pct']:.2f}%")
    
    st.subheader("Fraud Risk vs Defaults (Sample)")
    fig = px.histogram(test_fraud, x='fraud_score', color='is_fraud_risk',
                       title="Fraud Score Distribution", barmode="overlay")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Fraud Risk vs Defaults Scatter")
    fig2 = px.scatter(test_fraud.sample(1000), x='fraud_score', y='SeriousDlqin2yrs',
                     color='is_fraud_risk', opacity=0.5)
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Recent Fraud Alerts")
    fraud_cases = test_fraud[test_fraud['is_fraud_risk'] == 1].head(10)
    display_cols = ['age', 'MonthlyIncome', 'RevolvingUtilizationOfUnsecuredLines', 'fraud_score']
    st.dataframe(fraud_cases[display_cols])
