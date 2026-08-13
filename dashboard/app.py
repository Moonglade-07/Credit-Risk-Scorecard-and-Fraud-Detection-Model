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
    return lr, xgb, iso_forest, woe_mappings

@st.cache_data
def load_data():
    test_df_raw = pd.read_csv(processed_data_path / 'test_raw.csv')
    test_df_woe = pd.read_csv(processed_data_path / 'test_woe.csv')
    test_fraud = pd.read_csv(processed_data_path / 'test_with_fraud.csv')
    with open(reports_path / 'model_comparison.json', 'r') as f:
        model_metrics = json.load(f)
    return test_df_raw, test_df_woe, test_fraud, model_metrics

def convert_to_score(prob: float, offset: float = 600, factor: float = 28.8539) -> float:
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
    lr, xgb, iso_forest, woe_mappings = load_models()
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
        
        woe_data = input_data.copy()
        for feature in input_data.columns:
            edges = woe_mappings[feature]['edges']
            woe_map = woe_mappings[feature]['woe_map']
            bin_idx = pd.cut(woe_data[feature], bins=edges)
            woe_data[feature] = bin_idx.map(woe_map).astype(float).fillna(0)
            
        prob = lr.predict_proba(woe_data)[0, 1]
        score = convert_to_score(prob)
        band_text, band_color = determine_risk_band(score)
        
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
            st.error("🚨 HIGH FRAUD RISK DETECTED by Isolation Forest")
            
        st.subheader("SHAP Values (XGBoost)")
        import shap
        explainer = shap.TreeExplainer(xgb)
        shap_vals = explainer.shap_values(woe_data)
        
        # Determine format of shap values based on version/model output
        shap_array = shap_vals[0] if isinstance(shap_vals, list) or (isinstance(shap_vals, np.ndarray) and len(shap_vals.shape) > 1 and shap_vals.shape[0] == 1) else shap_vals
        if isinstance(shap_array, np.ndarray) and len(shap_array.shape) > 1:
            shap_array = shap_array[0]
            
        fig = go.Figure(go.Waterfall(
            orientation="h",
            measure=["relative"] * len(woe_data.columns) + ["total"],
            y=list(woe_data.columns) + ["Total Score"],
            x=list(shap_array) + [sum(shap_array)],
            textposition="outside"
        ))
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
