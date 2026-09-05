"""
Risk Strategy Framework.
Applies rule-based risk strategy on top of model scores and calculates Expected Loss.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

def setup_directories() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_data_path = project_root / 'data' / 'processed'
    models_path = project_root / 'models'
    reports_path = project_root / 'reports'
    return processed_data_path, models_path, reports_path

_PDO_FACTOR = 40 / np.log(2)  # 57.708 — PDO=40
_PDO_OFFSET = 507.8

def convert_to_score(prob: np.ndarray, offset: float = _PDO_OFFSET, factor: float = _PDO_FACTOR) -> np.ndarray:
    """PDO=20 calibration. At dataset mean default rate (6.68%), score = 660."""
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    odds_good = (1 - prob) / prob
    score = offset + factor * np.log(odds_good)
    score = np.clip(score, 300, 850)
    return score

def determine_risk_band(score: float) -> str:
    if score >= 750:
        return 'Approve - Green'
    elif score >= 650:
        return 'Approve with conditions - Yellow'
    elif score >= 550:
        return 'Manual review - Orange'
    else:
        return 'Decline - Red'

def main():
    processed_data_path, models_path, reports_path = setup_directories()
    
    print("Loading test data and model...")
    test_df_woe = pd.read_csv(processed_data_path / 'test_woe.csv')
    X_test = test_df_woe.drop('SeriousDlqin2yrs', axis=1)
    y_test = test_df_woe['SeriousDlqin2yrs']
    
    lr_model = joblib.load(models_path / 'logistic_regression.pkl')
    probs = lr_model.predict_proba(X_test)[:, 1]
    scores = convert_to_score(probs)
    
    results_df = pd.DataFrame({
        'Probability': probs,
        'Score': scores,
        'Actual_Default': y_test
    })
    
    results_df['Risk_Band'] = results_df['Score'].apply(determine_risk_band)
    
    # Calculate Expected Loss
    EAD = 10000
    LGD = 0.45
    results_df['Expected_Loss'] = results_df['Probability'] * LGD * EAD
    
    print("Generating Strategy Summary...")
    total_count = len(results_df)
    
    band_summary = results_df.groupby('Risk_Band').agg(
        Count=('Score', 'count'),
        Default_Rate=('Actual_Default', 'mean'),
        Avg_Expected_Loss=('Expected_Loss', 'mean')
    ).reset_index()
    
    band_summary['Approval_Rate'] = band_summary['Count'] / total_count
    
    approval_rates_by_band = dict(zip(band_summary['Risk_Band'], band_summary['Approval_Rate']))
    expected_default_rates_by_band = dict(zip(band_summary['Risk_Band'], band_summary['Default_Rate']))
    avg_expected_loss_by_band = dict(zip(band_summary['Risk_Band'], band_summary['Avg_Expected_Loss']))
    
    strategy_summary = {
        'approval_rates_by_band': approval_rates_by_band,
        'expected_default_rates_by_band': expected_default_rates_by_band,
        'avg_expected_loss_by_band': avg_expected_loss_by_band
    }
    
    with open(reports_path / 'strategy_summary.json', 'w') as f:
        json.dump(strategy_summary, f, indent=4)
        
    print("Strategy summary saved.")

if __name__ == "__main__":
    main()
