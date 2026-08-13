"""
Fraud Detection Layer using Isolation Forest.
"""
import pandas as pd
from pathlib import Path
import joblib
from sklearn.ensemble import IsolationForest
import json

def setup_directories() -> tuple[Path, Path, Path]:
    """
    Setup necessary directories.
    
    Returns:
        tuple[Path, Path, Path]: Processed data, models, and reports paths.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_data_path = project_root / 'data' / 'processed'
    models_path = project_root / 'models'
    reports_path = project_root / 'reports'
    return processed_data_path, models_path, reports_path

def train_isolation_forest() -> None:
    """
    Train Isolation Forest model on full original preprocessed dataset.
    """
    processed_data_path, models_path, reports_path = setup_directories()
    
    train_df = pd.read_csv(processed_data_path / 'train_raw.csv')
    test_df = pd.read_csv(processed_data_path / 'test_raw.csv')
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    fraud_features = [
        'RevolvingUtilizationOfUnsecuredLines',
        'NumberOfTimes90DaysLate',
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTime60-89DaysPastDueNotWorse',
        'DebtRatio'
    ]
    
    print("Training Isolation Forest...")
    iso_forest = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    iso_forest.fit(full_df[fraud_features])
    joblib.dump(iso_forest, models_path / 'isolation_forest.pkl')
    
    print("Evaluating Fraud Overlap...")
    test_df['fraud_score'] = iso_forest.decision_function(test_df[fraud_features])
    preds = iso_forest.predict(test_df[fraud_features])
    test_df['is_fraud_risk'] = (preds == -1).astype(int)
    
    total_fraud_flags = test_df['is_fraud_risk'].sum()
    fraud_rate = (total_fraud_flags / len(test_df)) * 100
    
    fraud_and_default = test_df[(test_df['is_fraud_risk'] == 1) & (test_df['SeriousDlqin2yrs'] == 1)]
    if total_fraud_flags > 0:
        overlap_pct = (len(fraud_and_default) / total_fraud_flags) * 100
    else:
        overlap_pct = 0
        
    analysis = {
        'fraud_rate_pct': float(fraud_rate),
        'overlap_with_default_pct': float(overlap_pct),
        'top_fraud_features': fraud_features
    }
    
    with open(reports_path / 'fraud_analysis.json', 'w') as f:
        json.dump(analysis, f, indent=4)
        
    test_df.to_csv(processed_data_path / 'test_with_fraud.csv', index=False)
    print("Fraud detection complete.")

if __name__ == "__main__":
    train_isolation_forest()
