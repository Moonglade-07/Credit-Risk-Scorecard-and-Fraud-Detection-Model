"""
Credit Risk Scorecard modeling script.
Builds Logistic Regression, Decision Tree, XGBoost, and tuned XGBoost.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from xgboost import XGBClassifier
import optuna
import json

def setup_directories() -> tuple[Path, Path, Path]:
    """
    Setup necessary directories for data, models, and reports.
    
    Returns:
        tuple[Path, Path, Path]: Processed data, models, and reports paths.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_data_path = project_root / 'data' / 'processed'
    models_path = project_root / 'models'
    reports_path = project_root / 'reports'
    
    models_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    return processed_data_path, models_path, reports_path

def load_data(processed_data_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load WoE-transformed train and test sets.
    
    Args:
        processed_data_path (Path): Path to processed data.
        
    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Train and test datasets.
    """
    train_df = pd.read_csv(processed_data_path / 'train_woe.csv')
    test_df = pd.read_csv(processed_data_path / 'test_woe.csv')
    return train_df, test_df

def convert_to_score(prob: np.ndarray, offset: float = 600, factor: float = 28.8539) -> np.ndarray:
    """
    Convert probability to credit score.
    
    Args:
        prob (np.ndarray): Array of predicted default probabilities.
        offset (float): Base score.
        factor (float): PDO factor.
        
    Returns:
        np.ndarray: Array of credit scores.
    """
    prob = np.clip(prob, 1e-10, 1 - 1e-10)
    odds_good = (1 - prob) / prob
    score = offset + factor * np.log(odds_good)
    score = np.clip(score, 300, 850)
    return score

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, models_path: Path, reports_path: Path) -> LogisticRegression:
    """
    Train Logistic Regression baseline scorecard model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        models_path (Path): Path to save the model.
        reports_path (Path): Path to save reports.
        
    Returns:
        LogisticRegression: Trained model.
    """
    print("Training Model 1: Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    joblib.dump(lr, models_path / 'logistic_regression.pkl')
    
    prob = lr.predict_proba(X_train)[:, 1]
    scores = convert_to_score(prob)
    
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=50, color='skyblue', edgecolor='black')
    plt.title('Credit Score Distribution (Train Set)')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(reports_path / 'score_distribution.png')
    plt.close()
    
    return lr

def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series, models_path: Path, reports_path: Path) -> DecisionTreeClassifier:
    """
    Train Decision Tree model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        models_path (Path): Path to save the model.
        reports_path (Path): Path to save reports.
        
    Returns:
        DecisionTreeClassifier: Trained model.
    """
    print("Training Model 2: Decision Tree...")
    dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=100, class_weight='balanced', random_state=42)
    dt.fit(X_train, y_train)
    
    joblib.dump(dt, models_path / 'decision_tree.pkl')
    
    plt.figure(figsize=(25, 10))
    plot_tree(dt, feature_names=X_train.columns, class_names=['Good', 'Bad'], filled=True, max_depth=3)
    plt.title('Decision Tree (Depth truncated to 3 for visibility)')
    plt.tight_layout()
    plt.savefig(reports_path / 'decision_tree.png')
    plt.close()
    
    tree_rules = export_text(dt, feature_names=list(X_train.columns))
    with open(reports_path / 'decision_rules.txt', 'w') as f:
        f.write("Top decision rules (max depth 5):\n")
        lines = tree_rules.split('\n')
        for line in lines[:50]:
            f.write(line + '\n')
            
    return dt

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models_path: Path, reports_path: Path) -> XGBClassifier:
    """
    Train XGBoost model.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Validation features.
        y_test (pd.Series): Validation target.
        models_path (Path): Path to save the model.
        reports_path (Path): Path to save reports.
        
    Returns:
        XGBClassifier: Trained model.
    """
    print("Training Model 3: XGBoost...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=30
    )
    
    eval_set = [(X_train, y_train), (X_test, y_test)]
    xgb.fit(X_train, y_train, eval_set=eval_set, verbose=50)
    
    joblib.dump(xgb, models_path / 'xgboost_base.pkl')
    
    results = xgb.evals_result()
    epochs = len(results['validation_0']['auc'])
    x_axis = range(0, epochs)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, results['validation_0']['auc'], label='Train')
    plt.plot(x_axis, results['validation_1']['auc'], label='Test')
    plt.legend()
    plt.ylabel('AUC')
    plt.title('XGBoost Learning Curve')
    plt.tight_layout()
    plt.savefig(reports_path / 'xgb_learning_curve.png')
    plt.close()
    
    return xgb

def train_xgboost_optuna(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, models_path: Path) -> XGBClassifier:
    """
    Train XGBoost model with Optuna hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        X_test (pd.DataFrame): Validation features.
        y_test (pd.Series): Validation target.
        models_path (Path): Path to save the model.
        
    Returns:
        XGBClassifier: Tuned model.
    """
    print("Training Model 4: XGBoost with Optuna...")
    scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
    
    def objective_auc(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'eval_metric': 'auc',
            'early_stopping_rounds': 30
        }
        
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        from sklearn.metrics import roc_auc_score
        preds = clf.predict_proba(X_test)[:, 1]
        return roc_auc_score(y_test, preds)
        
    study = optuna.create_study(direction='maximize')
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective_auc, n_trials=30)
    
    best_params = study.best_params
    best_params['scale_pos_weight'] = scale_pos_weight
    best_params['random_state'] = 42
    
    with open(models_path / 'xgb_best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
        
    best_xgb = XGBClassifier(**best_params)
    best_xgb.fit(X_train, y_train)
    joblib.dump(best_xgb, models_path / 'xgboost_tuned.pkl')
    
    return best_xgb

def main() -> None:
    """
    Main function to run scorecard modeling.
    """
    processed_data_path, models_path, reports_path = setup_directories()
    
    print("Loading preprocessed data...")
    train_df, test_df = load_data(processed_data_path)
    
    X_train = train_df.drop('SeriousDlqin2yrs', axis=1)
    y_train = train_df['SeriousDlqin2yrs']
    X_test = test_df.drop('SeriousDlqin2yrs', axis=1)
    y_test = test_df['SeriousDlqin2yrs']
    
    train_logistic_regression(X_train, y_train, models_path, reports_path)
    train_decision_tree(X_train, y_train, models_path, reports_path)
    train_xgboost(X_train, y_train, X_test, y_test, models_path, reports_path)
    train_xgboost_optuna(X_train, y_train, X_test, y_test, models_path)
    
    print("Modeling complete.")

if __name__ == "__main__":
    main()
