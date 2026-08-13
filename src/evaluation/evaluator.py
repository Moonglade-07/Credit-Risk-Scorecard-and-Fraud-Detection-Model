"""
Model Evaluation script.
Computes metrics and generates evaluation plots for all 4 models.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, brier_score_loss, precision_score,
    recall_score, f1_score
)
from sklearn.calibration import calibration_curve
import shap
import json

def setup_directories() -> tuple[Path, Path, Path]:
    project_root = Path(__file__).resolve().parent.parent.parent
    processed_data_path = project_root / 'data' / 'processed'
    models_path = project_root / 'models'
    reports_path = project_root / 'reports'
    return processed_data_path, models_path, reports_path

def load_models_and_data(processed_data_path: Path, models_path: Path):
    test_df = pd.read_csv(processed_data_path / 'test_woe.csv')
    X_test = test_df.drop('SeriousDlqin2yrs', axis=1)
    y_test = test_df['SeriousDlqin2yrs']
    
    models = {
        'Logistic Regression': joblib.load(models_path / 'logistic_regression.pkl'),
        'Decision Tree': joblib.load(models_path / 'decision_tree.pkl'),
        'XGBoost': joblib.load(models_path / 'xgboost_base.pkl'),
        'XGBoost Tuned': joblib.load(models_path / 'xgboost_tuned.pkl')
    }
    return X_test, y_test, models

def ks_statistic(y_true, y_prob):
    df = pd.DataFrame({'y_true': y_true, 'y_prob': y_prob})
    df = df.sort_values('y_prob', ascending=False)
    
    events = df['y_true'].sum()
    non_events = len(df) - events
    
    df['cum_events'] = df['y_true'].cumsum() / events
    df['cum_non_events'] = (1 - df['y_true']).cumsum() / non_events
    
    ks = np.abs(df['cum_events'] - df['cum_non_events']).max()
    return ks

def evaluate_models(X_test, y_test, models, reports_path):
    metrics = {}
    preds_dict = {}
    
    for name, model in models.items():
        preds = model.predict_proba(X_test)[:, 1]
        preds_dict[name] = preds
        
        y_pred = (preds > 0.5).astype(int)
        
        auc = roc_auc_score(y_test, preds)
        gini = 2 * auc - 1
        ks = ks_statistic(y_test, preds)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, preds)
        
        metrics[name] = {
            'AUC-ROC': float(auc),
            'Gini': float(gini),
            'KS Statistic': float(ks),
            'Precision': float(prec),
            'Recall': float(rec),
            'F1': float(f1),
            'Brier Score': float(brier)
        }
        
    with open(reports_path / 'model_comparison.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    return preds_dict, metrics

def plot_roc_curves(y_test, preds_dict, reports_path):
    plt.figure(figsize=(10, 8))
    for name, preds in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_test, preds)
        auc = roc_auc_score(y_test, preds)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_path / 'roc_curves.png')
    plt.close()

def plot_pr_curves(y_test, preds_dict, reports_path):
    plt.figure(figsize=(10, 8))
    for name, preds in preds_dict.items():
        prec, rec, _ = precision_recall_curve(y_test, preds)
        plt.plot(rec, prec, label=name)
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_path / 'pr_curves.png')
    plt.close()

def plot_ks_statistic(y_test, best_preds, reports_path):
    df = pd.DataFrame({'y_true': y_test, 'y_prob': best_preds})
    df = df.sort_values('y_prob', ascending=False)
    
    events = df['y_true'].sum()
    non_events = len(df) - events
    
    df['cum_events'] = df['y_true'].cumsum() / events
    df['cum_non_events'] = (1 - df['y_true']).cumsum() / non_events
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['y_prob'].values, df['cum_events'].values, label='Cumulative Defaults')
    plt.plot(df['y_prob'].values, df['cum_non_events'].values, label='Cumulative Non-Defaults')
    
    plt.xlim(1, 0)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Cumulative Proportion')
    plt.title('KS Statistic Plot (Best Model)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_path / 'ks_plot.png')
    plt.close()

def plot_confusion_matrices(y_test, preds_dict, reports_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (name, preds) in enumerate(preds_dict.items()):
        y_pred = (preds > 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False)
        axes[i].set_title(f'Confusion Matrix: {name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
        
    plt.tight_layout()
    plt.savefig(reports_path / 'confusion_matrices.png')
    plt.close()

def plot_feature_importance(xgb_tuned, X_cols, reports_path):
    importance = xgb_tuned.feature_importances_
    df = pd.DataFrame({'Feature': X_cols, 'Importance': importance})
    df = df.sort_values('Importance', ascending=True).tail(15)
    
    plt.figure(figsize=(10, 8))
    plt.barh(df['Feature'], df['Importance'], color='skyblue')
    plt.xlabel('Importance (Gain)')
    plt.title('Top 15 Features by XGBoost Gain')
    plt.tight_layout()
    plt.savefig(reports_path / 'feature_importance_xgb.png')
    plt.close()

def plot_shap_summary(xgb_tuned, X_test, reports_path):
    X_sample = shap.utils.sample(X_test, 2000) if len(X_test) > 2000 else X_test
    explainer = shap.TreeExplainer(xgb_tuned)
    shap_values = explainer.shap_values(X_sample)
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(reports_path / 'shap_summary.png')
    plt.close()

def plot_calibration_curve(y_test, best_preds, reports_path):
    prob_true, prob_pred = calibration_curve(y_test, best_preds, n_bins=10)
    
    plt.figure(figsize=(10, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='o', label='Best Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_path / 'calibration_curve.png')
    plt.close()

def main():
    processed_data_path, models_path, reports_path = setup_directories()
    
    print("Loading models and test data...")
    X_test, y_test, models = load_models_and_data(processed_data_path, models_path)
    
    print("Evaluating models...")
    preds_dict, metrics = evaluate_models(X_test, y_test, models, reports_path)
    
    print("Plotting ROC Curves...")
    plot_roc_curves(y_test, preds_dict, reports_path)
    
    print("Plotting PR Curves...")
    plot_pr_curves(y_test, preds_dict, reports_path)
    
    print("Plotting Confusion Matrices...")
    plot_confusion_matrices(y_test, preds_dict, reports_path)
    
    best_model_name = max(metrics, key=lambda k: metrics[k]['AUC-ROC'])
    print(f"Best Model is {best_model_name}")
    best_preds = preds_dict[best_model_name]
    
    print("Plotting KS Statistic...")
    plot_ks_statistic(y_test, best_preds, reports_path)
    
    print("Plotting Calibration Curve...")
    plot_calibration_curve(y_test, best_preds, reports_path)
    
    print("Plotting Feature Importance...")
    plot_feature_importance(models['XGBoost Tuned'], X_test.columns, reports_path)
    
    print("Plotting SHAP Summary...")
    plot_shap_summary(models['XGBoost Tuned'], X_test, reports_path)
    
    print("Evaluation complete.")

if __name__ == "__main__":
    main()
