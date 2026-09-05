"""
Data preprocessing script applying imputation, outlier capping, and WoE binning.
Fixed: split-first ordering, saved cap bounds, custom bins for count features.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

CUSTOM_BINS = {
    "NumberOfTime30-59DaysPastDueNotWorse": [-float("inf"), 0.5, 1.5, 2.5, float("inf")],
    "NumberOfTimes90DaysLate":              [-float("inf"), 0.5, 1.5, 2.5, float("inf")],
    "NumberOfTime60-89DaysPastDueNotWorse": [-float("inf"), 0.5, 1.5, 2.5, float("inf")],
    "NumberRealEstateLoansOrLines":         [-float("inf"), 0.5, 1.5, 2.5, 3.5, float("inf")],
    "NumberOfOpenCreditLinesAndLoans":      [-float("inf"), 2.5, 5.5, 8.5, 12.5, float("inf")],
    "NumberOfDependents":                   [-float("inf"), 0.5, 1.5, 2.5, float("inf")],
}

def setup_directories():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_path = project_root / "data" / "raw"
    processed_data_path = project_root / "data" / "processed"
    reports_path = project_root / "reports"
    processed_data_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    return raw_data_path, processed_data_path, reports_path

def load_data(raw_data_path):
    file_path = raw_data_path / "cs-training.csv"
    first_row = pd.read_csv(file_path, nrows=1)
    df = pd.read_csv(file_path, index_col=0) if "Unnamed: 0" in first_row.columns else pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    df = df.copy()
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)
    df["MonthlyIncome"] = df.groupby("NumberOfDependents")["MonthlyIncome"].transform(
        lambda x: x.fillna(x.median())
    )
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    return df

def fit_and_apply_caps(df_train, df_test):
    df_train = df_train.copy()
    df_test  = df_test.copy()
    features = [c for c in df_train.columns if c != "SeriousDlqin2yrs"]
    cap_bounds = {}
    for col in features:
        Q1  = df_train[col].quantile(0.25)
        Q3  = df_train[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower = float(Q1 - 1.5 * IQR)
            upper = float(Q3 + 1.5 * IQR)
            cap_bounds[col] = (lower, upper)
            df_train[col] = np.clip(df_train[col], lower, upper)
            df_test[col]  = np.clip(df_test[col],  lower, upper)
        else:
            cap_bounds[col] = None
    return df_train, df_test, cap_bounds

def _get_woe_map(df, feature, edges):
    tmp = df.copy()
    tmp["_bin"] = pd.cut(tmp[feature], bins=edges, include_lowest=True)
    g = tmp.groupby("_bin", observed=False)["SeriousDlqin2yrs"].agg(["count","sum"])
    g.columns = ["count","events"]
    g["non_events"] = g["count"] - g["events"]
    te = max(g["events"].sum(), 1)
    tn = max(g["non_events"].sum(), 1)
    g["ed"] = np.maximum(g["events"] / te, 1e-4)
    g["nd"] = np.maximum(g["non_events"] / tn, 1e-4)
    g["woe"] = np.log(g["ed"] / g["nd"])
    g["iv"]  = (g["ed"] - g["nd"]) * g["woe"]
    return g["woe"].to_dict(), g["iv"].sum()

def apply_woe_binning(df_train, df_test, reports_path, processed_data_path):
    df_tr = df_train.copy()
    df_te = df_test.copy()
    features = [c for c in df_tr.columns if c != "SeriousDlqin2yrs"]
    iv_results   = []
    woe_mappings = {}
    for feature in features:
        if feature in CUSTOM_BINS:
            edges = CUSTOM_BINS[feature]
        else:
            try:
                _, edges = pd.qcut(df_tr[feature], q=10, retbins=True, duplicates="drop")
            except ValueError:
                _, edges = pd.cut(df_tr[feature], bins=10, retbins=True)
            edges = list(edges)
            if len(edges) <= 1:
                edges = [-float("inf"), float("inf")]
            else:
                edges[0]  = -float("inf")
                edges[-1] = float("inf")

        woe_map, iv_total = _get_woe_map(df_tr, feature, edges)

        if   iv_total < 0.02: power = "Useless"
        elif iv_total < 0.1:  power = "Weak"
        elif iv_total < 0.3:  power = "Medium"
        elif iv_total <= 0.5: power = "Strong"
        else:                 power = "Suspicious"
        iv_results.append({"Feature": feature, "IV": iv_total, "Predictive_Power": power})

        df_tr[feature] = pd.cut(df_tr[feature], bins=edges, include_lowest=True).map(woe_map).astype(float).fillna(0.0)
        df_te[feature] = pd.cut(df_te[feature], bins=edges, include_lowest=True).map(woe_map).astype(float).fillna(0.0)

        woe_mappings[feature] = {"edges": edges, "woe_map": woe_map}

    iv_df = pd.DataFrame(iv_results).sort_values("IV", ascending=False)
    iv_df.to_csv(reports_path / "iv_table.csv", index=False)
    print("IV Table saved to reports/iv_table.csv")
    print(iv_df.to_string(index=False))
    joblib.dump(woe_mappings, processed_data_path / "woe_mappings.pkl")
    return df_tr, df_te

def main():
    raw_data_path, processed_data_path, reports_path = setup_directories()

    print("Loading data...")
    df = load_data(raw_data_path)

    print("Handling missing values...")
    df = handle_missing_values(df)

    print("Splitting data (80/20) BEFORE any transformation to prevent leakage...")
    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["SeriousDlqin2yrs"], random_state=42
    )

    print("Capping outliers using TRAIN statistics only...")
    df_train, df_test, cap_bounds = fit_and_apply_caps(df_train, df_test)
    joblib.dump(cap_bounds, processed_data_path / "cap_bounds.pkl")
    print("Cap bounds saved.")

    df_train.to_csv(processed_data_path / "train_raw.csv", index=False)
    df_test.to_csv(processed_data_path / "test_raw.csv",  index=False)

    print("Applying WoE binning (fit on train, applied to both)...")
    df_train_woe, df_test_woe = apply_woe_binning(df_train, df_test, reports_path, processed_data_path)

    print("Saving WoE-transformed data...")
    df_train_woe.to_csv(processed_data_path / "train_woe.csv", index=False)
    df_test_woe.to_csv(processed_data_path / "test_woe.csv",  index=False)

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
