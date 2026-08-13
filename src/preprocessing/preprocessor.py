"""
Data preprocessing script applying imputation, outlier capping, and WoE binning.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import joblib

def setup_directories() -> tuple[Path, Path, Path]:
    """
    Setup necessary directories for data and reports.
    
    Returns:
        tuple[Path, Path, Path]: Raw data, processed data, and reports paths.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_data_path = project_root / 'data' / 'raw'
    processed_data_path = project_root / 'data' / 'processed'
    reports_path = project_root / 'reports'
    
    processed_data_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    return raw_data_path, processed_data_path, reports_path

def load_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Load the training dataset.
    
    Args:
        raw_data_path (Path): Path to the raw data directory.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = raw_data_path / 'cs-training.csv'
    df = pd.read_csv(file_path, index_col=0) if 'Unnamed: 0' in pd.read_csv(file_path, nrows=1).columns else pd.read_csv(file_path)
    return df

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df (pd.DataFrame): The dataset.
        
    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    df = df.copy()
    
    # NumberOfDependents: fill with 0
    df['NumberOfDependents'] = df['NumberOfDependents'].fillna(0)
    
    # MonthlyIncome: fill with median grouped by NumberOfDependents
    df['MonthlyIncome'] = df.groupby('NumberOfDependents')['MonthlyIncome'].transform(lambda x: x.fillna(x.median()))
    # If any still missing, fill with overall median
    df['MonthlyIncome'] = df['MonthlyIncome'].fillna(df['MonthlyIncome'].median())
    
    return df

def cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cap outliers using IQR method (1.5x IQR) for all numeric features.
    
    Args:
        df (pd.DataFrame): The dataset.
        
    Returns:
        pd.DataFrame: Dataset with outliers capped.
    """
    df = df.copy()
    features = [c for c in df.columns if c != 'SeriousDlqin2yrs']
    
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        if IQR > 0:
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    return df

def apply_woe_binning(df: pd.DataFrame, reports_path: Path, processed_data_path: Path) -> pd.DataFrame:
    """
    Apply WoE binning to features and save IV table.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save reports.
        processed_data_path (Path): Path to save mapping logic.
        
    Returns:
        pd.DataFrame: Dataset transformed with WoE values.
    """
    df = df.copy()
    features = [c for c in df.columns if c != 'SeriousDlqin2yrs']
    
    iv_results = []
    woe_mappings = {}
    binning_edges = {}
    
    for feature in features:
        try:
            _, edges = pd.qcut(df[feature], q=10, retbins=True, duplicates='drop')
        except ValueError:
            _, edges = pd.cut(df[feature], bins=10, retbins=True)
            
        edges = list(edges)
        if len(edges) == 1:
            edges = [-np.inf, np.inf]
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
        binning_edges[feature] = edges
        
        df['temp_bin'] = pd.cut(df[feature], bins=edges)
        
        grouped = df.groupby('temp_bin', observed=False)['SeriousDlqin2yrs'].agg(['count', 'sum'])
        grouped['non_events'] = grouped['count'] - grouped['sum']
        grouped['events'] = grouped['sum']
        
        total_events = max(grouped['events'].sum(), 1)
        total_non_events = max(grouped['non_events'].sum(), 1)
        
        grouped['event_dist'] = np.maximum(grouped['events'] / total_events, 0.0001)
        grouped['non_event_dist'] = np.maximum(grouped['non_events'] / total_non_events, 0.0001)
        
        grouped['woe'] = np.log(grouped['event_dist'] / grouped['non_event_dist'])
        grouped['iv'] = (grouped['event_dist'] - grouped['non_event_dist']) * grouped['woe']
        
        iv_total = grouped['iv'].sum()
        woe_map = grouped['woe'].to_dict()
        
        power = "Suspicious"
        if iv_total < 0.02: power = "Useless"
        elif iv_total < 0.1: power = "Weak"
        elif iv_total < 0.3: power = "Medium"
        elif iv_total <= 0.5: power = "Strong"
            
        iv_results.append({'Feature': feature, 'IV': iv_total, 'Predictive_Power': power})
        
        df[feature] = df['temp_bin'].map(woe_map).astype(float)
        
        woe_mappings[feature] = {'edges': edges, 'woe_map': woe_map}
        
    df.drop(columns=['temp_bin'], inplace=True)
    
    iv_df = pd.DataFrame(iv_results).sort_values('IV', ascending=False)
    iv_df.to_csv(reports_path / 'iv_table.csv', index=False)
    print("IV Table saved to reports/iv_table.csv")
    
    joblib.dump(woe_mappings, processed_data_path / 'woe_mappings.pkl')
    
    return df

def main() -> None:
    """
    Main function to run preprocessing pipeline.
    """
    raw_data_path, processed_data_path, reports_path = setup_directories()
    
    print("Loading data...")
    df = load_data(raw_data_path)
    
    print("Handling missing values...")
    df = handle_missing_values(df)
    
    print("Capping outliers...")
    df = cap_outliers(df)
    
    print("Splitting data (80/20)...")
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['SeriousDlqin2yrs'], random_state=42)
    
    print("Applying WoE binning...")
    df_train_woe = apply_woe_binning(df_train, reports_path, processed_data_path)
    
    woe_mappings = joblib.load(processed_data_path / 'woe_mappings.pkl')
    df_test_woe = df_test.copy()
    features = [c for c in df_test.columns if c != 'SeriousDlqin2yrs']
    for feature in features:
        edges = woe_mappings[feature]['edges']
        woe_map = woe_mappings[feature]['woe_map']
        df_test_woe[feature] = pd.cut(df_test_woe[feature], bins=edges).map(woe_map).astype(float)
        df_test_woe[feature] = df_test_woe[feature].fillna(0)
    
    print("Saving processed data...")
    df_train_woe.to_csv(processed_data_path / 'train_woe.csv', index=False)
    df_test_woe.to_csv(processed_data_path / 'test_woe.csv', index=False)
    
    df_train.to_csv(processed_data_path / 'train_raw.csv', index=False)
    df_test.to_csv(processed_data_path / 'test_raw.csv', index=False)
    
    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
