"""
Exploratory Data Analysis for the Give Me Some Credit dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def setup_directories() -> tuple[Path, Path]:
    """
    Setup necessary directories for data and reports.
    
    Returns:
        tuple[Path, Path]: Data directory and reports directory paths.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    data_path = project_root / 'data' / 'raw'
    reports_path = project_root / 'reports' / 'eda'
    
    data_path.mkdir(parents=True, exist_ok=True)
    reports_path.mkdir(parents=True, exist_ok=True)
    
    return data_path, reports_path

def load_data(data_path: Path) -> pd.DataFrame:
    """
    Load the training dataset.
    
    Args:
        data_path (Path): Path to the raw data directory.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    file_path = data_path / 'cs-training.csv'
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
    
    df = pd.read_csv(file_path, index_col=0) if 'Unnamed: 0' in pd.read_csv(file_path, nrows=1).columns else pd.read_csv(file_path)
    return df

def basic_info(df: pd.DataFrame) -> None:
    """
    Print basic information about the dataset.
    
    Args:
        df (pd.DataFrame): The dataset.
    """
    print("=== Dataset Shape ===")
    print(df.shape)
    print("\n=== Data Types ===")
    print(df.dtypes)
    
    print("\n=== Missing Values ===")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Count': missing, 'Percentage': missing_pct})
    print(missing_df[missing_df['Count'] > 0])
    
    print("\n=== Class Distribution (SeriousDlqin2yrs) ===")
    print(df['SeriousDlqin2yrs'].value_counts())
    print(df['SeriousDlqin2yrs'].value_counts(normalize=True) * 100)

def plot_class_distribution(df: pd.DataFrame, reports_path: Path) -> None:
    """
    Plot and save the class distribution bar chart.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='SeriousDlqin2yrs')
    plt.title('Class Distribution (Default vs Non-Default)')
    plt.xlabel('SeriousDlqin2yrs (1 = Default)')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(reports_path / 'class_distribution.png')
    plt.close()

def plot_missing_values(df: pd.DataFrame, reports_path: Path) -> None:
    """
    Plot and save horizontal bar chart of missing value percentages.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save the plot.
    """
    missing_pct = (df.isnull().sum() / len(df)) * 100
    missing_pct = missing_pct[missing_pct > 0].sort_values()
    
    if len(missing_pct) > 0:
        plt.figure(figsize=(10, 6))
        missing_pct.plot(kind='barh')
        plt.title('Percentage of Missing Values per Feature')
        plt.xlabel('Percentage (%)')
        plt.tight_layout()
        plt.savefig(reports_path / 'missing_values.png')
        plt.close()

def plot_feature_distributions(df: pd.DataFrame, reports_path: Path) -> None:
    """
    Plot and save histograms for all 10 features in a 2x5 grid.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save the plot.
    """
    features = [c for c in df.columns if c != 'SeriousDlqin2yrs']
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, feature in enumerate(features[:10]):
        sns.histplot(df[feature], bins=30, ax=axes[i], kde=True)
        axes[i].set_title(feature[:20] + '..')
        
    plt.tight_layout()
    plt.savefig(reports_path / 'feature_distributions.png')
    plt.close()

def plot_correlation_heatmap(df: pd.DataFrame, reports_path: Path) -> None:
    """
    Plot and save Seaborn correlation heatmap.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save the plot.
    """
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', cbar=True, square=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(reports_path / 'correlation_heatmap.png')
    plt.close()

def plot_default_rate_by_age(df: pd.DataFrame, reports_path: Path) -> None:
    """
    Plot and save line chart of default rate binned by age decade.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save the plot.
    """
    df_copy = df.copy()
    bins = [20, 30, 40, 50, 60, 120]
    labels = ['20s', '30s', '40s', '50s', '60s+']
    df_copy['AgeDecade'] = pd.cut(df_copy['age'], bins=bins, labels=labels, right=False)
    
    default_rate = df_copy.groupby('AgeDecade', observed=False)['SeriousDlqin2yrs'].mean()
    
    plt.figure(figsize=(8, 6))
    default_rate.plot(kind='line', marker='o')
    plt.title('Default Rate by Age Decade')
    plt.xlabel('Age Decade')
    plt.ylabel('Default Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(reports_path / 'default_rate_by_age.png')
    plt.close()

def plot_default_rate_by_utilization(df: pd.DataFrame, reports_path: Path) -> None:
    """
    Plot and save default rate by RevolvingUtilization decile.
    
    Args:
        df (pd.DataFrame): The dataset.
        reports_path (Path): Path to save the plot.
    """
    df_copy = df.copy()
    df_copy['UtilDecile'] = pd.qcut(df_copy['RevolvingUtilizationOfUnsecuredLines'], q=10, duplicates='drop')
    
    default_rate = df_copy.groupby('UtilDecile', observed=False)['SeriousDlqin2yrs'].mean()
    
    plt.figure(figsize=(10, 6))
    default_rate.plot(kind='bar', color='skyblue')
    plt.title('Default Rate by Revolving Utilization Decile')
    plt.xlabel('Utilization Decile')
    plt.ylabel('Default Rate')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(reports_path / 'default_rate_by_utilization.png')
    plt.close()

def print_key_insights(df: pd.DataFrame) -> None:
    """
    Print key insights extracted from EDA.
    
    Args:
        df (pd.DataFrame): The dataset.
    """
    print("\n=== Key Insights ===")
    
    corr = df.corr()['SeriousDlqin2yrs'].drop('SeriousDlqin2yrs').sort_values(ascending=False, key=abs)
    print("Top 3 features correlated with default:")
    print(corr.head(3))
    
    overall_default_rate = df['SeriousDlqin2yrs'].mean() * 100
    print(f"\nOverall Default Rate: {overall_default_rate:.2f}%")
    
    print("\nMissing Value Summary:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    for col, pct in missing_pct[missing_pct > 0].items():
        print(f" - {col}: {pct:.2f}% missing")

def main() -> None:
    """
    Main function to run EDA pipeline.
    """
    data_path, reports_path = setup_directories()
    
    print("Loading data...")
    df = load_data(data_path)
    
    print("Generating basic info...")
    basic_info(df)
    
    print("Generating plots...")
    plot_class_distribution(df, reports_path)
    plot_missing_values(df, reports_path)
    plot_feature_distributions(df, reports_path)
    plot_correlation_heatmap(df, reports_path)
    plot_default_rate_by_age(df, reports_path)
    plot_default_rate_by_utilization(df, reports_path)
    
    print_key_insights(df)
    print(f"EDA complete. Plots saved to {reports_path}")

if __name__ == "__main__":
    main()
