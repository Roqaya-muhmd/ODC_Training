import pandas as pd


def load_data(file_path):
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(file_path)


def get_basic_info(df):
    """Return basic information about the dataset."""
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "column_names": df.columns.tolist(),
        "missing_values": df.isnull().sum().to_dict()
    }


def get_summary_statistics(df):
    """Return summary statistics for numerical columns."""
    return df.describe().to_dict()


def get_column_statistics(df, column):
    """Return statistics for a specific numerical column."""
    if column not in df.columns:
        return f"Column '{column}' does not exist."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    return {
        "mean": df[column].mean(),
        "median": df[column].median(),
        "min": df[column].min(),
        "max": df[column].max()
    }
    
def compare_groups(df, group_column, value_column):
    """Compare a numerical column across groups."""
    
    if group_column not in df.columns:
        return f"Column '{group_column}' does not exist."
    
    if value_column not in df.columns:
        return f"Column '{value_column}' does not exist."
    
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return f"'{value_column}' must be numerical."
    
    return (
        df.groupby(group_column)[value_column]
        .agg(["count", "mean", "median", "min", "max"])
        .round(2)
        .to_dict("index")
    )


def get_correlations(df):
    """Return correlations between numerical columns."""
    
    numerical_df = df.select_dtypes(include="number")
    
    if numerical_df.empty:
        return {}
    
    return numerical_df.corr().round(2).to_dict()