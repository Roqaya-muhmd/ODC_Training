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


def get_outliers(df, column):
    """
    Count outliers in a numerical column using the IQR method:
    anything below Q1 - 1.5*IQR or above Q3 + 1.5*IQR is an outlier.
    """
    if column not in df.columns:
        return f"Column '{column}' does not exist."

    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numerical."

    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outlier_count = df[(df[column] < lower_bound) | (df[column] > upper_bound)].shape[0]

    return {
        "outlier_count": outlier_count,
        "lower_bound": round(lower_bound, 2),
        "upper_bound": round(upper_bound, 2)
    }


def get_top_correlations(df, top_n=3):
    """
    Returns the top_n strongest correlations (positive or negative) between
    different numerical column pairs, excluding a column's correlation with itself.
    """
    numerical_df = df.select_dtypes(include="number")

    if numerical_df.shape[1] < 2:
        return []

    corr_matrix = numerical_df.corr().abs()

    pairs = []
    columns = corr_matrix.columns
    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col_a, col_b = columns[i], columns[j]
            pairs.append((col_a, col_b, corr_matrix.loc[col_a, col_b]))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_n]


def summarize_dataset(df, dataset_name="uploaded_dataset"):
    """
    Turn a DataFrame into a single structured text block, so it can be
    embedded and stored in the SAME FAISS vector store as the PDFs/TXT
    knowledge base, instead of being handled by separate pandas calls.
    """
    info = get_basic_info(df)

    lines = [f"Uploaded Dataset Summary: {dataset_name}"]
    lines.append(f"Number of rows: {info['rows']}")
    lines.append(f"Number of columns: {info['columns']}")

    lines.append("\nColumns:")
    for col in df.columns:
        lines.append(f"- {col} -> {df[col].dtype}")

    lines.append("\nMissing values:")
    for col, missing in info["missing_values"].items():
        lines.append(f"- {col} -> {missing}")

    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) > 0:
        lines.append("\nStatistics:")
        for col in numeric_cols:
            stats = get_column_statistics(df, col)
            lines.append(f"{col}:")
            lines.append(f"  Mean: {stats['mean']:.3f}")
            lines.append(f"  Median: {stats['median']:.3f}")
            lines.append(f"  Min: {stats['min']:.3f}")
            lines.append(f"  Max: {stats['max']:.3f}")

        lines.append("\nOutliers (IQR method):")
        for col in numeric_cols:
            outlier_info = get_outliers(df, col)
            lines.append(
                f"- {col}: {outlier_info['outlier_count']} outliers "
                f"(normal range: {outlier_info['lower_bound']} to {outlier_info['upper_bound']})"
            )

        top_corrs = get_top_correlations(df, top_n=3)
        if top_corrs:
            lines.append("\nStrongest correlations between numerical columns:")
            for col_a, col_b, value in top_corrs:
                lines.append(f"- {col_a} and {col_b}: {value:.2f}")

    categorical_cols = df.select_dtypes(exclude="number").columns
    if len(categorical_cols) > 0:
        lines.append("\nCategorical columns (top values):")
        for col in categorical_cols:
            top = df[col].value_counts().head(3).to_dict()
            lines.append(f"{col}: {top}")

    return "\n".join(lines)
