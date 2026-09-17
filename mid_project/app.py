from src.data_analysis import load_data, get_basic_info

# Load the dataset
df = load_data("data/sample.csv")

# Get basic information
info = get_basic_info(df)

print("Dataset loaded successfully! ✅")
print("Rows:", info["rows"])
print("Columns:", info["columns"])
print("Column names:", info["column_names"])
print("Missing values:", info["missing_values"])