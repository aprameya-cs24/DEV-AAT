import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# 1. Load Primary & Supplementary Datasets
# -------------------------------------------------------------------------
# Primary: CGWB Ground Water Levels
df_gw = pd.read_csv("cgwb-changes-in-depth-to-water-level.csv")  #

# Supplementary: India Agricultural Crop Production
df_crop = pd.read_csv("India Agriculture Crop Production.csv")  #

# -------------------------------------------------------------------------
# 2. Key Alignment & Data Standardization
# -------------------------------------------------------------------------
# Extract Year from groundwater 'date' column
df_gw["date"] = pd.to_datetime(df_gw["date"], errors="coerce")
df_gw["Year"] = df_gw["date"].dt.year  #[cite: 1]

# Parse agricultural 'Year' feature into an integer year representation
df_crop["Year"] = (
    pd.to_datetime(df_crop["Year"], errors="coerce").dt.year.fillna(
        df_crop["Year"]
    )
)  #

# Normalize state and district string columns (strip whitespace & lowercase)
df_gw["state_clean"] = df_gw["state_name"].astype(str).str.strip().str.lower()
df_gw["district_clean"] = (
    df_gw["district_name"].astype(str).str.strip().str.lower()
)  #[cite: 1]

df_crop["state_clean"] = df_crop["State"].astype(str).str.strip().str.lower()
df_crop["district_clean"] = (
    df_crop["District"].astype(str).str.strip().str.lower()
)  #[cite: 2]

# -------------------------------------------------------------------------
# 3. Dataset Merging (Spatial & Temporal Join)
# -------------------------------------------------------------------------
# Merge datasets using a left join on state, district, and year
df_merged = pd.merge(
    df_gw,
    df_crop,
    left_on=["state_clean", "district_clean", "Year"],
    right_on=["state_clean", "district_clean", "Year"],
    how="left",
    suffixes=("_gw", "_crop"),
)  #

print("=== DATASET SHAPES ===")
print(f"Groundwater Dataset: {df_gw.shape}")
print(f"Agriculture Dataset: {df_crop.shape}")
print(f"Merged Dataset: {df_merged.shape}\n")


# -------------------------------------------------------------------------
# 4. Summary Statistics Calculation Function
# -------------------------------------------------------------------------
def calculate_summary_stats(df, columns):
  """Calculates Sample Size, Mean, Median, Q1, Q3, IQR, Absolute Min, Absolute Max,

  Standard Deviation, and Missing Values for numerical columns.
  """
  results = []

  for col in columns:
    if col not in df.columns:
      continue

    series = df[col].dropna()
    missing_cnt = df[col].isna().sum()
    missing_pct = (missing_cnt / len(df)) * 100

    n_sample = len(series)
    mean_val = series.mean()
    median_val = series.median()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr_val = q3 - q1
    min_val = series.min()
    max_val = series.max()
    std_val = series.std()

    results.append({
        "Variable": col,
        "Sample Size (N)": n_sample,
        "Missing Count": missing_cnt,
        "Missing (%)": round(missing_pct, 2),
        "Mean": round(mean_val, 4),
        "Median": round(median_val, 4),
        "Std Dev": round(std_val, 4),
        "Q1 (25%)": round(q1, 4),
        "Q3 (75%)": round(q3, 4),
        "IQR Spread": round(iqr_val, 4),
        "Absolute Min": round(min_val, 4),
        "Absolute Max": round(max_val, 4),
    })

  return pd.DataFrame(results)


# -------------------------------------------------------------------------
# 5. Execute Summary Statistics Extraction
# -------------------------------------------------------------------------
target_cols = [
    "currentlevel",
    "level_diff",  # Groundwater metrics[cite: 1]
    "Area",
    "Production",
    "Yield",  # Crop statistics metrics[cite: 2]
]

summary_stats_table = calculate_summary_stats(df_merged, target_cols)

# Display tabular summary statistics
print("=== COMPREHENSIVE SUMMARY STATISTICS TABLE ===")
print(summary_stats_table.to_string(index=False))
