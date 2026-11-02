from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# Find the project folder containing this script
project_folder = Path(__file__).resolve().parent

# Set up the data and output folders
raw_file = project_folder / "data" / "raw" / "mtsamples.csv"
processed_folder = project_folder / "data" / "processed"
reports_folder = project_folder / "reports"

processed_folder.mkdir(parents=True, exist_ok=True)
reports_folder.mkdir(parents=True, exist_ok=True)

# Stop with a clear error if the CSV isn't at the expected location
if not raw_file.exists():
    raise FileNotFoundError(f"Could not find the CSV here: {raw_file}")

# Read the CSV into a pandas DataFrame
df = pd.read_csv(raw_file)

# Tidy up any accidental spaces around column names
df.columns = df.columns.str.strip()

TEXT_COL = "transcription"
LABEL_COL = "medical_specialty"

print("Original rows and columns:", df.shape)
print("\nColumn names:", df.columns.tolist())

# Check that the two columns needed for this project exist
if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
    raise ValueError(
        f"Expected columns '{TEXT_COL}' and '{LABEL_COL}'. "
        f"Found: {df.columns.tolist()}"
    )

# Inspect column types and missing-value counts
print("\nColumn information:")
df.info()

print("\nMissing values in each column:")
print(df.isna().sum())

# Identify rows with blank text or blank labels
text = df[TEXT_COL].fillna("").astype(str).str.strip()
label = df[LABEL_COL].fillna("").astype(str).str.strip()

valid_rows = (text != "") & (label != "")
blank_rows = int((~valid_rows).sum())

# Create a working copy containing only rows with both text and a label
df_clean = df.loc[valid_rows].copy()
df_clean[TEXT_COL] = text.loc[valid_rows]
df_clean[LABEL_COL] = label.loc[valid_rows]

print("\nRows with blank text or label:", blank_rows)
print("Rows remaining after removing blank text or labels:", len(df_clean))

# Create normalized text for duplicate checking.
# Lowercasing and standardizing spaces lets us spot copies
# that differ only in capitalization or spacing.
df_clean["_normalized_text"] = (
    df_clean[TEXT_COL]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# Check whether the same normalized transcription has conflicting labels
labels_per_text = df_clean.groupby("_normalized_text")[LABEL_COL].nunique()
conflicting_texts = set(labels_per_text[labels_per_text > 1].index)

print("\nDistinct transcriptions assigned to multiple specialties:",
      len(conflicting_texts))

# Save conflicting cases for inspection, if any are found
if conflicting_texts:
    conflict_rows = df_clean[
        df_clean["_normalized_text"].isin(conflicting_texts)
    ].copy()

    conflict_rows.to_csv(
        reports_folder / "duplicate_label_conflicts.csv",
        index=False
    )

    # Leave these ambiguous notes out of the cleaned working copy.
    # Keeping one at random could give a note the wrong label.
    df_clean = df_clean[
        ~df_clean["_normalized_text"].isin(conflicting_texts)
    ].copy()

# Remove repeated notes, keeping one copy of each
rows_before_deduplication = len(df_clean)

df_clean = df_clean.drop_duplicates(
    subset="_normalized_text",
    keep="first"
).copy()

duplicate_rows_removed = rows_before_deduplication - len(df_clean)

# Remove the temporary helper column
df_clean = df_clean.drop(columns="_normalized_text")

print("Duplicate rows removed:", duplicate_rows_removed)
print("Rows in cleaned working copy:", len(df_clean))

# Count examples for each specialty
label_counts = df_clean[LABEL_COL].value_counts()

label_summary = label_counts.rename("count").to_frame()
label_summary["percent"] = (
    label_summary["count"] / label_summary["count"].sum() * 100
).round(2)

print("\nExamples per specialty:")
print(label_summary)

print("\nNumber of specialties:", len(label_summary))
print("Smallest specialty group:", label_counts.min())
print("Largest specialty group:", label_counts.max())

# Save the counts and cleaned data
label_summary.to_csv(reports_folder / "specialty_counts.csv")
df_clean.to_csv(
    processed_folder / "mtsamples_day1_clean.csv",
    index=False
)

# Plot the 15 most common specialties
top_15 = label_counts.head(15).sort_values()

top_15.plot(kind="barh", figsize=(10, 7))
plt.title("15 Most Frequent Medical Specialties")
plt.xlabel("Number of transcriptions")
plt.ylabel("Specialty")
plt.tight_layout()
plt.savefig(reports_folder / "specialty_counts.png", dpi=150)
plt.close()

print("\nSaved cleaned data to:", processed_folder / "mtsamples_day1_clean.csv")
print("Saved specialty counts to:", reports_folder / "specialty_counts.csv")
print("Saved chart to:", reports_folder / "specialty_counts.png")