from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------
# 1. Set up project folders and file paths
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent

raw_file = project_folder / "data" / "raw" / "mtsamples.csv"
processed_folder = project_folder / "data" / "processed"
reports_folder = project_folder / "reports"

processed_folder.mkdir(parents=True, exist_ok=True)
reports_folder.mkdir(parents=True, exist_ok=True)

if not raw_file.exists():
    raise FileNotFoundError(
        f"Original CSV was not found here:\n{raw_file}"
    )


# ------------------------------------------------------------
# 2. Read the original CSV and check its columns
# ------------------------------------------------------------

df = pd.read_csv(raw_file)
df.columns = df.columns.str.strip()

TEXT_COL = "transcription"
LABEL_COL = "medical_specialty"

required_columns = {TEXT_COL, LABEL_COL}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Columns found: {df.columns.tolist()}"
    )

# Keep only the model input and the label
df = df[[TEXT_COL, LABEL_COL]].copy()

print("Original rows and columns:", df.shape)


# ------------------------------------------------------------
# 3. Clean blank text and labels
# ------------------------------------------------------------

df[TEXT_COL] = (
    df[TEXT_COL]
    .fillna("")
    .astype(str)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

df[LABEL_COL] = (
    df[LABEL_COL]
    .fillna("")
    .astype(str)
    .str.strip()
)

valid_rows = (df[TEXT_COL] != "") & (df[LABEL_COL] != "")
blank_rows_removed = int((~valid_rows).sum())

df = df.loc[valid_rows].copy()

print("Rows removed because text or label was blank:", blank_rows_removed)
print("Rows remaining after blank removal:", len(df))


# ------------------------------------------------------------
# 4. Normalize text to identify repeated notes
# ------------------------------------------------------------

# This temporary column treats differences in capitalization
# and repeated spaces as the same text.
df["_normalized_text"] = (
    df[TEXT_COL]
    .str.lower()
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)


# ------------------------------------------------------------
# 5. Resolve repeated notes with conflicting labels
# ------------------------------------------------------------

# Count how often each normalized note appears with each label.
text_label_counts = (
    df.groupby(["_normalized_text", LABEL_COL])
    .size()
    .reset_index(name="label_count")
)

# Find the largest label count for each normalized note.
largest_count = (
    text_label_counts
    .groupby("_normalized_text")["label_count"]
    .transform("max")
)

# Keep the most common label for each note.
majority_labels = text_label_counts[
    text_label_counts["label_count"] == largest_count
].copy()

# If two or more labels are tied for most common, the note is ambiguous.
number_of_majority_labels = (
    majority_labels
    .groupby("_normalized_text")
    .size()
)

tied_texts = set(
    number_of_majority_labels[
        number_of_majority_labels > 1
    ].index
)

# Count notes that appeared with more than one label.
labels_per_text = df.groupby("_normalized_text")[LABEL_COL].nunique()
conflicting_texts = labels_per_text[labels_per_text > 1]

print(
    "Distinct notes appearing with multiple labels:",
    len(conflicting_texts)
)
print("Tied notes excluded:", len(tied_texts))

# Remove tied notes from the majority-label table.
# For each remaining note, the most frequent label is now unambiguous.
majority_labels = majority_labels[
    ~majority_labels["_normalized_text"].isin(tied_texts)
]

majority_label_by_text = majority_labels.set_index(
    "_normalized_text"
)[LABEL_COL]

# Keep one copy of each normalized note.
unique_notes = df.drop_duplicates(
    subset="_normalized_text",
    keep="first"
).copy()

# Apply the majority label to each unique note.
unique_notes[LABEL_COL] = unique_notes["_normalized_text"].map(
    majority_label_by_text
)

# Remove notes that had tied labels.
unique_notes = unique_notes.dropna(subset=[LABEL_COL]).copy()

# Remove the temporary helper column.
unique_notes = unique_notes.drop(columns="_normalized_text")

print("Unique notes after duplicate handling:", len(unique_notes))


# ------------------------------------------------------------
# 6. Select up to eight specialties with enough examples
# ------------------------------------------------------------

all_specialty_counts = unique_notes[LABEL_COL].value_counts()

# Keep labels with at least 10 examples, then select the eight largest.
eligible_counts = all_specialty_counts[all_specialty_counts >= 10]
selected_labels = eligible_counts.head(8).index

if len(selected_labels) < 2:
    raise ValueError(
        "Fewer than two specialties have at least 10 examples after "
        "cleaning. Inspect the data and duplicate-label counts."
    )

df_model = unique_notes[
    unique_notes[LABEL_COL].isin(selected_labels)
].copy()

selected_counts = df_model[LABEL_COL].value_counts()

print("\nSpecialties selected for the model:")
print(selected_counts)


# ------------------------------------------------------------
# 7. Create stratified 70/15/15 splits
# ------------------------------------------------------------

# First, allocate 70% to training and 30% to a temporary set.
train_df, temporary_df = train_test_split(
    df_model,
    test_size=0.30,
    random_state=42,
    stratify=df_model[LABEL_COL]
)

# Divide the remaining 30% equally:
# 15% for validation and 15% for testing overall.
validation_df, test_df = train_test_split(
    temporary_df,
    test_size=0.50,
    random_state=42,
    stratify=temporary_df[LABEL_COL]
)


# ------------------------------------------------------------
# 8. Print split sizes and compare specialty proportions
# ------------------------------------------------------------

total_rows = len(df_model)

print("\nSplit sizes:")
print(f"Training:   {len(train_df)} ({len(train_df) / total_rows:.1%})")
print(
    f"Validation: {len(validation_df)} "
    f"({len(validation_df) / total_rows:.1%})"
)
print(f"Test:       {len(test_df)} ({len(test_df) / total_rows:.1%})")

distribution = pd.DataFrame({
    "train_percent": (
        train_df[LABEL_COL]
        .value_counts(normalize=True)
        .reindex(selected_labels, fill_value=0) * 100
    ),
    "validation_percent": (
        validation_df[LABEL_COL]
        .value_counts(normalize=True)
        .reindex(selected_labels, fill_value=0) * 100
    ),
    "test_percent": (
        test_df[LABEL_COL]
        .value_counts(normalize=True)
        .reindex(selected_labels, fill_value=0) * 100
    ),
}).round(1)

print("\nSpecialty percentages in each split:")
print(distribution)


# ------------------------------------------------------------
# 9. Save the prepared data and split files
# ------------------------------------------------------------

prepared_file = processed_folder / "mtsamples_day2_clean.csv"
train_file = processed_folder / "train.csv"
validation_file = processed_folder / "validation.csv"
test_file = processed_folder / "test.csv"
counts_file = reports_folder / "day2_specialty_counts.csv"

df_model.to_csv(prepared_file, index=False)
train_df.to_csv(train_file, index=False)
validation_df.to_csv(validation_file, index=False)
test_df.to_csv(test_file, index=False)
selected_counts.to_csv(counts_file, header=["count"])

print("\nSaved files:")
print(prepared_file)
print(train_file)
print(validation_file)
print(test_file)
print(counts_file)