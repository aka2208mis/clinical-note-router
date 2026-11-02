from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)


# ------------------------------------------------------------
# 1. Find the project files
# ------------------------------------------------------------

project_folder = Path(__file__).resolve().parent

processed_folder = project_folder / "data" / "processed"
artifacts_folder = project_folder / "artifacts"
reports_folder = project_folder / "reports"

reports_folder.mkdir(parents=True, exist_ok=True)

model_file = artifacts_folder / "specialty_router.joblib"
validation_file = processed_folder / "validation.csv"

for file_path in [model_file, validation_file]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{file_path}"
        )


# ------------------------------------------------------------
# 2. Load the saved model and validation data
# ------------------------------------------------------------

model = joblib.load(model_file)
validation_df = pd.read_csv(validation_file)

TEXT_COL = "transcription"
LABEL_COL = "medical_specialty"

required_columns = {TEXT_COL, LABEL_COL}
missing_columns = required_columns - set(validation_df.columns)

if missing_columns:
    raise ValueError(
        f"validation.csv is missing these columns: {missing_columns}"
    )

X_validation = (
    validation_df[TEXT_COL]
    .fillna("")
    .astype(str)
)

y_true = (
    validation_df[LABEL_COL]
    .fillna("")
    .astype(str)
)

print("Validation rows:", len(validation_df))


# ------------------------------------------------------------
# 3. Predict specialties
# ------------------------------------------------------------

y_pred = model.predict(X_validation)

# Get the highest probability estimate for each note.
# Treat this as a model score, not medical certainty.
probabilities = model.predict_proba(X_validation)
top_scores = probabilities.max(axis=1)


# ------------------------------------------------------------
# 4. Calculate overall and per-specialty metrics
# ------------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

print("\nValidation accuracy:", round(accuracy, 4))
print("Validation macro-F1:", round(macro_f1, 4))

print("\nFull classification report:")
print(
    classification_report(
        y_true,
        y_pred,
        digits=3,
        zero_division=0
    )
)

labels = sorted(y_true.unique())

report = classification_report(
    y_true,
    y_pred,
    labels=labels,
    target_names=labels,
    output_dict=True,
    zero_division=0
)

per_class_metrics = pd.DataFrame({
    "precision": [report[label]["precision"] for label in labels],
    "recall": [report[label]["recall"] for label in labels],
    "f1_score": [report[label]["f1-score"] for label in labels],
    "support": [report[label]["support"] for label in labels],
})

per_class_metrics.index.name = "specialty"

per_class_file = reports_folder / "day4_per_class_metrics.csv"
per_class_metrics.to_csv(per_class_file)

print("\nPer-specialty metrics:")
print(per_class_metrics.round(3))


# ------------------------------------------------------------
# 5. Build and save the confusion matrix
# ------------------------------------------------------------

# Rows represent actual specialties.
# Columns represent specialties predicted by the model.
cm = confusion_matrix(y_true, y_pred, labels=labels)

cm_table = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

cm_table.index.name = "actual_specialty"
cm_table.columns.name = "predicted_specialty"

cm_table.to_csv(reports_folder / "day4_confusion_matrix_counts.csv")

# Draw a normalized confusion matrix.
# Each row is scaled to show the proportion of examples in that
# actual specialty assigned to each predicted specialty.
fig, ax = plt.subplots(figsize=(13, 10))

ConfusionMatrixDisplay.from_predictions(
    y_true,
    y_pred,
    labels=labels,
    display_labels=labels,
    normalize="true",
    values_format=".2f",
    cmap="Blues",
    xticks_rotation=90,
    ax=ax
)

ax.set_title("Validation Confusion Matrix (Row-Normalized)")
fig.tight_layout()

matrix_image = reports_folder / "day4_confusion_matrix.png"
fig.savefig(matrix_image, dpi=150)
plt.close(fig)


# ------------------------------------------------------------
# 6. Find the most frequent specialty mix-ups
# ------------------------------------------------------------

confusion_pairs = []

for actual_index, actual_label in enumerate(labels):
    for predicted_index, predicted_label in enumerate(labels):
        # Ignore the diagonal: those are correct predictions.
        if actual_index != predicted_index:
            error_count = cm[actual_index, predicted_index]

            if error_count > 0:
                confusion_pairs.append({
                    "actual_specialty": actual_label,
                    "predicted_specialty": predicted_label,
                    "error_count": int(error_count),
                })

pairs_df = pd.DataFrame(confusion_pairs)

if not pairs_df.empty:
    pairs_df = pairs_df.sort_values(
        "error_count",
        ascending=False
    ).reset_index(drop=True)

    pairs_df.to_csv(
        reports_folder / "day4_common_error_pairs.csv",
        index=False
    )

    print("\nMost frequent specialty mix-ups:")
    print(pairs_df.head(10).to_string(index=False))
else:
    print("\nNo misclassified specialty pairs were found.")


# ------------------------------------------------------------
# 7. Save examples of incorrect predictions for review
# ------------------------------------------------------------

error_examples = pd.DataFrame({
    "actual_specialty": y_true.to_numpy(),
    "predicted_specialty": y_pred,
    "model_score": top_scores,
    "transcription": X_validation.to_numpy(),
})

error_examples = error_examples[
    error_examples["actual_specialty"]
    != error_examples["predicted_specialty"]
].copy()

# Keep only the first 400 characters in the review file
error_examples["text_preview"] = (
    error_examples["transcription"]
    .str.replace(r"\s+", " ", regex=True)
    .str.slice(0, 400)
)

error_examples = error_examples[
    [
        "actual_specialty",
        "predicted_specialty",
        "model_score",
        "text_preview",
    ]
]

error_examples_file = reports_folder / "day4_misclassified_examples.csv"
error_examples.to_csv(error_examples_file, index=False)

print("\nNumber of incorrect validation predictions:", len(error_examples))

if not pairs_df.empty:
    print("\nA few examples from the most common mix-ups:")

    for _, pair in pairs_df.head(5).iterrows():
        actual_label = pair["actual_specialty"]
        predicted_label = pair["predicted_specialty"]

        matching_examples = error_examples[
            (error_examples["actual_specialty"] == actual_label)
            & (error_examples["predicted_specialty"] == predicted_label)
        ]

        print(
            f"\nActual: {actual_label} | "
            f"Predicted: {predicted_label} | "
            f"Count: {int(pair['error_count'])}"
        )

        for _, example in matching_examples.head(2).iterrows():
            print("Text preview:", example["text_preview"])


# ------------------------------------------------------------
# 8. List saved outputs
# ------------------------------------------------------------

print("\nSaved Day 4 outputs:")
print(per_class_file)
print(reports_folder / "day4_confusion_matrix_counts.csv")
print(matrix_image)
print(reports_folder / "day4_common_error_pairs.csv")
print(error_examples_file)