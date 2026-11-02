from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


# ------------------------------------------------------------
# 1. Find the project folders and split files
# ------------------------------------------------------------

# This finds the folder containing train_model.py
project_folder = Path(__file__).resolve().parent

processed_folder = project_folder / "data" / "processed"
artifacts_folder = project_folder / "artifacts"
reports_folder = project_folder / "reports"

# Create output folders if they do not exist
artifacts_folder.mkdir(parents=True, exist_ok=True)
reports_folder.mkdir(parents=True, exist_ok=True)

train_file = processed_folder / "train.csv"
validation_file = processed_folder / "validation.csv"

# Check that Day 2 created the required files
for file_path in [train_file, validation_file]:
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required split file was not found:\n{file_path}\n"
            "Run day2_split.py first."
        )


# ------------------------------------------------------------
# 2. Load the training and validation data
# ------------------------------------------------------------

train_df = pd.read_csv(train_file)
validation_df = pd.read_csv(validation_file)

TEXT_COL = "transcription"
LABEL_COL = "medical_specialty"

# Check that the required columns are in both files
required_columns = {TEXT_COL, LABEL_COL}

for file_name, data in [
    ("train.csv", train_df),
    ("validation.csv", validation_df),
]:
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"{file_name} is missing these columns: {missing_columns}\n"
            f"Available columns: {data.columns.tolist()}"
        )

# Make sure the text and label columns contain usable values
for file_name, data in [
    ("train.csv", train_df),
    ("validation.csv", validation_df),
]:
    if data[TEXT_COL].isna().any() or data[LABEL_COL].isna().any():
        raise ValueError(
            f"{file_name} contains missing text or specialty labels. "
            "Check the Day 2 cleaning step."
        )

X_train = train_df[TEXT_COL].astype(str)
y_train = train_df[LABEL_COL].astype(str)

X_validation = validation_df[TEXT_COL].astype(str)
y_validation = validation_df[LABEL_COL].astype(str)

print("Training rows:", len(train_df))
print("Validation rows:", len(validation_df))
print("Specialties:", sorted(y_train.unique()))


# ------------------------------------------------------------
# 3. Create the TF-IDF and Logistic Regression pipeline
# ------------------------------------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_features=30000,
            sublinear_tf=True,
            stop_words=None
        )
    ),
    (
        "classifier",
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        )
    )
])


# ------------------------------------------------------------
# 4. Train the model using training data only
# ------------------------------------------------------------

print("\nTraining the model...")
model.fit(X_train, y_train)
print("Training complete.")

# Show how many word/phrase features TF-IDF learned
tfidf_model = model.named_steps["tfidf"]
print("TF-IDF vocabulary size:", len(tfidf_model.vocabulary_))


# ------------------------------------------------------------
# 5. Check performance on validation data
# ------------------------------------------------------------

print("\nPredicting specialties for validation data...")
validation_predictions = model.predict(X_validation)

accuracy = accuracy_score(y_validation, validation_predictions)
macro_f1 = f1_score(
    y_validation,
    validation_predictions,
    average="macro",
    zero_division=0
)

print("\nValidation accuracy:", round(accuracy, 4))
print("Validation macro-F1:", round(macro_f1, 4))

print("\nValidation classification report:")
print(
    classification_report(
        y_validation,
        validation_predictions,
        digits=3,
        zero_division=0
    )
)


# ------------------------------------------------------------
# 6. Save validation predictions for review
# ------------------------------------------------------------

# These scores are model probability estimates, not clinical certainty.
validation_probabilities = model.predict_proba(X_validation)
best_class_indices = validation_probabilities.argmax(axis=1)
top_scores = validation_probabilities.max(axis=1)

validation_results = pd.DataFrame({
    "actual_specialty": y_validation.to_numpy(),
    "predicted_specialty": validation_predictions,
    "model_score": top_scores
})

validation_results.to_csv(
    reports_folder / "baseline_validation_predictions.csv",
    index=False
)


# ------------------------------------------------------------
# 7. Save the trained model
# ------------------------------------------------------------

model_path = artifacts_folder / "specialty_router.joblib"
joblib.dump(model, model_path)

print("\nSaved trained model to:", model_path)
print(
    "Saved validation predictions to:",
    reports_folder / "baseline_validation_predictions.csv"
)