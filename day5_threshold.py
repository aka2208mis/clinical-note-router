from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score


# Find project folders
project_folder = Path(__file__).resolve().parent
processed_folder = project_folder / "data" / "processed"
artifacts_folder = project_folder / "artifacts"
reports_folder = project_folder / "reports"

reports_folder.mkdir(parents=True, exist_ok=True)

model_file = artifacts_folder / "specialty_router.joblib"
validation_file = processed_folder / "validation.csv"

# Check required files
for file_path in [model_file, validation_file]:
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

# Load the trained Day 3 model and validation data
model = joblib.load(model_file)
validation_df = pd.read_csv(validation_file)

TEXT_COL = "transcription"
LABEL_COL = "medical_specialty"

X_validation = validation_df[TEXT_COL].fillna("").astype(str)
y_true = validation_df[LABEL_COL].fillna("").astype(str).to_numpy()

# Get the model's predicted class and its top score for each note
probabilities = model.predict_proba(X_validation)
classifier = model.named_steps["classifier"]
classes = classifier.classes_

best_indices = probabilities.argmax(axis=1)
predictions = classes[best_indices]
top_scores = probabilities.max(axis=1)

# Try several possible review thresholds
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
results = []

for threshold in thresholds:
    accepted = top_scores >= threshold
    sent_for_review = ~accepted

    if accepted.sum() > 0:
        accuracy_when_accepted = accuracy_score(
            y_true[accepted],
            predictions[accepted]
        )
    else:
        accuracy_when_accepted = None

    results.append({
        "threshold": threshold,
        "accepted_notes": int(accepted.sum()),
        "accepted_accuracy": accuracy_when_accepted,
        "coverage_percent": round(accepted.mean() * 100, 1),
        "sent_for_review": int(sent_for_review.sum()),
    })

threshold_table = pd.DataFrame(results)

print("Threshold results on validation data:")
print(threshold_table.to_string(index=False))

output_file = reports_folder / "day5_threshold_analysis.csv"
threshold_table.to_csv(output_file, index=False)

print("\nSaved threshold analysis to:", output_file)