from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# Replace 0.50 with your chosen Day 5 validation threshold
REVIEW_THRESHOLD = 0.50

project_folder = Path(__file__).resolve().parent
model_file = project_folder / "artifacts" / "specialty_router.joblib"


@st.cache_resource
def load_model():
    if not model_file.exists():
        raise FileNotFoundError(
            f"Model not found: {model_file}. "
            "Run train_model.py first."
        )

    return joblib.load(model_file)


try:
    model = load_model()
except FileNotFoundError as error:
    st.error(str(error))
    st.stop()

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["classifier"]


def get_supporting_terms(text, predicted_label, number_of_terms=5):
    """Return terms that positively contribute to the predicted class score."""

    text_vector = vectorizer.transform([text])
    class_index = list(classifier.classes_).index(predicted_label)
    class_coefficients = classifier.coef_[class_index]

    contributions = (
        text_vector.multiply(class_coefficients)
        .toarray()
        .ravel()
    )

    feature_names = vectorizer.get_feature_names_out()
    positive_indices = np.flatnonzero(contributions > 0)

    if len(positive_indices) == 0:
        return []

    order = np.argsort(contributions[positive_indices])[::-1]
    top_indices = positive_indices[order[:number_of_terms]]

    return [
        (feature_names[index], contributions[index])
        for index in top_indices
    ]


st.title("Clinical Note Routing Assistant")

st.write(
    "Enter a sample transcription to see the model's suggested specialty. "
    "Low-scoring predictions are flagged for human review."
)

st.caption(
    "Educational prototype using sample data. "
    "Not for diagnosis or treatment decisions."
)

text = st.text_area(
    "Medical transcription",
    height=250,
    placeholder="Paste sample transcription text here..."
)

if st.button("Predict specialty"):
    if not text.strip():
        st.warning("Please enter a transcription first.")
    else:
        # Match the whitespace cleanup used during data preparation.
        cleaned_text = " ".join(text.split())

        probabilities = model.predict_proba([cleaned_text])
        probabilities = probabilities[0]

        ranked_indices = np.argsort(probabilities)[::-1]
        best_index = ranked_indices[0]

        predicted_label = classifier.classes_[best_index]
        top_score = probabilities[best_index]

        st.subheader("Suggested specialty")

        if top_score < REVIEW_THRESHOLD:
            st.warning(
                f"Suggested: **{predicted_label}**. "
                "The model score is below the review threshold. "
                "Please send this note for human review."
            )
        else:
            st.success(f"Suggested specialty: **{predicted_label}**")

        st.write(f"Top model score: **{top_score:.1%}**")
        st.write(f"Review threshold: **{REVIEW_THRESHOLD:.1%}**")

        st.subheader("Top three suggestions")

        top_three = pd.DataFrame({
            "Specialty": classifier.classes_[ranked_indices[:3]],
            "Model score": probabilities[ranked_indices[:3]],
        })

        top_three["Model score"] = top_three["Model score"].map(
            lambda score: f"{score:.1%}"
        )

        st.dataframe(top_three, hide_index=True)

        st.subheader("Terms supporting the suggested class")

        terms = get_supporting_terms(cleaned_text, predicted_label)

        if terms:
            for term, contribution in terms:
                st.write(f"- **{term}** — score contribution: {contribution:.3f}")
        else:
            st.write("No strong supporting terms were found.")

        st.caption(
            "These terms show features that influenced the model's score. "
            "They are not a medical explanation or proof that the prediction "
            "is correct."
        )