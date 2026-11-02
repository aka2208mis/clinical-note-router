# Clinical Note Routing Assistant

A learning project that uses NLP to suggest a medical specialty for a sample transcription. Low-scoring suggestions are flagged for human review.

> This is an educational prototype. It is not for diagnosis or treatment decisions.

## Demo

![Streamlit app screenshot](assets/streamlit_demo.png)

## Project goal

Given a sample transcription, predict its specialty using the note text. Show the model's top suggestions, supporting terms, and a human-review warning for low-scoring predictions.

## Dataset

The project uses the public Medical Transcriptions dataset from Kaggle:
https://www.kaggle.com/datasets/tboyle10/medicaltranscriptions

The project uses the `transcription` column as input and the `medical_specialty` column as the target. The raw dataset is not included in this repository.

## Method

1. Clean blank notes and labels.
2. Handle repeated notes and ambiguous labels.
3. Select the specialties used in this project.
4. Create stratified training, validation, and test splits.
5. Convert text to TF-IDF features.
6. Train a Logistic Regression classifier.
7. Review macro-F1 and per-specialty results.
8. Send low-scoring predictions for human review.

## Results

- Number of specialties: [enter your actual number]
- Validation macro-F1: [enter the value from your output]
- Validation accuracy: [enter the value from your output]
- Review threshold: [enter the threshold you selected]

Do not invent or estimate results. Use the numbers printed by your scripts.

## Project structure

```text
clinical-note-router/
├── app.py
├── train_model.py
├── day2_split.py
├── day4_evaluate.py
├── day5_threshold.py
├── data/
├── artifacts/
├── reports/
├── requirements.txt
└── README.md
```

## Setup and run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

4. Download the dataset and place `mtsamples.csv` in `data/raw/`.
5. Create the splits:

   ```bash
   python day2_split.py
   ```

6. Train the model:

   ```bash
   python train_model.py
   ```

7. Choose a threshold using validation data:

   ```bash
   python day5_threshold.py
   ```

8. Start the app:

   ```bash
   python -m streamlit run app.py
   ```

## Limitations

- The model covers only the specialties selected for this project.
- The top model score is an estimate, not a guarantee that a prediction is correct.
- Specialty predictions require human review for low-scoring examples.
- This prototype is not a clinical decision-support system.