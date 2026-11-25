# Diabetes Prediction Using Machine Learning

A practical machine learning project that explores the Pima Indians Diabetes dataset, applies a leakage-aware preprocessing workflow, trains and compares classical classification algorithms, and evaluates their performance using multiple metrics.

> **Project status:** Educational ML project. The notebook contains executed outputs for the experiments documented below.

## Project Overview

The goal is to build a binary classification workflow that predicts the dataset's target label:

- `0` — No diabetes
- `1` — Diabetes

The project emphasizes the complete workflow rather than accuracy alone: understanding the data, identifying invalid values, building preprocessing pipelines, comparing models, validating performance, tuning a model, and examining feature importance.

## Dataset

This project uses the **Pima Indians Diabetes Dataset**.

| Item | Value |
|---|---:|
| Records | 768 |
| Input features | 8 |
| Target column | `Outcome` |
| Target class 0 | 500 records |
| Target class 1 | 268 records |
| Total columns | 9 |

### Input features

| Feature | Description |
|---|---|
| `Pregnancies` | Number of pregnancies |
| `Glucose` | Plasma glucose concentration |
| `BloodPressure` | Diastolic blood pressure |
| `SkinThickness` | Triceps skin-fold thickness |
| `Insulin` | 2-hour serum insulin |
| `BMI` | Body mass index |
| `DiabetesPedigreeFunction` | Diabetes pedigree function |
| `Age` | Age in years |
| `Outcome` | Binary target label |

## Workflow

The full implementation is available in [`preprocessing_project/preprocessing.ipynb`](preprocessing_project/preprocessing.ipynb).

1. **Data loading and inspection** — inspect shape, sample rows, data types, descriptive statistics, missing values, duplicates, and target distribution.
2. **Exploratory data analysis** — inspect zero values, visualize possible outliers with boxplots, and explore feature correlations.
3. **Invalid-value handling** — treat zero values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` as missing measurements for preprocessing.
4. **Train/test split** — use an 80/20 split with stratification and a fixed random seed.
5. **Preprocessing pipeline** — apply median imputation and standard scaling within a Scikit-learn Pipeline.
6. **Model comparison** — train Logistic Regression, Random Forest, K-Nearest Neighbors (KNN), and Decision Tree classifiers.
7. **Cross-validation** — use 5-fold StratifiedKFold validation for the Logistic Regression pipeline.
8. **Overfitting check** — compare training and test accuracy for the four baseline models.
9. **Hyperparameter tuning** — use GridSearchCV with 5-fold cross-validation to tune a Random Forest pipeline using F1 score.
10. **Feature importance** — inspect the fitted Random Forest feature importances.
11. **Model persistence** — save the tuned pipeline with Joblib as `diabetes_model.pkl`.

## Technologies

- Python
- Jupyter Notebook
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib

## Model Evaluation

The notebook evaluates models using:

- **Accuracy:** overall proportion of correct predictions.
- **Precision:** proportion of positive predictions that are correct.
- **Recall:** proportion of actual positive cases correctly identified.
- **F1-score:** harmonic mean of precision and recall.

### Baseline model comparison

The following values are copied from the executed model-comparison cell in the notebook. The positive class is `Outcome = 1`.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| K-Nearest Neighbors (KNN) | 0.753 | 0.660 | 0.611 | 0.635 |
| Random Forest | 0.747 | 0.660 | 0.574 | 0.614 |
| Logistic Regression | 0.708 | 0.600 | 0.500 | 0.545 |
| Decision Tree | 0.682 | 0.553 | 0.481 | 0.515 |

### Cross-validation

The 5-fold StratifiedKFold results recorded for the **Logistic Regression pipeline** are:

| Metric | Mean score | Standard deviation |
|---|---:|---:|
| Accuracy | 0.772 | 0.017 |
| Precision | 0.726 | 0.062 |
| Recall | 0.571 | 0.047 |
| F1-score | 0.636 | 0.021 |

### Tuned Random Forest

GridSearchCV selected the following parameters for the Random Forest pipeline:

- `n_estimators = 100`
- `max_depth = 15`
- `min_samples_split = 2`

| Evaluation | Score |
|---|---:|
| Best cross-validation F1-score | 0.663 |
| Test accuracy | 0.766 |
| Test precision | 0.696 |
| Test recall | 0.593 |
| Test F1-score | 0.640 |

These are the notebook's recorded results for its current split and settings; they are not a guarantee of performance on new data. The baseline comparison, Logistic Regression cross-validation, and tuned Random Forest results are separate experiments and should not be treated as one identical evaluation.

## Feature Importance

The tuned Random Forest reports the following feature importances:

| Feature | Importance |
|---|---:|
| Glucose | 0.2764 |
| BMI | 0.1643 |
| DiabetesPedigreeFunction | 0.1252 |
| Age | 0.1126 |
| Insulin | 0.0889 |
| BloodPressure | 0.0810 |
| Pregnancies | 0.0795 |
| SkinThickness | 0.0721 |

Feature importance indicates how the fitted tree-based model used features; it does not establish causation or medical significance.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/MohamedElhadry/Diabetes-Prediction-ML-Copy.git
cd Diabetes-Prediction-ML-Copy
```

### 2. (Recommended) Create and activate a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Open the notebook

```bash
cd preprocessing_project
jupyter notebook preprocessing.ipynb
```

Run the notebook cells from top to bottom. The command above starts Jupyter from the notebook's directory, so the relative path `diabetes.csv` resolves correctly.

## Repository Structure

```text
Diabetes-Prediction-ML-Copy/
├── README.md
├── requirements.txt
├── .gitignore
└── preprocessing_project/
    ├── preprocessing.ipynb
    ├── diabetes.csv
    └── diabetes_model.pkl
```

## Notes and Limitations

- The dataset contains 768 records, so results may vary with different train/test splits and model settings.
- Some measurements use zero as a placeholder even though zero is not a plausible measured value for that field. The notebook treats these values as missing for the selected features.
- The notebook's initial missing-value check reports no `NaN` values; the zero-placeholder inspection is a separate data-quality check.
- The baseline comparison, cross-validation experiment, and tuned-model evaluation use the exact procedures shown in the notebook. Their scores should be interpreted in that context.
- Recall is particularly important to inspect in a screening-related classification task because false negatives can be consequential.
- The saved `.pkl` file is a Python object serialized with Joblib. Only load pickle-based model files from sources you trust, and use compatible Python and Scikit-learn versions.

## Disclaimer

This project is for learning and experimentation only. It is **not** a medical diagnostic tool and must not be used to make healthcare decisions. Any real medical assessment should be made by qualified healthcare professionals.
