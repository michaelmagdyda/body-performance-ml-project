# Body Performance Analytics & Intelligent Classification System

**Introduction to AI & Machine Learning — Course Project (Digilians, 2025–2026)**

An end-to-end machine learning pipeline on the Body Performance dataset: exploratory data analysis, rule-based data cleaning, classification of fitness level (classes A–D) with six models across three train/test splits, K-Means clustering, and regression models predicting broad-jump distance.

## Dataset
Physical fitness measurements for **13,393 participants** (13,288 after cleaning): age, gender, height, weight, body fat %, blood pressure (diastolic/systolic), grip force, sit-and-bend-forward flexibility, sit-ups count and broad jump. **BMI** was added as an engineered feature. Target: performance `class` A (best) to D.

## Pipeline

| Step | Notebook | Description |
|---|---|---|
| 1. EDA | `01. EDA/Final/01_EDA _new.ipynb` | Structure, distributions, missing values, duplicates, class balance, correlations |
| 2. Preprocessing | `01. EDA/Final/02_Preprocessing_(1).ipynb` | Sequential physiological validity rules (blood pressure, body composition, etc.) |
| 3. Classification | `03. Modelling_Python Files/Classification Models/04_Modelling_(all).ipynb` | 6 pipelines tuned with `RandomizedSearchCV` on 80:20, 70:30 and 50:50 splits |
| 4. Evaluation | `03. Modelling_Python Files/Evaluation/05_Evaluation_v2.ipynb` | Accuracy, macro precision/recall/F1, confusion matrices, radar charts, feature importance |
| 5. Regression | `03. Modelling_Python Files/Regression/` | Linear Regression, Decision Tree and SVR predicting `broad_jump_cm` |

## Classification Results (test accuracy)

| Model | 80:20 | 70:30 | 50:50 |
|---|---|---|---|
| **Neural Network (MLP)** | **0.759** | **0.746** | **0.737** |
| Random Forest | 0.753 | 0.745 | 0.723 |
| SVM | 0.719 | 0.711 | 0.700 |
| Decision Tree | 0.681 | 0.685 | 0.645 |
| KNN | 0.638 | 0.640 | 0.625 |
| Logistic Regression | 0.620 | 0.617 | 0.619 |

## Key Findings
- The **strength & power features** (grip force, sit-ups, broad jump) carry the strongest signal; body fat % has a strong negative effect on performance.
- A **K-Means (k=4)** experiment on PCA-reduced data produced clean, well-separated clusters, while the original class labels overlap heavily — suggesting the original labels don't fully reflect the data's natural structure (see `K_mean Clustering.png` and `02. Data/Data Cleaned with clustered_Target.csv`).
- For broad-jump regression, the **Decision Tree** (R² ≈ 0.72–0.75) outperformed Linear Regression (R² ≈ 0.64–0.68).

## Repository Structure
```
01. EDA/Final/                      EDA & preprocessing notebooks
02. Data/                           Original, cleaned and cluster-labelled datasets
03. Modelling_Python Files/
  Classification Models/            Training notebook + Output/ (splits, CV results, trained .pkl models)
  Evaluation/                       Evaluation notebook + outputs/ (charts, result summaries)
  Regression/                       Linear Regression, Decision Tree, SVM regression
05. Report and presentation/        Final report (PDF) and presentation
```

## Run It
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
jupyter notebook
```
Run the notebooks in order (EDA → Preprocessing → Modelling → Evaluation). Trained models in `Output/` can be loaded with `joblib.load()`.

## Tools
Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Jupyter
