# 🏠 House Price Prediction using Machine Learning

### End-to-End Regression Project | Kaggle Competition Solution

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green.svg)
![Status](https://img.shields.io/badge/Project-Completed-success.svg)

## 📌 Project Overview

This project is a complete **Machine Learning Regression Pipeline** built on the Ames Housing Dataset from Kaggle.

It covers:
- Data Cleaning
- EDA
- Missing Value Handling
- Outlier Detection
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Cross Validation
- Model Saving


## 📁 Project Structure

```
house-price-prediction/
│
├── data/
│   └── train Data House Price.csv     # Dataset (download from Kaggle)
│
├── notebooks/
│   └── house_price_prediction.ipynb   # Main notebook — full pipeline
│
├── plots/                             # All saved EDA and evaluation plots
│   ├── 01_saleprice_distribution.png
│   ├── 02_quality_vs_price.png
│   ├── 03_area_vs_price.png
│   ├── 04_yearbuilt_vs_price.png
│   ├── 05_neighborhood_vs_price.png
│   ├── 07_basement_vs_price.png
│   ├── 08_full_correlation_heatmap.png
│   ├── 09_top_correlated_features.png
│   ├── 10_missing_values.png
│   ├── 11_outlier_removal_comparison.png
│   ├── 12_statistical_significance.png
│   ├── 13_train_vs_test_r2.png
│   ├── 14_cv_r2_comparison.png
│   ├── 15_actual_vs_predicted.png
│   └── 16_feature_importance.png
│
├── models/
│   ├── best_model_Ridge_Regression.pkl   # Best trained model
│   ├── scaler.pkl                        # StandardScaler
│   └── feature_names.pkl                 # Feature names list
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 🔄 Pipeline Steps

```
Step 1  →  Data Loading & Basic Exploration
Step 2  →  Exploratory Data Analysis (EDA) — 7 plots
Step 3  →  Missing Value Treatment
Step 4  →  Outlier Detection & Removal (GrLivArea + IQR method)
Step 5  →  Statistical Tests — Pearson, Spearman, ANOVA
Step 6  →  Feature Engineering — 7 new features created
Step 7  →  Label Encoding (Categorical → Numeric)
Step 8  →  Train-Test Split (80/20) + StandardScaler
Step 9  →  Baseline Model Training (4 Models)
Step 10 →  Hyperparameter Tuning (RandomizedSearchCV)
Step 11 →  Cross Validation (5-Fold)
Step 12 →  Best Model Selection
Step 13 →  All Plots Saved
Step 14 →  Model Saved as .pkl
```

---
---

## 📊 Models Used

| Model | Type | Data Used |
|---|---|---|
| Ridge Regression | Linear + L2 Regularization | Scaled |
| Random Forest | Ensemble — Bagging | Unscaled |
| Gradient Boosting | Ensemble — Boosting | Unscaled |
| XGBoost | Optimized Gradient Boosting | Unscaled |

---

## 📈 Baseline Model Results

| Model | Train R² | Test R² | RMSE | Overfit |
|---|---|---|---|---|
| Linear Regression | 0.9047 | 0.8875 | 0.1111 | ✅ No |
| Random Forest | 0.9793 | 0.8663 | 0.1211 | ⚠️ Yes |
| Gradient Boosting | 0.9631 | 0.8913 | 0.1092 | ⚠️ Yes |
| XGBoost | 0.9999 | 0.8652 | 0.1216 | ⚠️ Yes |

---

## 📈 After Hyperparameter Tuning

| Model | Train R² | Test R² | RMSE | Overfit |
|---|---|---|---|---|
| **Ridge Regression** | **0.9031** | **0.8911** | **0.1093** | **✅ No** |
| Random Forest | 0.9806 | 0.8698 | 0.1195 | ⚠️ Yes |
| Gradient Boosting | 0.9646 | 0.8947 | 0.1075 | ⚠️ Yes |
| XGBoost | 0.9861 | 0.9085 | 0.1002 | ⚠️ Yes |

## 🎯 Final Selected Model

# 🥇 Ridge Regression

Chosen because it delivered:
- Stable cross-validation performance
- Better generalization
- Simpler deployment
- Fast inference
- > 🏆 **Best Model: Ridge Regression**
> Selected for clean generalization — zero overfitting (Train-Test diff = 0.012), stable CV performance, and production reliability.


## 📂 Dataset Information

| Item | Value |
|------|------|
| Source | Kaggle |
| Rows (Original) | 1460 |
| Rows (After Cleaning) | 1241 |
| Features | 79 |
| Target | SalePrice |

- **Source:** [Kaggle — House Prices: Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques)


## 🧪 Key EDA Findings

- **SalePrice** is right-skewed → Log transformation applied (`np.log1p`)
- **OverallQual** is the strongest predictor (Pearson r = 0.79)
- **GrLivArea** has 2 famous outliers (large area, very low price) → removed
- **Neighborhood** significantly affects price (ANOVA confirmed)
- **8 weak features dropped** after statistical testing:
  `Street`, `LandSlope`, `MiscFeature`, `PoolQC`, `BsmtFinSF2`, `YrSold`, `BsmtHalfBath`, `MSSubClass`

----

## 🛠️ Feature Engineering

7 new features created from existing columns:

| Feature | Description |
|---|---|
| `TotalArea` | Basement + 1st Floor + 2nd Floor area combined |
| `TotalBath` | All bathrooms combined (half bath = 0.5) |
| `HouseAge` | 2010 minus year built |
| `RemodelAge` | 2010 minus last remodel year |
| `HasGarage` | Binary — garage exists (1) or not (0) |
| `HasBasement` | Binary — basement exists (1) or not (0) |
| `HasFireplace` | Binary — fireplace exists (1) or not (0) |

---

## 🤖 Models Compared

- Linear Regression
- Ridge Regression
- Random Forest
- Gradient Boosting
- XGBoost

## 📈 Final Tuned Performance

| Model | Train R² | Test R² | RMSE |
|------|---------|--------|------|
| Ridge Regression | 0.9031 | 0.8911 | 0.1093 |
| Random Forest | 0.9806 | 0.8698 | 0.1195 |
| Gradient Boosting | 0.9646 | 0.8947 | 0.1075 |
| XGBoost | 0.9861 | 0.9085 | 0.1002 |

## 💾 Saved Files

```text
models/
├── best_model_Ridge_Regression.pkl
├── scaler.pkl
└── feature_names.pkl
```
## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

# 2. Create virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# Mac/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 How to Run

```bash
jupyter notebook notebooks/house_price_prediction.ipynb
```

Run all cells from top to bottom. The notebook is clearly divided into labeled sections.

> ⚠️ **Note:** Place the dataset file inside the `data/` folder before running:
> `data/train Data House Price.csv`

---
## 🔮 Load and Use the Saved Model

```python
import joblib
import numpy as np

# Load model and scaler
model  = joblib.load('models/best_model_Ridge_Regression.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare input (must be preprocessed the same way as training)
# X_input = your_preprocessed_data

# Scale input
# X_scaled = scaler.transform(X_input)

# Predict (output is log-scale)
# log_prediction = model.predict(X_scaled)

# Convert to actual price
# actual_price = np.expm1(log_prediction)
```

---
## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
scipy
joblib
jupyter
```
Install all:
```bash
pip install -r requirements.txt
```

---

## 👤 Author

**Sunil Kumar**
MSc Artificial Intelligence & Machine Learning — IIIT Lucknow

- 📧 Sunilkumar100403@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/sunil-kumar-ab174420a/)
- 🐙 [GitHub](https://github.com/sunilsaini01)

---

## 🌟 Future Improvements

- Streamlit Deployment
- Ensemble Stacking
- SHAP Explainability
- API Deployment
