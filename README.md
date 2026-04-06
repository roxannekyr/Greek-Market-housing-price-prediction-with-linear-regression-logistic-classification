# 🏠 Greek Real Estate — Price Prediction & Market Segmentation

> End-to-end machine learning analysis of **183,000+ notarial deed 
> transactions** in Greece, combining econometric regression diagnostics, 
> feature engineering, and binary classification to model property value 
> and market positioning.

![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-OLS%20%7C%20VIF%20%7C%20DW-4B8BBE)
![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?logo=powerbi&logoColor=black)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f)

---

## 📌 What This Project Does

The Greek real estate market is characterised by high heterogeneity — 
historic Athenian apartments, rural agricultural storage, post-war 
residential blocks, and premium new builds transact side by side. 
Standard linear models underperform on this data without deliberate 
feature engineering.

This project addresses that with two complementary analytical lenses:

| Notebook | Problem Type | Model | Primary Metric |
|---|---|---|---|
| `GR_Housing_Price_Prediction` | Regression | OLS Linear + Ridge/Lasso | R² = **0.508** on log-price |
| `GR_Housing_Price_Classification` | Classification | Logistic Regression (L2) | ROC-AUC = **0.854** |

> **Why two models?**  
> Regression answers *"what is this property worth?"*  
> Classification answers *"is this property above or below the market 
> median?"* — a more robust question when price variance is extreme.

---

## 📊 Key Results

### Part A — Linear Regression (Price Prediction)

| Metric | Value |
|---|---|
| R² Score (log scale) | 0.508 |
| MAE (real prices) | €37,393 |
| RMSE (real prices) | €81,466 |
| Observations | 183,348 |
| Durbin-Watson | 1.998 ✅ |

**Model verdict**: Reliable for standard residential properties. 
Heteroscedasticity is present — the model under-predicts at the low end 
and over-predicts at the high end, consistent with the structural 
diversity of the Greek market. Luxury and distressed assets fall outside 
the model's reliable range; a segmented or non-linear model would be 
the natural next step.

### Part B — Logistic Regression (Market Positioning)

| Metric | Value |
|---|---|
| ROC-AUC | **0.854** |
| Recall — Expensive class | **83.3%** |
| 5-Fold CV AUC Std | ±0.002 |
| F1 Score | — |

**Model verdict**: Stable and commercially useful. The model 
correctly identifies 4 in 5 expensive properties — the commercially 
critical outcome for valuation, acquisition screening, or tax 
assessment workflows.

---

## 🔬 Feature Engineering Decisions

These choices drove model performance more than algorithm selection:

**Log-transformation of target and continuous features**  
Property prices are right-skewed with extreme outliers. Log-transforming 
the target converts a multiplicative price structure into an additive one 
that OLS can handle correctly.

**Polynomial size term** (`log_area²`)  
Raw area was a poor predictor. The squared log-area term revealed that 
price scales non-linearly with size — extra square metres on already-large 
properties command disproportionately higher prices per m².

**Location × Size interaction** (`log_area × zone_price`)  
Engineered to capture a fundamental real estate principle: location 
multiplies the value of size. A 120m² flat in a high-zone district 
is not simply additive — it is the strongest single predictor in both 
models.

**Era-based age buckets** instead of raw construction year  
Greek construction history has meaningful regulatory breakpoints:
- `is_NewBuild`: post-2010 (modern seismic & energy standards)
- `is_PostWar`: 1950–1990 (pre-regulation, high renovation liability)
- `is_Historic`: pre-1940 (listed building restrictions)

**Legal encumbrance flags**  
Unbuildable plots, expropriation liens, and unfinished construction 
status each carry independent negative price signals that raw features 
would miss.

---

## 🔍 Diagnostic Rigour

Beyond standard ML metrics, the analysis includes:

- **VIF (Variance Inflation Factor)** — multicollinearity check on all 
  continuous features; justified retention of correlated polynomial terms
- **Durbin-Watson test** — serial correlation in residuals (1.998, within 
  acceptable bounds)
- **Residual plots** — Homoscedasticity, Q-Q Normal plot, Scale-Location, 
  and Cook's Distance for influential observations
- **Ridge vs Lasso comparison** — concluded that regularisation adds 
  limited value given the feature engineering already performed; 
  documented reasoning rather than blind application
- **Cross-validation** (5-fold KFold) — AUC variance of ±0.002 confirms 
  the classification model is not overfitting

---

## 📈 Visualisations

### Residual Diagnostics — Linear Regression
![Residual Plots](assets/residual_plots.png)

### ROC Curve & Confusion Matrix — Logistic Regression
![ROC and Confusion Matrix](assets/roc_confusion.png)

### Feature Importance — Logistic Regression Coefficients
![Feature Importance](assets/feature_importance_logistic.png)

### Power BI Dashboard
![Dashboard Preview](assets/dashboard_preview.png)

> 📥 The full interactive dashboard is available as 
> [`GR_Housing_Dashboard.pbix`](GR_Housing_Dashboard.pbix)  
> *(Requires Power BI Desktop — free download from Microsoft)*

---

## 💼 Business Insights

The models surface findings with direct commercial relevance:

1. **Location amplifies size** — the interaction term is the dominant 
   driver in both models; location-only or size-only analysis materially 
   understates price
2. **New builds carry a measurable premium** — post-2010 construction 
   adds ~30% to predicted price (coefficient 0.26 on log scale: 
   (e^0.26 − 1) × 100%), driven by seismic standards and energy 
   efficiency ratings
3. **Post-war and historic stock is structurally discounted** — 
   renovation liability and regulatory constraints outweigh heritage 
   appeal in transaction data
4. **Parking and storage are categorically separate assets** — their 
   coefficients (−1.00, −0.93) indicate they should not be valued using 
   residential comparables
5. **Legal encumbrances are computable price signals** — unbuildable 
   classification alone carries a −1.38 coefficient, quantifying the 
   litigation risk premium in real terms

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data wrangling | `pandas`, `numpy` |
| ML & preprocessing | `scikit-learn` (LinearRegression, LogisticRegression, Ridge, Lasso, StandardScaler, KFold) |
| Statistical diagnostics | `statsmodels` (OLS, VIF, Durbin-Watson) |
| Visualisation | `matplotlib`, `seaborn` |
| BI & reporting | Power BI Desktop |

---

## Data Notice

> **⚠️ The raw dataset is not included in this repository.**
>
> The dataset contains property transaction records subject to data 
> handling obligations and is not redistributable. The full data schema 
> — column names, data types, and value descriptions — is documented 
> in [`data_schema.md`](data_schema.md) for reproducibility.
>
> Both notebooks are fully annotated with inline commentary. Any dataset 
> conforming to the documented schema can be used to reproduce the 
> analysis. The `Data/` directory is tracked as a placeholder only.

---

## 📂 Repository Contents

| File | Description |
|---|---|
| `GR_Housing_Price_Prediction.ipynb` | Part A: Linear regression Model|
| `GR_Housing_Price_Classification.ipynb` | Part B: Logistic regression Model |
