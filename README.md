# Greek Real Estate Price Intelligence
### Predictive Modelling on 300K+ Property Transactions with Linear & Logistic Regression 

## Project Overview

This project applies supervised machine learning to a large-scale dataset of Greek property transactions to answer two commercially relevant questions:

1. **What is a property worth?** — Continuous price prediction via Linear Regression
2. **Is this property expensive?** — Binary classification via Logistic Regression

The dataset contains **304,206 real estate transactions** across Greek municipalities, making this a production-scale modelling exercise rather than a toy example.

## Business Motivation

Real estate pricing in Greece is notoriously opaque. Zone-based valuations (Τιμή Ζώνης) are set by the state, but actual transaction prices diverge significantly based on size, age, floor, and location. This project builds interpretable models that quantify the true price drivers — directly applicable to:

- **Automated Valuation Models (AVMs)** for banks and mortgage lenders
- **Due diligence tools** for PE/RE funds acquiring Greek assets
- **Tax compliance analytics** for detecting under-declared property transactions

## Methodology

Part 1 — Data Cleaning & EDA (`300K rows → 183K clean observations`)

| Challenge | Approach |
|---|---|
| Missing values in `floor` | Imputed with column median |
| Missing values in `base_price` | Hierarchical imputation: area → city → region median |
| Outliers in `price` (target) | IQR on log-transformed values; extreme fence at 3×IQR |
| Floor encoding | Basement floor `"Υ"` → `-1`; capped at 7th floor |
| Skewed distributions | Log transformation applied to `base_price`, `main_area_sqm`, `price` |
| Greek column names | Mapped to clean English equivalents for reproducibility |

Part 2 — Feature Engineering

Beyond the raw features, the following were derived to improve model fit:

- `property_age` = 2025 − `year_built`
- `log_main_area_sqm` + `log_main_area_sqm²` — captures the **non-linear relationship** between size and price
- `Interaction_Area_Location` = `log_main_area_sqm × log_base_price` — a location-adjusted size premium
- `is_NewBuild` = 1 if built after 2010
- One-Hot Encoding on `property_type`
- StandardScaler applied to all numeric features prior to modelling

## Model Results

### Linear Regression — Price Prediction

| Metric | Value |
|---|---|
| R² Score | **0.513** |
| Durbin-Watson | 1.998 ✅ (no autocorrelation) |
| VIF | < 5 for most features (multicollinearity acceptable) |
| Observations | 183,348 |
| Validation | 5-Fold Cross-Validation |

> The model explains ~51% of variance in log-transformed transaction prices on a statistically robust sample. The log-log specification yields directly interpretable **elasticity coefficients** (% change in price per % change in a feature).

**Top price drivers (by coefficient magnitude):**
- `log_main_area_sqm²` — size effect is non-linear; larger properties command disproportionately higher prices per sqm
- `Interaction_Area_Location` — a large property in a high-zone area is the single strongest predictor
- `log_base_price` — state zone valuation correlates meaningfully with actual transaction prices, but is not a 1:1 proxy

### Logistic Regression — "Is This Property Expensive?" Classification

Binary target: `Expensive = 1` if `price > median`, else `0`

| Metric | Value |
|---|---|
| ROC-AUC | **0.854** |
| Recall (Expensive class) | **83.3%** — catches 4 in 5 expensive properties |
| CV Stability | Std ±0.002 across 5 folds |
| Validation | 5-Fold Cross-Validation |

> AUC of 0.854 places the model well into the "good discriminator" band. High recall on the expensive class is the commercially preferred outcome — missing an expensive property is a costlier error than a false alarm.

**Key business findings:**
1. **Size is non-linear** : squared log-area is the most important feature; extra space on large properties commands an exponential premium
2. **Location × size interaction** is the second-strongest signal : a big property in a premium zone is the clearest indicator of an expensive transaction
3. **New builds** (post-2010) carry a strong positive coefficient (+0.48), reflecting modern energy standards and buyer preference
4. **Post-war (1950–1990) and pre-war (<1940) properties** are significantly cheaper, consistent with poor insulation, seismic risk, and high renovation costs

## Tech Stack

| Category | Libraries |
|---|---|
| Data manipulation | `pandas`, `numpy` |
| Visualisation | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` (LinearRegression, LogisticRegression, KFold, StandardScaler) |
| Statistical diagnostics | `statsmodels` (OLS, Durbin-Watson, VIF) |
| Distribution testing | `scipy.stats` |

## Key Analytical Decisions & Trade-offs

- **Why log-transform the target?** The raw `price` distribution is heavily right-skewed (as expected in real estate). Log transformation stabilises variance, improves OLS assumptions, and produces elasticity-interpretable coefficients.
- **Why keep high-VIF polynomial terms?** `log_main_area_sqm` and its square share variance by design. Removing either would worsen predictive power and break the non-linear size relationship. Context-aware VIF interpretation matters.
- **Why not use IQR alone for outliers?** Applied to 300K rows, standard IQR flagged 13,000+ rows as outliers — a mass removal that would introduce selection bias. Log transformation followed by a wider fence was the better-calibrated approach.
- **Ridge/Lasso relevance?** With VIF largely under control and ~183K observations, regularisation adds limited value here. It becomes more relevant if the feature set is expanded (e.g., adding neighbourhood-level aggregates).
- **Class imbalance in classification:** A 55/45 split between expensive/non-expensive in the test set is mild. SMOTE would be the next step if recall on the non-expensive class needs improvement.

## About

Built as part of an applied ML assignment on real-world Greek cadastral data.
