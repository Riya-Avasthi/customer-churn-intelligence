# 📡 ChurnIQ — Customer Churn Intelligence System

> A real-time, interactive customer churn prediction web application built with **Streamlit** and **Scikit-learn**, powered by the IBM Telco Customer Churn dataset.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Tech Stack](#tech-stack)
4. [Dataset Description](#dataset-description)
5. [Project Architecture](#project-architecture)
6. [Feature Engineering](#feature-engineering)
7. [ML Pipeline Deep Dive](#ml-pipeline-deep-dive)
8. [Application Walkthrough](#application-walkthrough)
9. [Code Structure](#code-structure)
10. [Setup & Installation](#setup--installation)
11. [How to Run](#how-to-run)
12. [Key Design Decisions](#key-design-decisions)
13. [Future Enhancements](#future-enhancements)

---

## Project Overview

**ChurnIQ** is a Customer Churn Intelligence System that predicts whether a telecom customer is likely to leave (churn) based on their demographics, account details, and subscribed services. The application:

- Trains a **Random Forest Classifier** on the IBM Telco dataset at startup
- Handles **class imbalance** using **SMOTE** (Synthetic Minority Oversampling Technique)
- Performs **feature engineering** to create 7 new predictive features
- Provides a **dark-themed, modern UI** built with custom CSS in Streamlit
- Generates **actionable business recommendations** based on each customer's risk profile
- Displays a **visual risk breakdown** with weighted risk factors

---

## Problem Statement

Customer churn (also called customer attrition) is when customers stop doing business with a company. In the telecom industry, acquiring a new customer costs **5–25× more** than retaining an existing one.

**Goal:** Build an intelligent system that:
1. Predicts the probability of a customer churning
2. Identifies key risk factors driving the prediction
3. Provides actionable retention recommendations

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **Python 3.x** | Core programming language |
| **Streamlit** | Web application framework for data apps |
| **Pandas** | Data manipulation and cleaning |
| **NumPy** | Numerical computations |
| **Scikit-learn** | ML pipeline, preprocessing, model training |
| **imbalanced-learn (SMOTE)** | Handling class imbalance in training data |
| **XGBoost** | Available as an alternative classifier (in requirements) |
| **SHAP** | Available for model explainability (in requirements) |
| **Matplotlib / Seaborn** | Available for data visualization (in requirements) |

---

## Dataset Description

**Source:** IBM Telco Customer Churn Dataset  
**File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`  
**Records:** 7,043 customers  
**Features:** 21 columns  
**Target Variable:** `Churn` (Yes / No)  
**Class Distribution:** No = 5,174 (73.5%) | Yes = 1,869 (26.5%) — **imbalanced**

### Column Descriptions

| Column | Type | Description |
|---|---|---|
| `customerID` | String | Unique identifier for each customer |
| `gender` | Categorical | Male / Female |
| `SeniorCitizen` | Binary (0/1) | Whether the customer is a senior citizen |
| `Partner` | Categorical | Whether the customer has a partner (Yes/No) |
| `Dependents` | Categorical | Whether the customer has dependents (Yes/No) |
| `tenure` | Numeric | Number of months the customer has stayed |
| `PhoneService` | Categorical | Whether the customer has phone service (Yes/No) |
| `MultipleLines` | Categorical | Yes / No / No phone service |
| `InternetService` | Categorical | DSL / Fiber optic / No |
| `OnlineSecurity` | Categorical | Yes / No / No internet service |
| `OnlineBackup` | Categorical | Yes / No / No internet service |
| `DeviceProtection` | Categorical | Yes / No / No internet service |
| `TechSupport` | Categorical | Yes / No / No internet service |
| `StreamingTV` | Categorical | Yes / No / No internet service |
| `StreamingMovies` | Categorical | Yes / No / No internet service |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `PaperlessBilling` | Categorical | Yes / No |
| `PaymentMethod` | Categorical | Electronic check / Mailed check / Bank transfer / Credit card |
| `MonthlyCharges` | Numeric | Monthly charge amount in dollars |
| `TotalCharges` | Numeric | Total amount charged over lifetime |
| `Churn` | Target | Whether the customer churned (Yes/No) |

---

## Project Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        ChurnIQ Application                       │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌────────────┐    ┌──────────────┐    ┌────────────────────┐   │
│   │   CSV Data  │───▶│ train_model()│───▶│  Cached ML Model   │   │
│   │  (7043 rows)│    │              │    │  + Preprocessor    │   │
│   └────────────┘    │ • Clean data │    └────────┬───────────┘   │
│                     │ • Engineer   │             │               │
│                     │   features   │             │               │
│                     │ • Preprocess │             ▼               │
│                     │ • SMOTE      │    ┌────────────────────┐   │
│                     │ • Train RF   │    │  predict_proba()   │   │
│                     └──────────────┘    └────────┬───────────┘   │
│                                                  │               │
│   ┌────────────────────┐                         │               │
│   │  Streamlit UI       │                         ▼               │
│   │                     │    ┌────────────────────────────────┐   │
│   │ • Input form (left) │───▶│  build_input_row(params)       │   │
│   │ • Results (right)   │    │  → feature engineering         │   │
│   │ • Risk card         │    │  → preprocessor.transform()    │   │
│   │ • Recommendations   │    │  → model.predict_proba()       │   │
│   │ • Risk breakdown    │    └────────────────────────────────┘   │
│   └────────────────────┘                                         │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Feature Engineering

Seven new features are engineered from the raw data to improve model performance:

| Engineered Feature | Formula / Logic | Why It Helps |
|---|---|---|
| `AvgMonthlyCharges` | `TotalCharges / (tenure + 1)` | Captures the average spend trend over lifetime |
| `ChargeIncrease` | `MonthlyCharges - AvgMonthlyCharges` | Detects recent price hikes that frustrate customers |
| `TotalServices` | Count of services where value = 'Yes' | More services = higher switching cost = lower churn |
| `HasMultipleServices` | `1 if TotalServices >= 3` | Binary flag for bundled customers |
| `IsHighValue` | `1 if MonthlyCharges > 75th percentile` | High-paying customers have different churn patterns |
| `SeniorNoSupport` | `1 if SeniorCitizen=1 AND TechSupport=No` | Seniors without support churn due to frustration |
| `HighRiskProfile` | `1 if Month-to-month AND Electronic check` | The highest-churn combination in telecom data |

---

## ML Pipeline Deep Dive

### Step 1: Data Loading & Cleaning
```python
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
```
- `TotalCharges` has 11 blank strings → converted to `NaN` then filled with `0`
- Target variable `Churn` is binarized: `Yes → 1`, `No → 0`

### Step 2: Preprocessing Pipeline
```python
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])
```
- **Numerical:** Impute missing values with median → Scale using StandardScaler (z-score normalization)
- **Categorical:** Impute missing with mode → One-hot encode with `drop='first'` to avoid multicollinearity

### Step 3: Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    feature_df, y, test_size=0.2, random_state=42, stratify=y
)
```
- 80/20 split with **stratification** to maintain class balance ratio in both sets

### Step 4: SMOTE (Handling Class Imbalance)
```python
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)
```
- Original: 73.5% No, 26.5% Yes → After SMOTE: 50% No, 50% Yes
- SMOTE creates synthetic samples for the minority class (Churn=Yes) using k-nearest neighbors

### Step 5: Model Training
```python
model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
model.fit(X_train_bal, y_train_bal)
```
- **Random Forest** with 150 decision trees
- `n_jobs=-1` uses all CPU cores for parallel training
- `random_state=42` ensures reproducibility

### Step 6: Caching with `@st.cache_resource`
```python
@st.cache_resource(show_spinner=False)
def train_model():
    ...
```
- Model trains **only once** when the app starts
- All subsequent predictions reuse the cached model object
- No `.pkl` file needed — the model lives in memory (avoids pickle version mismatch issues)

---

## Application Walkthrough

### Left Panel — Input Form
The left column contains three input sections:

1. **Customer Demographics** — Gender, Senior Citizen, Partner, Dependents
2. **Account Info** — Tenure, Monthly Charges, Contract Type, Billing, Payment Method
3. **Services Subscribed** — Internet, Phone, Security, Backup, Support, Streaming

### Right Panel — Results Dashboard
After clicking **"ANALYSE CHURN RISK →"**, the right panel displays:

1. **Risk Card** — Large churn probability percentage with color-coded risk level
   - 🔴 **HIGH RISK** (≥ 65%)
   - 🟡 **MEDIUM RISK** (35–64%)
   - 🟢 **LOW RISK** (< 35%)

2. **Metric Pills** — Tenure, Monthly Charge, Service Count, Lifetime Value (LTV)

3. **Retention Recommendations** — Rule-based actionable business advice:
   - Contract risk (month-to-month → annual)
   - New customer onboarding (tenure ≤ 12 months)
   - Security upsell (high-paying, no security)
   - Payment method change (electronic check → auto-pay)
   - Senior care (senior + no tech support)
   - Bundle opportunity (< 3 services)

4. **Risk Breakdown** — Weighted visual bars showing which factors contribute most to churn risk

---

## Code Structure

```
Customer Churn/
├── churn_app.py                              # Main application (528 lines)
│   ├── Imports & config                      # Lines 1–22
│   ├── Custom CSS styling                    # Lines 26–184
│   ├── train_model()                         # Lines 193–258 (cached ML pipeline)
│   ├── build_input_row()                     # Lines 266–315 (feature engineering for prediction)
│   ├── get_recommendations()                 # Lines 321–337 (business logic)
│   └── UI Layout                             # Lines 340–528 (Streamlit interface)
│       ├── Left panel (inputs)               # Lines 356–400
│       └── Right panel (results)             # Lines 405–528
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      # Dataset (7,043 records × 21 columns)
├── requirements.txt                          # Python dependencies
└── .venv/                                    # Virtual environment
```

### Key Functions

| Function | Lines | Purpose |
|---|---|---|
| `train_model()` | 194–258 | Loads CSV, engineers features, builds sklearn pipeline, applies SMOTE, trains Random Forest. Cached with `@st.cache_resource`. |
| `build_input_row(params)` | 266–315 | Converts user inputs from the UI into a feature-engineered DataFrame row that mirrors the training pipeline's feature engineering. |
| `get_recommendations(params, prob)` | 321–337 | Rule-based engine that generates actionable retention strategies based on customer profile. |

---

## Setup & Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Riya-Avasthi/customer-churn-intelligence.git
cd customer-churn-intelligence

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

### Dependencies (requirements.txt)
```
streamlit
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
shap
matplotlib
seaborn
```

---

## How to Run

```bash
# Start the Streamlit application
streamlit run churn_app.py
```

The app will open in your browser at `http://localhost:8501`

**First launch:** The model trains on the dataset (~2-5 seconds). All subsequent interactions use the cached model instantly.

---

## Key Design Decisions

### 1. No Pickle File (In-Memory Model)
Instead of saving/loading a pre-trained `.pkl` file, the model trains at startup and is cached. This avoids:
- Pickle deserialization vulnerabilities
- Version mismatch errors between training and serving environments
- Need for a separate training script

### 2. Feature Engineering Consistency
The `build_input_row()` function replicates the **exact same** feature engineering used in `train_model()`. This ensures the model sees consistent data at training and prediction time — a critical ML best practice called **training-serving skew prevention**.

### 3. SMOTE Over Undersampling
The dataset is imbalanced (73.5% No vs 26.5% Yes). SMOTE was chosen over:
- **Undersampling:** Would discard 3,305 majority samples, losing valuable information
- **Class weights:** Works but SMOTE provides more robust boundary learning
- **No handling:** Would bias the model toward predicting "No Churn"

### 4. Random Forest Over Other Models
- **Interpretability:** Feature importances are directly available
- **Robustness:** Less prone to overfitting than single decision trees
- **No tuning needed:** Performs well with default hyperparameters
- **Handles mixed features:** Works well with both numerical and encoded categorical features

### 5. Custom CSS Dark Theme
The dark terminal-inspired UI was chosen to:
- Create a professional "intelligence dashboard" aesthetic
- Reduce eye strain for analysts using it throughout the day
- Differentiate from default Streamlit styling

---

## Future Enhancements

- [ ] **SHAP Integration** — Add SHAP waterfall plots for individual prediction explanations
- [ ] **XGBoost Toggle** — Allow switching between Random Forest and XGBoost models
- [ ] **Batch Prediction** — Upload a CSV of customers and get bulk churn predictions
- [ ] **Model Comparison Dashboard** — Side-by-side metrics for multiple models
- [ ] **Customer Segmentation** — Add clustering (K-Means) to identify distinct customer groups
- [ ] **Historical Trend Analysis** — Track churn predictions over time
- [ ] **API Endpoint** — Expose the model via FastAPI for integration with other systems
- [ ] **A/B Testing Framework** — Track which retention recommendations actually reduce churn

---

## License

This project uses the [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) dataset, available under the Apache 2.0 License on Kaggle.

---

*Built with ❤️ using Streamlit and Scikit-learn*
