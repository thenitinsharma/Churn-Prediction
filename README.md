# Telco Customer Churn Prediction

A churn prediction project on the IBM Telco Customer Churn dataset, comparing Random Forest and XGBoost classifiers, with feature importance analysis and a customer segmentation layer built on top of the model's churn probabilities.

## Dataset

- **Source**: `Telco_customer_churn.xlsx`
- **Size**: 7,043 customers, 33 raw columns
- After dropping identifier/leakage columns (CustomerID, Count, Country, State, Zip Code, Lat Long, Latitude, Longitude, Churn Label, Churn Score, CLTV, Churn Reason, City), the working set has 20 feature columns.
- Target: `Churn Value` (1 = churned, 0 = retained)
- Class balance: 5,174 retained vs 1,869 churned (~26.5% churn rate) — a moderately imbalanced problem.

## Workflow

### 1. Exploratory Data Analysis
- Distribution of `Tenure Months` and `Monthly Charges` (overall and split by churn label)
- Churned customers skew toward **higher monthly charges** (median ~₹79.65 vs ~₹64.43 for non-churners) and **shorter tenure**
- Contract type vs churn count comparison — month-to-month customers churn far more than one/two-year contract holders

### 2. Data Cleaning
- `Total Charges` had 11 rows with non-numeric/missing values; coerced to numeric and filled with 0
- One-hot encoded all categorical columns (`pd.get_dummies`, `drop_first=True`), expanding to 31 columns

### 3. Modeling — Random Forest (4 approaches)
| Approach | Configuration | Accuracy | Recall (churn) | F1 (churn) |
|---|---|---|---|---|
| Baseline | `n_estimators=100, max_depth=3` | 0.77 | 0.31 | 0.44 |
| 1. Class-weight balancing | `class_weight='balanced'` | 0.79 | 0.52 | 0.59 |
| 2. Hyperparameter tuning | `n_estimators=300, max_depth=10, class_weight='balanced'` | 0.78 | 0.75 | 0.66 |
| 3. Feature selection | Dropped 2 low-importance features (`Phone Service_Yes`, `Multiple Lines_No phone service`) | 0.78 | 0.74 | 0.66 |

A grid search over `n_estimators ∈ {100,200,300,400,500}` and `max_depth ∈ {5,10,15,20}`, ranked by recall then accuracy, confirmed `n_estimators=300, max_depth=10` as the best-performing tuned configuration (**rf_tuning**), used as the final Random Forest model.

**Final Random Forest (rf_tuning) — 5-fold CV:**
- CV Accuracy: 0.779
- CV Recall: 0.734
- ROC AUC: 0.857

**Top features driving churn (Random Forest):** Tenure Months, Total Charges, two-year contract, Monthly Charges, having dependents, fiber optic internet service, and electronic check payment method.

### 4. Modeling — XGBoost
- `XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=2.84, eval_metric='logloss')`
- `scale_pos_weight` computed from the train-set class ratio to handle imbalance
- Test accuracy: 0.76 | Recall (churn): 0.77 | F1 (churn): 0.65
- ROC AUC: 0.855
- 5-fold CV Accuracy: 0.762 | CV Recall: 0.758

**Top features (XGBoost):** two-year contract, fiber optic internet, "no internet service" indicators (online security/backup), one-year contract, and having dependents — broadly consistent with the Random Forest ranking.

### 5. Final Model Comparison

| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|---|
| Random Forest (tuned) | 0.783 | 0.593 | 0.748 | 0.662 | 0.857 |
| XGBoost | 0.764 | 0.562 | 0.773 | 0.651 | 0.855 |

ROC curves for both models and a grouped bar chart comparing all five metrics are plotted side by side for visual comparison.

**Takeaway:** Random Forest (tuned) edges out XGBoost on accuracy, precision, F1, and AUC, while XGBoost achieves slightly higher recall — useful if catching more true churners (at the cost of more false positives) is the priority.

### 6. Customer Segmentation
Using the tuned Random Forest's predicted churn probabilities, customers were clustered with **K-Means** (k chosen via the elbow method, k=3) on standardized `Tenure Months`, `Monthly Charges`, `Total Charges`, and `Churn Probability`:

| Segment | Avg. Tenure | Avg. Monthly Charges | Avg. Total Charges | Avg. Churn Probability |
|---|---|---|---|---|
| Budget Loyal Customers | 32.1 months | ₹32.85 | ₹1,047.70 | 0.121 |
| High Risk New Customers | 11.0 months | ₹71.96 | ₹884.07 | 0.691 |
| Loyal Premium Customers | 58.4 months | ₹90.43 | ₹5,278.00 | 0.231 |

Scatter plots visualize each segment against Monthly Charges, Tenure, and Total Charges vs. predicted churn probability — **"High Risk New Customers"** stands out as the clearest target for retention efforts.

## Tech Stack
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Modeling**: scikit-learn (RandomForestClassifier, KMeans, StandardScaler), XGBoost
- **Environment**: Google Colab

## Repository Structure
```
.
├── Telco_customer_churn.xlsx     # raw dataset
├── churn_prediction.ipynb        # main notebook (EDA → modeling → segmentation)
└── README.md
```

## Key Findings
- Contract type and tenure are the strongest churn predictors — month-to-month, short-tenure customers churn the most.
- Higher monthly charges correlate with higher churn likelihood.
- Class imbalance handling (balanced class weights / `scale_pos_weight`) was essential — the unweighted baseline missed most actual churners despite "good" accuracy.
- Random Forest (tuned) and XGBoost perform comparably overall; the choice between them depends on whether precision or recall matters more for the business use case.
- Segmenting customers by churn risk surfaces an actionable group — high-paying, low-tenure customers — for targeted retention campaigns.
