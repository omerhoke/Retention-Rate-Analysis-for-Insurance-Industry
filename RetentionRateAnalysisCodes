# -----------------------------
# RETENTION RATE ANALYSIS WITH FIXED EFFECTS
# -----------------------------

import pandas as pd
import numpy as np
import statsmodels.api as sm

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("Data.csv")

# Columns
depvar_col = "sales"
state_col = "state"
time_col = "year"

# Check dependent variable
if depvar_col not in df.columns:
    raise ValueError(f"Dependent variable column '{depvar_col}' not found in data.")

# Select variables
base_vars = ["premium", "age", "income", "location", "education", "policytype"]

# Ensure numeric
for col in base_vars + [depvar_col]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop missing
df = df.dropna(subset=base_vars + [depvar_col])

# Dependent variable
y = df[depvar_col]

# =========================================================
# 2. BASE MODEL X MATRIX
# =========================================================
X1 = df[base_vars].copy()

# =========================================================
# 3. ADD STATE FIXED EFFECTS
# =========================================================
state_dummies = pd.get_dummies(df[state_col], prefix="state", drop_first=True)
X2 = pd.concat([X1, state_dummies], axis=1)

# =========================================================
# 4. ADD TIME FIXED EFFECTS
# =========================================================
time_dummies = pd.get_dummies(df[time_col], prefix="year", drop_first=True)
X3 = pd.concat([X2, time_dummies], axis=1)

# =========================================================
# 5. STATE-SPECIFIC TIME TRENDS
# =========================================================
df['year_numeric'] = pd.to_numeric(df[time_col], errors="coerce")

trend_terms = pd.DataFrame()
for col in state_dummies.columns:
    trend_terms[col + "_trend"] = state_dummies[col] * df["year_numeric"]

X4 = pd.concat([X3, trend_terms], axis=1)

# =========================================================
# FUNCTION TO RUN MODELS
# =========================================================
def run_logit(X, y, name):
    X = sm.add_constant(X)
    model = sm.Logit(y, X).fit(disp=False)
    print(f"\n========== {name} ==========")
    print(model.summary())

    # Save results
    params = pd.DataFrame({
        "Coefficient": model.params,
        "Std_Error": model.bse,
        "t_value": model.tvalues,
        "p_value": model.pvalues
    })
    params.to_excel(f"{name}.xlsx")

    # Predicted probabilities
    pred = model.predict(X)
    return model, pred

# =========================================================
# 6. RUN 4 LOGIT REGRESSIONS
# =========================================================
model1, probs1 = run_logit(X1, y, "Model1_Base")
model2, probs2 = run_logit(X2, y, "Model2_StateFE")
model3, probs3 = run_logit(X3, y, "Model3_State_Time_FE")
model4, probs4 = run_logit(X4, y, "Model4_State_Time_Trends")

# =========================================================
# 7. SELECT TOP 30% FOR MARKETING (LOGIT MODELS)
# =========================================================
df["prob_logit4"] = probs4
k = int(0.30 * len(df))
top_idx = df["prob_logit4"].nlargest(k).index
df["send_logit4"] = 0
df.loc[top_idx, "send_logit4"] = 1

df[["prob_logit4", "send_logit4"]].to_excel("logit_targeting.xlsx", index=False)
print("\nSaved logit targeting selection: logit_targeting.xlsx")

# =========================================================
# 8. XGBOOST + AUC + TARGETING
# =========================================================
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

print("\nTraining XGBoost model...")

X_xgb = X1.copy()
y_xgb = y.copy()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_xgb, y_xgb, test_size=0.25, random_state=42)

# Train model
xgb_model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)

# Predict
xgb_probs_test = xgb_model.predict_proba(X_test)[:, 1]

# AUC for XGBoost
xgb_auc = roc_auc_score(y_test, xgb_probs_test)
print(f"XGBoost AUC: {xgb_auc:.4f}")

# AUC for Logit Model 1
logit1_auc = roc_auc_score(y_test, probs1.loc[X_test.index])
print(f"Logit Model 1 AUC: {logit1_auc:.4f}")

# Predict full sample
df["prob_xgb"] = xgb_model.predict_proba(X_xgb)[:, 1]

# Select top 30%
k = int(0.30 * len(df))
top_idx_xgb = df["prob_xgb"].nlargest(k).index
df["send_xgb"] = 0
df.loc[top_idx_xgb, "send_xgb"] = 1

# Save XGBoost summary
xgb_summary = pd.DataFrame({
    "Model": ["XGBoost"],
    "AUC": [xgb_auc],
    "Selected_Customers": [k]
})
xgb_summary.to_excel("xgb_results.xlsx", index=False)

df.to_excel("xgb_targeting.xlsx", index=False)
print("\nXGBoost targeting saved to xgb_targeting.xlsx")

# Done
print("\n=========== ALL MODELS FINISHED SUCCESSFULLY ===========")
