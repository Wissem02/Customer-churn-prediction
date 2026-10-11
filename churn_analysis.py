"""
Customer Churn Prediction & Revenue-at-Risk Analysis
Dataset: IBM Telco Customer Churn (7,043 customers, 21 attributes)

Steps
1. Data cleaning & quality checks
2. Exploratory analysis: churn rate by segment
3. Predictive models: Logistic Regression vs Random Forest
4. Key churn drivers & monthly revenue at risk
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).parent
IMG = ROOT / "images"
IMG.mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#bbb", "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED})

# ---------------------------------------------------------------- 1. Cleaning
df = pd.read_csv(ROOT / "data" / "Telco-Customer-Churn.csv")
quality = {"rows": len(df), "duplicate_ids": int(df["customerID"].duplicated().sum())}

# TotalCharges is stored as text; blanks = brand-new customers (tenure 0)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
quality["blank_total_charges"] = int(df["TotalCharges"].isna().sum())
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["Churn"] = (df["Churn"] == "Yes").astype(int)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                           labels=["0-12 m", "13-24 m", "25-48 m", "49-72 m"])
print("Data quality:", quality)

# ------------------------------------------------------------------- 2. EDA
churn_rate = df["Churn"].mean()
by_contract = df.groupby("Contract")["Churn"].mean().sort_values(ascending=False)
by_tenure = df.groupby("TenureGroup", observed=True)["Churn"].mean()
by_payment = df.groupby("PaymentMethod")["Churn"].mean().sort_values(ascending=False)
by_internet = df.groupby("InternetService")["Churn"].mean().sort_values(ascending=False)
print(f"Overall churn rate: {churn_rate:.1%}")
print(by_contract.round(3), by_tenure.round(3), sep="\n")


def bar(series, title, fname, highlight_max=True):
    fig, ax = plt.subplots(figsize=(6, 3.2))
    colors = [ORANGE if (highlight_max and v == series.max()) else BLUE for v in series.values]
    ax.barh(series.index.astype(str), series.values * 100, color=colors, height=0.6)
    for i, v in enumerate(series.values):
        ax.text(v * 100 + 0.8, i, f"{v:.0%}", va="center", color=INK, fontsize=9)
    ax.axvline(churn_rate * 100, color=MUTED, ls="--", lw=1)
    ax.text(churn_rate * 100, len(series) - 0.4, f" avg {churn_rate:.0%}", color=MUTED, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Churn rate (%)")
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / fname, dpi=150)
    plt.close(fig)


bar(by_contract, "Churn rate by contract type", "churn_by_contract.png")
bar(by_tenure, "Churn rate by customer tenure", "churn_by_tenure.png")
bar(by_payment, "Churn rate by payment method", "churn_by_payment.png")

# ---------------------------------------------------------------- 3. Models
target = "Churn"
drop = ["customerID", target, "TenureGroup"]
X = df.drop(columns=drop)
y = df[target]
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=5,
                                            class_weight="balanced", random_state=42, n_jobs=-1),
}
results, probas = {}, {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)]).fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    results[name] = {"roc_auc": round(roc_auc_score(y_test, p), 3),
                     "recall": round(recall_score(y_test, pred), 3),
                     "precision": round(precision_score(y_test, pred), 3),
                     "f1": round(f1_score(y_test, pred), 3)}
    probas[name] = (p, pipe)
    print(name, results[name])
    print(classification_report(y_test, pred, digits=3))

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_p, best_pipe = probas[best_name]

fig, ax = plt.subplots(figsize=(4.6, 4))
for (name, (p, _)), c in zip(probas.items(), [BLUE, ORANGE]):
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{name} (AUC {results[name]['roc_auc']:.2f})")
ax.plot([0, 1], [0, 1], color="#ccc", ls="--", lw=1)
ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
ax.set_title("ROC curve - test set", loc="left", color=INK, fontsize=11, fontweight="bold")
ax.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(IMG / "roc_curve.png", dpi=150); plt.close(fig)

# ------------------------------------------------- 4. Drivers & revenue at risk
lr_pipe = probas["Logistic Regression"][1]
names = lr_pipe.named_steps["pre"].get_feature_names_out()
coefs = pd.Series(lr_pipe.named_steps["clf"].coef_[0], index=names)
coefs.index = coefs.index.str.replace("num__", "").str.replace("cat__", "").str.replace("_", " = ")
top = pd.concat([coefs.nlargest(6), coefs.nsmallest(6)]).sort_values()

fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.barh(top.index, top.values, color=[ORANGE if v > 0 else BLUE for v in top.values], height=0.6)
ax.axvline(0, color="#999", lw=1)
ax.set_xlabel("Logistic regression coefficient (+ raises churn risk)")
ax.set_title("Top churn drivers", loc="left", color=INK, fontsize=11, fontweight="bold")
fig.tight_layout(); fig.savefig(IMG / "churn_drivers.png", dpi=150); plt.close(fig)

test = X_test.copy()
test["churn_proba"] = best_p
test["actual"] = y_test.values
high_risk = test[test["churn_proba"] >= 0.6]
captured = high_risk["actual"].sum() / test["actual"].sum()
# actual monthly revenue lost to churn, and the share the high-risk list would have flagged
churned_revenue = df.loc[df["Churn"] == 1, "MonthlyCharges"].sum()
test_churned = test[test["actual"] == 1]
revenue_captured = test_churned.loc[test_churned["churn_proba"] >= 0.6, "MonthlyCharges"].sum() / test_churned["MonthlyCharges"].sum()

summary = {
    "data_quality": quality,
    "churn_rate": round(churn_rate, 4),
    "churn_by_contract": by_contract.round(4).to_dict(),
    "churn_by_tenure": {str(k): round(v, 4) for k, v in by_tenure.items()},
    "churn_by_payment": by_payment.round(4).to_dict(),
    "churn_by_internet": by_internet.round(4).to_dict(),
    "models": results,
    "best_model": best_name,
    "high_risk_share_of_customers": round(len(high_risk) / len(test), 4),
    "high_risk_captures_share_of_churners": round(captured, 4),
    "monthly_revenue_lost_to_churn": round(churned_revenue, 0),
    "share_of_churned_revenue_flagged": round(revenue_captured, 4),
    "top_drivers_positive": coefs.nlargest(5).round(3).to_dict(),
}
(ROOT / "results.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
