# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score

from catboost import CatBoostClassifier

# =========================
# 📂 LOAD DATA
# =========================
df1 = pd.read_csv("Combine_Dengue.csv")
df2 = pd.read_csv("Combine_Zika.csv")

df = pd.concat([df1, df2], ignore_index=True)

df.columns = df.columns.str.strip()
df = df.fillna(df.mean(numeric_only=True)).fillna(0)

# =========================
# 🎯 TARGET
# =========================
TARGET = 'Class'
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

# =========================
# 🚀 OPTUNA OBJECTIVE
# =========================
def objective(trial):

    params = {
        "iterations": trial.suggest_int("iterations", 500, 1200),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 15),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0, 1),
        "random_strength": trial.suggest_float("random_strength", 0, 1),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "verbose": 0
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []
    acc_scores = []

    for train_idx, val_idx in skf.split(X, y):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)

        model.fit(
            X_train, y_train,
            eval_set=(X_val, y_val),
            early_stopping_rounds=50,
            verbose=0
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_val, y_prob))
        acc_scores.append(accuracy_score(y_val, y_pred))

    print(f"AUC: {np.mean(auc_scores):.4f}, Acc: {np.mean(acc_scores):.4f}")

    return np.mean(auc_scores)

# =========================
# 🔥 RUN OPTUNA
# =========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best Params:", study.best_params)

# =========================
# 📊 FINAL EVALUATION
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

final_auc = []
final_acc = []

for train_idx, val_idx in skf.split(X, y):

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = CatBoostClassifier(**study.best_params, verbose=0)

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=50,
        verbose=0
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    final_auc.append(roc_auc_score(y_val, y_prob))
    final_acc.append(accuracy_score(y_val, y_pred))

print("\n===== FINAL RESULTS =====")
print("Final AUC:", np.mean(final_auc))
print("Final Accuracy:", np.mean(final_acc))