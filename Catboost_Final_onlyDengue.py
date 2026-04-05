# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif

from catboost import CatBoostClassifier

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("Combine_Dengue.csv")
df.columns = df.columns.str.strip()

TARGET = 'Class'

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# FEATURE SELECTION
# =========================
selector = SelectKBest(score_func=f_classif, k=15)
X = selector.fit_transform(X, y)

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# OPTUNA OBJECTIVE
# =========================
def objective(trial):
    
    params = {
        "iterations": trial.suggest_int("iterations", 300, 800),
        "depth": trial.suggest_int("depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "verbose": 0
    }
    
    model = CatBoostClassifier(**params)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    auc_scores = []
    acc_scores = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        
        y_prob = model.predict_proba(X_val)[:,1]
        y_pred = (y_prob > 0.5).astype(int)

        auc_scores.append(roc_auc_score(y_val, y_prob))
        acc_scores.append(accuracy_score(y_val, y_pred))

    # 🔥 PRINT BOTH METRICS PER TRIAL
    print(f"AUC: {np.mean(auc_scores):.4f}, Accuracy: {np.mean(acc_scores):.4f}")

    return np.mean(auc_scores)

# =========================
# RUN OPTUNA
# =========================
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

print("\nBest Parameters:", study.best_params)

# =========================
# FINAL MODEL TRAINING
# =========================
best_model = CatBoostClassifier(**study.best_params, verbose=0)
best_model.fit(X, y)

# =========================
# FINAL EVALUATION
# =========================
y_prob = best_model.predict_proba(X)[:,1]
y_pred = (y_prob > 0.5).astype(int)

print("\n===== FINAL RESULTS =====")
print("Final AUC:", roc_auc_score(y, y_prob))
print("Final Accuracy:", accuracy_score(y, y_pred))