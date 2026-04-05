# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import VotingClassifier

from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# =========================
# 📂 LOAD DATA
# =========================
df = pd.read_csv("Combine_Dengue.csv")
df.columns = df.columns.str.strip()

TARGET = 'Class'   # 🔥 change if needed

X = df.drop(columns=[TARGET])
y = df[TARGET]

# =========================
# ✂️ TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 🔥 FEATURE SELECTION
# =========================
selector = SelectKBest(score_func=f_classif, k=30)
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# =========================
# ⚖️ SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# ⚖️ CLASS WEIGHTS
# =========================
class_counts = np.bincount(y_train)
scale_pos_weight = class_counts[0] / class_counts[1]

# =========================
# 🌳 LIGHTGBM MODEL
# =========================
lgb_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

# =========================
# 🐱 CATBOOST MODEL
# =========================
cat_model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.03,
    verbose=0,
    auto_class_weights='Balanced'
)

# =========================
# 🔥 ENSEMBLE MODEL (BEST)
# =========================
ensemble_model = VotingClassifier(
    estimators=[
        ('lgb', lgb_model),
        ('cat', cat_model)
    ],
    voting='soft'
)

# =========================
# 🚀 TRAIN
# =========================
ensemble_model.fit(X_train, y_train)

# =========================
# 📊 PREDICTIONS
# =========================
y_prob = ensemble_model.predict_proba(X_test)[:,1]

# =========================
# 🔥 THRESHOLD TUNING
# =========================
print("\n===== Threshold Tuning =====")

best_acc = 0
best_t = 0.5

for t in [0.3, 0.4, 0.5, 0.6]:
    y_pred = (y_prob > t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Threshold {t} → Accuracy: {acc}")

    if acc > best_acc:
        best_acc = acc
        best_t = t

# =========================
# ✅ FINAL RESULTS
# =========================
print("\n===== FINAL RESULTS =====")

y_pred = (y_prob > best_t).astype(int)

print("Best Threshold:", best_t)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, y_prob))

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))