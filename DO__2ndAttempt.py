import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb

# ==============================
# LOAD DATA (ONLY ONCE)
# ==============================
def load_data():
    df = pd.read_csv("Combine_Dengue.csv")

    X = df.drop("Class", axis=1).values
    y = df["Class"].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

X, y = load_data()

# ==============================
# DECODE SOLUTION
# ==============================
def decode_solution(sol):
    return {
        "learning_rate": 0.01 + sol[0] * 0.2,      # 0.01–0.21
        "max_depth": int(3 + sol[1] * 7),          # 3–10
        "n_estimators": int(100 + sol[2] * 400),   # 100–500
        "subsample": 0.5 + sol[3] * 0.5,           # 0.5–1.0
        "colsample_bytree": 0.5 + sol[4] * 0.5,    # 0.5–1.0
        "scale_pos_weight": 1 + sol[0] * 5         # 🔥 imbalance handling
    }

# ==============================
# OBJECTIVE FUNCTION (STABLE)
# ==============================
def objective_function(sol):
    params = decode_solution(sol)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            n_estimators=params["n_estimators"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            scale_pos_weight=params["scale_pos_weight"],
            eval_metric="auc",
            use_label_encoder=False,
            verbosity=0
        )

        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]

        aucs.append(roc_auc_score(y_val, preds))

    return np.mean(aucs)

# ==============================
# DANDELION OPTIMIZER
# ==============================
class DandelionOptimizer:
    def __init__(self, obj_func, bounds, pop_size=12, max_iter=20):
        self.obj_func = obj_func
        self.bounds = np.array(bounds)
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = len(bounds)

    def initialize(self):
        return np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            (self.pop_size, self.dim)
        )

    def optimize(self):
        population = self.initialize()
        fitness = np.array([self.obj_func(ind) for ind in population])

        best_idx = np.argmax(fitness)
        best_sol = population[best_idx].copy()
        best_fit = fitness[best_idx]

        for t in range(self.max_iter):
            new_population = []

            for i in range(self.pop_size):
                # 🌱 Exploration
                new_sol = population[i] + np.random.uniform(-0.1, 0.1, self.dim)

                # 🌬️ Move toward best
                new_sol += np.random.rand(self.dim) * (best_sol - population[i])

                # 🌍 Fine tuning
                new_sol += np.random.normal(0, 0.02, self.dim)

                new_sol = np.clip(
                    new_sol,
                    self.bounds[:, 0],
                    self.bounds[:, 1]
                )

                new_population.append(new_sol)

            population = np.array(new_population)
            fitness = np.array([self.obj_func(ind) for ind in population])

            curr_best_idx = np.argmax(fitness)

            if fitness[curr_best_idx] > best_fit:
                best_fit = fitness[curr_best_idx]
                best_sol = population[curr_best_idx].copy()

            print(f"Iteration {t+1}/{self.max_iter} | Best AUC: {best_fit:.4f}")

        return best_sol, best_fit

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":

    bounds = [(0, 1)] * 5

    optimizer = DandelionOptimizer(
        obj_func=objective_function,
        bounds=bounds,
        pop_size=12,   # 🔥 increased
        max_iter=20    # 🔥 increased
    )

    best_sol, best_auc = optimizer.optimize()

    print("\n🔥 FINAL RESULTS")
    print("Best Params:", decode_solution(best_sol))
    print("Best AUC:", best_auc)