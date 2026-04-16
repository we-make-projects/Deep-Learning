# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# =========================
# 📂 LOAD DATA
# =========================
df1 = pd.read_csv("Combine_Dengue.csv")
df2 = pd.read_csv("Combine_Zika.csv")

df = pd.concat([df1, df2], ignore_index=True)
df.columns = df.columns.str.strip()

df = df.fillna(df.mean(numeric_only=True)).fillna(0)

TARGET = 'Class'
X = df.drop(columns=[TARGET]).values
y = df[TARGET].astype(int).values

# =========================
# 🔥 SCALING (IMPORTANT FOR NN)
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# 🔁 RESHAPE FOR CNN/LSTM
# =========================
X = X.reshape(X.shape[0], X.shape[1], 1)

# =========================
# 🧠 MODEL FUNCTION
# =========================
def build_model(input_shape):
    
    inp = Input(shape=input_shape)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(pool_size=2)(x)

    x = Bidirectional(LSTM(64, return_sequences=False))(x)

    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=inp, outputs=out)

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy'
    )

    return model

# =========================
# 🔁 CROSS VALIDATION
# =========================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

auc_scores = []
acc_scores = []

for train_idx, val_idx in skf.split(X, y):

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = build_model(X.shape[1:])

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    y_prob = model.predict(X_val).ravel()
    y_pred = (y_prob > 0.5).astype(int)

    auc_scores.append(roc_auc_score(y_val, y_prob))
    acc_scores.append(accuracy_score(y_val, y_pred))

# =========================
# 📊 RESULTS
# =========================
print("\n===== CNN + BiLSTM RESULTS =====")
print("AUC:", np.mean(auc_scores))
print("Accuracy:", np.mean(acc_scores))