# =========================
# 📦 IMPORTS
# =========================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

# =========================
# 📂 LOAD DATA
# =========================

df = pd.read_csv("Combine_Dengue.csv")
print(df.columns)

# 🔥 IMPORTANT: change 'label' to your actual target column name
X = df.drop(columns=['Class'])   # replace 'Class' with actual name
y = df['Class']   # replace 'Class' with actual name

# =========================
# ✂️ TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# ⚖️ FEATURE SCALING
# =========================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# 🔄 RESHAPE FOR CNN + LSTM
# =========================
num_features = X_train.shape[1]

X_train = X_train.reshape(X_train.shape[0], num_features, 1)
X_test = X_test.reshape(X_test.shape[0], num_features, 1)

# =========================
# ⚖️ HANDLE CLASS IMBALANCE
# =========================
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

print("Class Weights:", class_weights)

# =========================
# 🧠 MODEL BUILDING
# =========================
model = Sequential()

# CNN layers
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(num_features, 1)))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(128, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))

# BiLSTM layer
model.add(Bidirectional(LSTM(64)))

# Regularization
model.add(Dropout(0.4))

# Dense layers
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# =========================
# ⚙️ COMPILE MODEL
# =========================
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# =========================
# 🚀 TRAIN MODEL
# =========================
history = model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    verbose=1
)

# =========================
# 📊 EVALUATION
# =========================
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print("\n✅ Accuracy:", accuracy)
print("✅ AUC Score:", auc)

print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))