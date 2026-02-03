import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
import pickle

# ───────────────────────────────────────────────
# Load and prepare data
# ───────────────────────────────────────────────
# Use a currently working GitHub raw link
url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
# Alternative: url = "https://raw.githubusercontent.com/Ruohan-Yang/Heart-Disease-Data-Set/main/UCI%20Heart%20Disease%20Dataset.csv"

df = pd.read_csv(url)

# Target: 0 = no disease, 1 = disease
X = df.drop('target', axis=1)
y = df['target']

# ───────────────────────────────────────────────
# Feature selection (keeping all 13 features for simplicity, or adjust k)
# ───────────────────────────────────────────────
selector = SelectKBest(score_func=f_classif, k=13)  # or k='all' to skip selection
X_selected = selector.fit_transform(X, y)

# ───────────────────────────────────────────────
# Scaling
# ───────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selected)

# ───────────────────────────────────────────────
# Train / test split
# ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ───────────────────────────────────────────────
# Model
# ───────────────────────────────────────────────
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ───────────────────────────────────────────────
# Quick evaluation
# ───────────────────────────────────────────────
train_acc = model.score(X_train, y_train)
test_acc  = model.score(X_test, y_test)

print(f"Training accuracy:  {train_acc:.4f}")
print(f"Test accuracy:      {test_acc:.4f}")

# ───────────────────────────────────────────────
# Save artifacts (these files are used by your Streamlit app)
# ───────────────────────────────────────────────
pickle.dump(model,    open("heart_model.pkl",   "wb"))
pickle.dump(scaler,   open("scaler.pkl",        "wb"))
pickle.dump(selector, open("selector.pkl",      "wb"))

print("Model, scaler, and selector saved successfully!")