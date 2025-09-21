import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

# Paths
DATA_PATH = os.path.join("data", "winequality-red-selected-missing.csv")
MODEL_PATH = os.path.join("models", "wine_quality_model.pkl")
IMPUTER_PATH = os.path.join("models", "imputer.pkl")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

# 2. Create label: 1 = Good (>=7), 0 = Not Good
df["label"] = (df["quality"] >= 7).astype(int)

# 3. Features & target
X = df.drop(columns=["quality", "label"])
y = df["label"]

# 4. Handle missing values
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# 5. Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 7. Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# 8. Evaluate
y_pred = model.predict(X_test)
print("✅ Training completed!")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 9. Save model & preprocessors
os.makedirs("models", exist_ok=True)
joblib.dump(model, MODEL_PATH)
joblib.dump(imputer, IMPUTER_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Model and preprocessors saved in 'models/' folder")
