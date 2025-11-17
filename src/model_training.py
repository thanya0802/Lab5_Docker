# src/model_training.py
# Train a RandomForest on the Wine dataset (using 4 features)
# and save the model + scaler.

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import joblib
import os


if __name__ == "__main__":
    # -------------------------
    # 1. Load Wine dataset
    # -------------------------
    wine = load_wine()
    X = wine.data        # shape (n_samples, 13)
    y = wine.target      # classes: 0, 1, 2

    # We will use 4 features to keep the UI simple
    # indices: 0=alcohol, 1=malic_acid, 9=color_intensity, 12=proline
    feature_idx = [0, 1, 9, 12]
    X = X[:, feature_idx]

    # -------------------------
    # 2. Train/test split
    # -------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------
    # 3. Scaling
    # -------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------
    # 4. RandomForest model
    # -------------------------
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=8,
        random_state=42
    )
    clf.fit(X_train_scaled, y_train)

    # (Optional) quick sanity check
    acc = clf.score(X_test_scaled, y_test)
    print(f"Test accuracy on Wine dataset: {acc:.3f}")

    # -------------------------
    # 5. Save model + scaler
    # -------------------------
    os.makedirs("model", exist_ok=True)
    bundle = {
        "model": clf,
        "scaler": scaler,
        "feature_idx": feature_idx
    }
    joblib.dump(bundle, "model/wine_model.pkl")
    print("✔ Wine model and scaler saved to model/wine_model.pkl")
