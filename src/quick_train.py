"""
Quick Model Training — Cross-platform fixed
Simplified version without heavy hyperparameter tuning
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# ✅ Cross-platform paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    d.mkdir(exist_ok=True)


# ==================================================
# QUICK TRAIN
# ==================================================

def quick_train_models():

    # Load preprocessed data
    preprocessed_data = joblib.load(DATA_DIR / "preprocessed_data.pkl")

    X_train = preprocessed_data["X_train"]
    X_test  = preprocessed_data["X_test"]
    y_train = preprocessed_data["y_train"]
    y_test  = preprocessed_data["y_test"]

    print("="*60)
    print("QUICK MODEL TRAINING")
    print("="*60)
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)

    models = {}

    # ---------------------------------------------
    # Logistic Regression
    # ---------------------------------------------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(
        random_state=42,
        max_iter=1000,
        class_weight="balanced"
    )
    lr.fit(X_train, y_train)
    models["logistic_regression"] = lr

    # ---------------------------------------------
    # Random Forest
    # ---------------------------------------------
    print("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=120,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    models["random_forest"] = rf

    # ---------------------------------------------
    # XGBoost
    # ---------------------------------------------
    print("Training XGBoost...")
    scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    xgb_model = xgb.XGBClassifier(
        n_estimators=120,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss"
    )

    xgb_model.fit(X_train, y_train)
    models["xgboost"] = xgb_model

    # ==================================================
    # EVALUATION
    # ==================================================

    print("\nMODEL EVALUATION")
    print("="*60)

    results = {}

    for name, model in models.items():

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }

        results[name] = metrics

        print(f"\n{name.upper()}")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

    # ==================================================
    # SAVE
    # ==================================================

    for name, model in models.items():
        joblib.dump(model, MODELS_DIR / f"{name}.pkl")

    joblib.dump(results, RESULTS_DIR / "quick_results.pkl")
    joblib.dump(
        {"X_test": X_test, "y_test": y_test},
        RESULTS_DIR / "test_data.pkl"
    )

    print("\nSaved models + results")

    return models, results, X_test, y_test


if __name__ == "__main__":
    quick_train_models()
