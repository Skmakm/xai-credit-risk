"""
Model Training Module — Cross-platform fixed
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# ✅ Cross-platform paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)


# ==================================================
# TRAINER
# ==================================================

class ModelTrainer:

    def __init__(self):
        self.models = {}

    def train_all(self, X_train, y_train):

        print("Training Logistic Regression...")
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(X_train, y_train)
        self.models["logistic_regression"] = lr

        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        rf.fit(X_train, y_train)
        self.models["random_forest"] = rf

        print("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )
        xgb_model.fit(X_train, y_train)
        self.models["xgboost"] = xgb_model

        return self.models

    def save_models(self):

        for name, model in self.models.items():
            path = MODELS_DIR / f"{name}.pkl"
            joblib.dump(model, path)
            print("Saved:", path)


# ==================================================
# EVALUATOR
# ==================================================

class ModelEvaluator:

    @staticmethod
    def evaluate(models, X_test, y_test):

        results = {}

        for name, model in models.items():

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:,1]

            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba)
            }

        df = pd.DataFrame(results).T
        print(df)

        joblib.dump(df, RESULTS_DIR / "quick_results.pkl")

        return df

    @staticmethod
    def plot_roc(models, X_test, y_test):

        plt.figure()

        for name, model in models.items():
            proba = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)
            plt.plot(fpr, tpr, label=f"{name} ({auc:.3f})")

        plt.plot([0,1],[0,1],"--")
        plt.legend()
        plt.title("ROC Curves")

        out = FIGURES_DIR / "roc_curves.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)
        plt.show()


# ==================================================
# MAIN
# ==================================================

def main():

    data = joblib.load(DATA_DIR / "preprocessed_data.pkl")

    X_train = data["X_train"]
    X_test  = data["X_test"]
    y_train = data["y_train"]
    y_test  = data["y_test"]

    trainer = ModelTrainer()
    models = trainer.train_all(X_train, y_train)
    trainer.save_models()

    ModelEvaluator.evaluate(models, X_test, y_test)
    ModelEvaluator.plot_roc(models, X_test, y_test)

    print("Training + evaluation complete.")

    return models


if __name__ == "__main__":
    main()
