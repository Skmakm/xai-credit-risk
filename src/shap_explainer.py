"""
SHAP Explainer — Cross-platform fixed
"""

import pandas as pd
import numpy as np
import shap
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
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
# SHAP EXPLAINER
# ==================================================

class SHAPExplainer:

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        self.explainer = None

    def create(self):

        name = type(self.model).__name__.lower()

        if "forest" in name or "xgb" in name or "tree" in name:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.LinearExplainer(self.model, self.X_train)

        return self.explainer

    def compute(self, X, sample_size=80):

        if self.explainer is None:
            self.create()

        Xs = X.sample(min(sample_size, len(X)), random_state=42)

        shap_values = self.explainer.shap_values(Xs)

        # binary classifier handling
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        return shap_values, Xs

    def summary_plot(self, shap_values, Xs, name):

        out = FIGURES_DIR / f"shap_summary_{name}.png"

        shap.summary_plot(shap_values, Xs, show=False)
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()

        print("Saved:", out)


# ==================================================
# MAIN
# ==================================================

def main():

    print("="*80)
    print("SHAP EXPLANATIONS")
    print("="*80)

    test_data = joblib.load(RESULTS_DIR / "test_data.pkl")
    X_test = test_data["X_test"]

    pre = joblib.load(DATA_DIR / "preprocessed_data.pkl")
    X_train = pre["X_train"]

    models = {
        "logistic_regression": joblib.load(MODELS_DIR / "logistic_regression.pkl"),
        "random_forest": joblib.load(MODELS_DIR / "random_forest.pkl"),
        "xgboost": joblib.load(MODELS_DIR / "xgboost.pkl"),
    }

    shap_results = {}

    for name, model in models.items():

        print("\nModel:", name)

        exp = SHAPExplainer(model, X_train)
        shap_vals, Xs = exp.compute(X_test)

        exp.summary_plot(shap_vals, Xs, name)

        shap_results[name] = {
            "shap_values": shap_vals,
            "feature_names": list(X_train.columns)
        }

    joblib.dump(shap_results, RESULTS_DIR / "shap_results.pkl")
    print("\nSaved SHAP results")

    return shap_results


if __name__ == "__main__":
    main()
