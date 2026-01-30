"""
LIME Explainer — Cross-platform fixed version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lime.lime_tabular import LimeTabularExplainer
import joblib
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
# LIME EXPLAINER
# ==================================================

class LIMEExplainer:

    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.explainer = LimeTabularExplainer(
            X_train.values,
            feature_names=self.feature_names,
            mode="classification",
            random_state=42
        )

    def explain_instance(self, instance, num_features=10, num_samples=3000):

        exp = self.explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
            num_samples=num_samples
        )

        pred = self.model.predict([instance])[0]

        return {
            "explanation": exp,
            "predicted_class": int(pred)
        }

    def plot(self, exp_result, save_name=None):

        exp_result["explanation"].as_pyplot_figure()

        if save_name:
            path = FIGURES_DIR / save_name
            plt.savefig(path, dpi=300, bbox_inches="tight")
            print("Saved:", path)

        plt.show()

    def contributions_df(self, exp_result):

        rows = []
        for feat, val in exp_result["explanation"].as_list():
            rows.append({
                "feature": feat,
                "contribution": val,
                "abs_contribution": abs(val)
            })

        return pd.DataFrame(rows).sort_values(
            "abs_contribution",
            ascending=False
        )


# ==================================================
# MAIN PIPELINE
# ==================================================

def main():

    print("="*80)
    print("LIME EXPLANATIONS")
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

    lime_results = {}

    for name, model in models.items():

        print("\nModel:", name)

        explainer = LIMEExplainer(model, X_train)

        results = []
        for i in range(3):
            exp = explainer.explain_instance(X_test.iloc[i].values)
            explainer.plot(exp, f"lime_{name}_{i}.png")

            results.append({
                "predicted_class": exp["predicted_class"],
                "feature_contributions": explainer.contributions_df(exp)
            })

        lime_results[name] = results

    joblib.dump(lime_results, RESULTS_DIR / "lime_results.pkl")
    print("Saved LIME results.")

    return lime_results


if __name__ == "__main__":
    main()
