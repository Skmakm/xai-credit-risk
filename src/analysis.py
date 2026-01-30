"""
Comprehensive Analysis and Visualization for XAI Credit Risk Project
Cross-platform fixed version
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# ✅ Cross-platform paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)


class XAIAnalysis:

    def __init__(self):
        self.model_results = None
        self.shap_results = None
        self.lime_results = None

    # --------------------------------------------------
    # LOAD RESULTS
    # --------------------------------------------------

    def load_results(self):
        self.model_results = joblib.load(RESULTS_DIR / "quick_results.pkl")
        self.shap_results = joblib.load(RESULTS_DIR / "shap_results.pkl")
        self.lime_results = joblib.load(RESULTS_DIR / "lime_results.pkl")

        print("Loaded all results for analysis")
        return self.model_results, self.shap_results, self.lime_results

    # --------------------------------------------------
    # INTERPRETABILITY SCORES
    # --------------------------------------------------

    def calculate_interpretability_scores(self):

        interpretability_scores = {}

        for model_name in self.model_results.keys():

            shap_score = 0
            lime_score = 0

            if model_name in self.shap_results:
                shap_values = self.shap_results[model_name]["shap_values"]
                shap_score = self._feature_consistency(shap_values)

            if model_name in self.lime_results:
                lime_score = self._lime_score(self.lime_results[model_name])

            complexity = {
                "logistic_regression": 1.0,
                "random_forest": 2.0,
                "xgboost": 3.0
            }.get(model_name, 2.0)

            overall = (shap_score * 0.6 + lime_score * 0.4) / complexity

            interpretability_scores[model_name] = {
                "overall_interpretability": overall,
                "shap_interpretability": shap_score,
                "lime_interpretability": lime_score,
                "model_complexity": complexity
            }

        return interpretability_scores

    def _feature_consistency(self, shap_values):
        shap_values = np.array(shap_values)
        return float(np.clip(1 - np.std(np.abs(shap_values)), 0, 1))

    def _lime_score(self, lime_explanations):
        if not lime_explanations:
            return 0
        return float(np.clip(np.mean([
            e["feature_contributions"]["abs_contribution"].mean()
            for e in lime_explanations
        ]) / 10, 0, 1))

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------

    def create_accuracy_vs_interpretability_plot(self):

        scores = self.calculate_interpretability_scores()

        rows = []
        for m in self.model_results:
            rows.append({
                "model": m,
                "accuracy": self.model_results[m]["accuracy"],
                "roc_auc": self.model_results[m]["roc_auc"],
                "interpretability": scores[m]["overall_interpretability"]
            })

        df = pd.DataFrame(rows)

        plt.figure(figsize=(8,6))
        plt.scatter(df["interpretability"], df["accuracy"], s=200)
        for i, r in df.iterrows():
            plt.text(r["interpretability"], r["accuracy"], r["model"])

        plt.xlabel("Interpretability")
        plt.ylabel("Accuracy")
        plt.title("Accuracy vs Interpretability")
        plt.grid(True)

        out = FIGURES_DIR / "accuracy_vs_interpretability.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)
        plt.show()

        return df


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    print("="*80)
    print("COMPREHENSIVE XAI ANALYSIS")
    print("="*80)

    analysis = XAIAnalysis()
    analysis.load_results()
    analysis.create_accuracy_vs_interpretability_plot()

    joblib.dump(
        analysis.calculate_interpretability_scores(),
        RESULTS_DIR / "final_analysis.pkl"
    )

    print("Final analysis saved.")
    return analysis


if __name__ == "__main__":
    main()
