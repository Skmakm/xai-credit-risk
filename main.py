"""
Main execution script for XAI Credit Risk Project
This script runs the complete pipeline from data loading to final analysis.
"""

import sys
from pathlib import Path
import joblib

# --------------------------------------------------
# ✅ Cross-platform project paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# create folders if missing
for d in [DATA_DIR, MODELS_DIR, RESULTS_DIR, FIGURES_DIR]:
    d.mkdir(exist_ok=True)

# allow src imports (cross-platform)
sys.path.append(str(BASE_DIR))

# --------------------------------------------------
# imports
# --------------------------------------------------
from src.data_loader import CreditRiskDataLoader
from src.preprocessor import CreditRiskPreprocessor
from src.quick_train import quick_train_models
from src.shap_explainer import main as shap_main
from src.lime_explainer import main as lime_main
from src.analysis import main as analysis_main


def main():
    print("="*80)
    print("EXPLAINABLE AI (XAI) FOR CREDIT RISK PREDICTION")
    print("COMPLETE PIPELINE EXECUTION")
    print("="*80)

    # --------------------------------------------------
    # STEP 1 — Data Loading
    # --------------------------------------------------
    print("\n🔍 STEP 1: DATA LOADING AND EXPLORATION")
    print("-" * 50)

    loader = CreditRiskDataLoader()
    data = loader.load_uci_credit_card()
    loader.explore_data(save_plots=True)

    # --------------------------------------------------
    # STEP 2 — Preprocessing
    # --------------------------------------------------
    print("\n⚙️ STEP 2: DATA PREPROCESSING")
    print("-" * 50)

    preprocessor = CreditRiskPreprocessor(scaler_type="standard")
    preprocessor.target_column = "default"
    preprocessor.feature_columns = [c for c in data.columns if c != "default"]

    X_train, X_test, y_train, y_test = preprocessor.split_data_transformed(
        data, test_size=0.2, random_state=42
    )

    preprocessed_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": X_train.columns.tolist()
    }

    joblib.dump(preprocessed_data, DATA_DIR / "preprocessed_data.pkl")
    joblib.dump(preprocessor, DATA_DIR / "preprocessor.pkl")

    print("✅ Preprocessing completed and saved")

    # --------------------------------------------------
    # STEP 3 — Model Training
    # --------------------------------------------------
    print("\n🤖 STEP 3: MODEL TRAINING")
    print("-" * 50)

    models, results, X_test_final, y_test_final = quick_train_models()
    print("✅ Model training completed")

    # --------------------------------------------------
    # STEP 4 — SHAP
    # --------------------------------------------------
    print("\n🔍 STEP 4: SHAP EXPLANATIONS")
    print("-" * 50)

    try:
        shap_main()
        print("✅ SHAP explanations completed")
    except Exception as e:
        print(f"⚠️ SHAP explanations failed: {e}")

    # --------------------------------------------------
    # STEP 5 — LIME
    # --------------------------------------------------
    print("\n🔍 STEP 5: LIME EXPLANATIONS")
    print("-" * 50)

    try:
        lime_main()
        print("✅ LIME explanations completed")
    except Exception as e:
        print(f"⚠️ LIME explanations failed: {e}")

    # --------------------------------------------------
    # STEP 6 — Analysis
    # --------------------------------------------------
    print("\n📊 STEP 6: COMPREHENSIVE ANALYSIS")
    print("-" * 50)

    try:
        analysis_main()
        print("✅ Comprehensive analysis completed")
    except Exception as e:
        print(f"⚠️ Analysis failed: {e}")

    # --------------------------------------------------
    # STEP 7 — Summary
    # --------------------------------------------------
    print("\n📋 STEP 7: PROJECT SUMMARY")
    print("-" * 50)

    try:
        model_results = joblib.load(RESULTS_DIR / "quick_results.pkl")

        best_acc = max(model_results[m]["accuracy"] for m in model_results)
        best_auc = max(model_results[m]["roc_auc"] for m in model_results)

        print("\n🏆 FINAL RESULTS SUMMARY:")
        print("📊 Dataset: UCI Credit Card Default (30,000 samples)")
        print(f"🎯 Best Accuracy: {best_acc:.3f}")
        print(f"🚀 Best ROC-AUC: {best_auc:.3f}")

        print("\n🎨 GENERATED OUTPUTS:")
        print("• Models → models/")
        print("• Explanations → results/")
        print("• Visualizations → figures/")
        print("• Technical Report → technical_report.md")
        print("• Documentation → README.md")

        print("\n✅ PROJECT COMPLETED SUCCESSFULLY!")

    except Exception as e:
        print(f"⚠️ Could not generate final summary: {e}")

    print("\n" + "="*80)
    print("THANK YOU FOR USING THE XAI CREDIT RISK FRAMEWORK")
    print("="*80)


if __name__ == "__main__":
    main()
