# Explainable AI (XAI) Framework for Credit Risk Prediction

![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-brightgreen.svg)

A comprehensive framework for credit risk prediction with Explainable AI (XAI) techniques, comparing accuracy vs. interpretability trade-offs in high-stakes financial decision-making.

## 🎯 Project Overview

This project implements and compares three machine learning models (Logistic Regression, Random Forest, XGBoost) for credit risk prediction, with state-of-the-art XAI techniques (SHAP and LIME) to provide transparency and interpretability in financial decisions.

### Key Features

- 📊 **Multiple ML Models**: Logistic Regression, Random Forest, XGBoost
- 🔍 **Dual XAI Approach**: SHAP (global explanations) + LIME (local explanations)
- 📈 **Comprehensive Analysis**: Accuracy vs. interpretability trade-offs
- 🎨 **Rich Visualizations**: Feature importance, individual explanations, comparisons
- 📋 **Complete Documentation**: Technical report with methodology and findings
- ⚖️ **Ethical AI**: Class imbalance handling, fairness considerations

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/xai-credit-risk.git
cd xai-credit-risk

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python src/main.py
```

### Requirements

- Python 3.12+
- pandas, numpy, scikit-learn
- xgboost, shap, lime
- matplotlib, seaborn, plotly

See [requirements.txt](requirements.txt) for complete list.

## 📁 Project Structure

```
xai_credit_risk/
├── src/                    # Source code
│   ├── data_loader.py      # Data loading and exploration
│   ├── preprocessor.py     # Data preprocessing pipeline
│   ├── quick_train.py      # Model training
│   ├── shap_explainer.py   # SHAP explanations
│   ├── lime_explainer.py   # LIME explanations
│   └── analysis.py         # Comparative analysis
├── data/                   # Processed data
├── models/                  # Trained models
├── results/                 # Results and explanations
├── figures/                 # Visualizations and plots
├── technical_report.md       # Comprehensive technical report
└── README.md              # This file
```

## 🏗️ Methodology

### 1. Data Pipeline
- **Dataset**: UCI Credit Card Default Dataset (30,000 samples, 24 features)
- **Target**: Default payment next month (binary classification)
- **Preprocessing**: Standard scaling, train-test split (80-20), class balancing

### 2. Machine Learning Models

| Model | Type | Strengths | Accuracy |
|--------|-------|-------------|----------|
| Logistic Regression | Linear | High interpretability | 68.0% |
| Random Forest | Ensemble | Non-linear patterns | 81.3% |
| XGBoost | Gradient Boosting | High performance | 76.0% |

### 3. XAI Techniques

#### SHAP (SHapley Additive exPlanations)
- **Global Explanations**: Feature importance across entire dataset
- **Local Explanations**: Individual prediction explanations
- **Visualizations**: Summary plots, force plots, waterfall charts

#### LIME (Local Interpretable Model-agnostic Explanations)
- **Local Explanations**: Surrogate models for individual instances
- **Rule-based**: Human-readable decision rules
- **Comparison**: Feature contributions across models

## 📊 Key Results

### Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|--------|----------|-----------|--------|-------------|----------|
| Logistic Regression | 0.680 | 0.367 | 0.620 | 0.461 | 0.708 |
| Random Forest | 0.813 | 0.647 | 0.343 | 0.448 | 0.752 |
| XGBoost | 0.760 | 0.465 | 0.573 | 0.513 | 0.753 |

### Interpretability Assessment

| Model | Interpretability Score | SHAP Score | LIME Score | Complexity |
|--------|---------------------|-------------|-------------|------------|
| Logistic Regression | 0.375 | 0.282 | 0.467 | 1.0 |
| Random Forest | 0.075 | 0.148 | 0.493 | 2.0 |
| XGBoost | 0.043 | 0.087 | 0.430 | 3.0 |

### Key Insights

- 🔑 **Most Important Feature**: x6 (Repayment status) consistently across all models
- ⚖️ **Trade-off**: Higher accuracy generally means lower interpretability
- 🎯 **Balanced Choice**: XGBoost offers good accuracy with reasonable interpretability
- 📊 **Explanation Alignment**: SHAP and LIME provide consistent insights

## 🎨 Visualizations

The framework generates comprehensive visualizations:

### Model Performance
- ROC curves comparison
- Confusion matrices
- Performance metric comparisons

### SHAP Explanations
- Feature importance plots
- Summary plots (beeswarm)
- Individual force plots
- Waterfall charts

### LIME Explanations
- Local feature contributions
- Instance-level explanations
- Cross-model comparisons

### Comparative Analysis
- Accuracy vs. interpretability scatter plots
- Radar charts for multi-dimensional comparison
- Executive summary visualizations

![Example Visualization](figures/comprehensive_comparison_radar.png)

## 🔬 How to Use

### Basic Usage

```python
# Load and train models
from src.quick_train import quick_train_models
models, results, X_test, y_test = quick_train_models()

# Generate SHAP explanations
from src.shap_explainer import SHAPExplainer
explainer = SHAPExplainer(models['logistic_regression'], X_train)
shap_values, X_sample = explainer.calculate_shap_values(X_test)
explainer.plot_summary()

# Generate LIME explanations
from src.lime_explainer import LIMEExplainer
lime_exp = LIMEExplainer(models['logistic_regression'], X_train)
explanation = lime_exp.explain_instance(X_test.iloc[0])
lime_exp.plot_explanation(explanation)
```

### Advanced Analysis

```python
# Comparative analysis
from src.analysis import XAIAnalysis
analysis = XAIAnalysis()
analysis.load_results()
analysis.create_accuracy_vs_interpretability_plot()
analysis.generate_executive_summary()
```

## 📋 Ethical Considerations

This project addresses key ethical AI concerns:

- **Fairness**: Class imbalance handling to prevent bias
- **Transparency**: Multiple explanation methods for robustness
- **Accountability**: Clear audit trails through SHAP/LIME
- **Regulatory Compliance**: Interpretable models for financial regulations

## 🏆 Why This Project Stands Out

### For Academic Admissions Committees

- ✅ **Theoretical Understanding**: Comprehensive XAI implementation
- ✅ **Ethical & Responsible AI**: Addresses EU AI Act requirements
- ✅ **Practical Skills**: End-to-end machine learning pipeline
- ✅ **Research Quality**: Systematic evaluation and analysis
- ✅ **Modern Techniques**: State-of-the-art explanation methods

### For Industry Applications

- 🎯 **Real-world Ready**: Production-quality code
- 🔧 **Modular Design**: Easy to extend and customize
- 📊 **Comprehensive Metrics**: Both performance and interpretability
- 📋 **Documentation**: Clear implementation guidance

## 📚 References

1. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
2. Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?". KDD.
3. European Commission (2019). Ethics Guidelines for Trustworthy AI.

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the Credit Card Default dataset
- SHAP and LIME development teams for excellent XAI libraries
- Scikit-learn, XGBoost communities for robust ML frameworks

---

**Note**: This project demonstrates best practices in Explainable AI for high-stakes decision systems. The comprehensive analysis provides insights for both technical and non-technical stakeholders in financial risk assessment.

*Project completed for Data Science & AI Masters applications* 🎓