"""
LIME (Local Interpretable Model-agnostic Explanations) Implementation for Credit Risk Models
This module provides comprehensive LIME explanations for all trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lime import lime_tabular
from lime.lime_tabular import LimeTabularExplainer
import joblib
import warnings
warnings.filterwarnings('ignore')

class LIMEExplainer:
    """
    Comprehensive LIME explanation class for credit risk models
    """
    
    def __init__(self, model, X_train, feature_names=None, mode='classification'):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.mode = mode
        self.explainer = None
        
    def create_explainer(self, discretize_continuous=True):
        """
        Create LIME explainer
        """
        self.explainer = LimeTabularExplainer(
            self.X_train.values,
            feature_names=self.feature_names,
            mode=self.mode,
            discretize_continuous=discretize_continuous,
            random_state=42
        )
        
        print(f"Created LIME explainer for {type(self.model).__name__}")
        return self.explainer
    
    def explain_instance(self, instance, num_features=10, num_samples=5000):
        """
        Explain a single prediction instance
        """
        if self.explainer is None:
            self.create_explainer()
        
        print(f"Explaining instance with LIME...")
        
        # Get prediction for the instance
        if hasattr(self.model, 'predict_proba'):
            prediction = self.model.predict_proba([instance])[0]
            predicted_class = np.argmax(prediction)
        else:
            prediction = self.model.predict([instance])[0]
            predicted_class = prediction
        
        # Create explanation
        explanation = self.explainer.explain_instance(
            instance,
            self.model.predict_proba if hasattr(self.model, 'predict_proba') else self.model.predict,
            num_features=num_features,
            num_samples=num_samples
        )
        
        return {
            'explanation': explanation,
            'instance': instance,
            'predicted_class': predicted_class,
            'prediction_proba': prediction if hasattr(self.model, 'predict_proba') else None
        }
    
    def plot_explanation(self, explanation_result, save_path=None):
        """
        Plot LIME explanation for an instance
        """
        explanation = explanation_result['explanation']
        
        plt.figure(figsize=(12, 8))
        explanation.as_pyplot_figure()
        plt.title(f"LIME Explanation - Predicted Class: {explanation_result['predicted_class']}")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME explanation plot saved to {save_path}")
        
        plt.show()
    
    def get_feature_contributions(self, explanation_result):
        """
        Extract feature contributions from LIME explanation
        """
        explanation = explanation_result['explanation']
        
        contributions = []
        for feature, importance in explanation.as_list():
            contributions.append({
                'feature': feature,
                'contribution': importance,
                'abs_contribution': abs(importance)
            })
        
        return pd.DataFrame(contributions).sort_values('abs_contribution', ascending=False)
    
    def explain_multiple_instances(self, instances, num_features=10, num_samples=5000):
        """
        Explain multiple instances and return summary
        """
        if self.explainer is None:
            self.create_explainer()
        
        all_explanations = []
        
        for i, instance in enumerate(instances):
            print(f"Explaining instance {i+1}/{len(instances)}...")
            exp_result = self.explain_instance(instance, num_features, num_samples)
            exp_result['instance_index'] = i
            all_explanations.append(exp_result)
        
        return all_explanations

class LIMEComparison:
    """
    Compare LIME explanations across different models
    """
    
    def __init__(self):
        self.explainers = {}
        self.results = {}
    
    def add_model(self, model_name, model, X_train, feature_names=None):
        """
        Add a model for LIME comparison
        """
        explainer = LIMEExplainer(model, X_train, feature_names)
        self.explainers[model_name] = explainer
        print(f"Added {model_name} to LIME comparison")
    
    def explain_all_models(self, instances, num_features=10, num_samples=5000):
        """
        Explain instances for all models
        """
        for model_name, explainer in self.explainers.items():
            print(f"\nExplaining instances with {model_name}...")
            explanations = explainer.explain_multiple_instances(
                instances, num_features, num_samples
            )
            self.results[model_name] = explanations
    
    def compare_feature_importance(self, instance_idx=0, save_path=None):
        """
        Compare feature importance for a specific instance across models
        """
        comparison_data = []
        
        for model_name, explanations in self.results.items():
            if instance_idx < len(explanations):
                exp_result = explanations[instance_idx]
                contributions = self.explainers[model_name].get_feature_contributions(exp_result)
                
                for _, row in contributions.head(10).iterrows():
                    comparison_data.append({
                        'model': model_name,
                        'feature': row['feature'],
                        'contribution': row['contribution'],
                        'abs_contribution': row['abs_contribution']
                    })
        
        if not comparison_data:
            print(f"No explanations found for instance {instance_idx}")
            return None
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Create comparison plot
        plt.figure(figsize=(15, 10))
        
        # Get top features by average importance
        top_features = comparison_df.groupby('feature')['abs_contribution'].mean().nlargest(8).index
        filtered_df = comparison_df[comparison_df['feature'].isin(top_features)]
        
        # Create grouped bar plot
        pivot_df = filtered_df.pivot(index='feature', columns='model', values='contribution')
        pivot_df.plot(kind='bar', figsize=(15, 8))
        plt.title(f'LIME Feature Contributions Comparison - Instance {instance_idx}')
        plt.xlabel('Features')
        plt.ylabel('Contribution')
        plt.xticks(rotation=45)
        plt.legend(title='Models')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"LIME feature comparison saved to {save_path}")
        
        plt.show()
        return comparison_df
    
    def create_individual_plots(self, instance_idx=0, save_dir='/home/akashmis/xai_credit_risk/figures'):
        """
        Create individual LIME plots for all models
        """
        for model_name, explanations in self.results.items():
            if instance_idx < len(explanations):
                exp_result = explanations[instance_idx]
                explainer = self.explainers[model_name]
                
                save_path = f"{save_dir}/lime_individual_{model_name}.png"
                explainer.plot_explanation(exp_result, save_path)
    
    def create_summary_report(self, save_path=None):
        """
        Create comprehensive LIME summary report
        """
        print("="*80)
        print("LIME EXPLANATIONS COMPARISON REPORT")
        print("="*80)
        
        for model_name, explanations in self.results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            
            # Analyze first few explanations
            for i in range(min(3, len(explanations))):
                exp_result = explanations[i]
                contributions = self.explainers[model_name].get_feature_contributions(exp_result)
                
                print(f"\nInstance {i} - Predicted Class: {exp_result['predicted_class']}")
                print("Top 5 Most Important Features:")
                for _, row in contributions.head(5).iterrows():
                    sign = "+" if row['contribution'] > 0 else "-"
                    print(f"  {row['feature']}: {sign}{abs(row['contribution']):.4f}")
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write("LIME EXPLANATIONS COMPARISON REPORT\n")
                f.write("="*80 + "\n")
                for model_name, explanations in self.results.items():
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    
                    for i in range(min(3, len(explanations))):
                        exp_result = explanations[i]
                        contributions = self.explainers[model_name].get_feature_contributions(exp_result)
                        
                        f.write(f"\nInstance {i} - Predicted Class: {exp_result['predicted_class']}\n")
                        f.write("Top 5 Most Important Features:\n")
                        for _, row in contributions.head(5).iterrows():
                            sign = "+" if row['contribution'] > 0 else "-"
                            f.write(f"  {row['feature']}: {sign}{abs(row['contribution']):.4f}\n")
            
            print(f"\nLIME report saved to {save_path}")

def main():
    """
    Demonstrate LIME explanations for all models
    """
    # Load data and models
    test_data = joblib.load('/home/akashmis/xai_credit_risk/results/test_data.pkl')
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    preprocessed_data = joblib.load('/home/akashmis/xai_credit_risk/data/preprocessed_data.pkl')
    X_train = preprocessed_data['X_train']
    feature_names = X_train.columns.tolist()
    
    print("="*80)
    print("LIME EXPLANATIONS FOR CREDIT RISK MODELS")
    print("="*80)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create LIME comparison
    comparison = LIMEComparison()
    
    # Load and add models
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    for model_name in model_names:
        model = joblib.load(f'/home/akashmis/xai_credit_risk/models/{model_name}.pkl')
        comparison.add_model(model_name, model, X_train, feature_names)
    
    # Select instances to explain (mix of correct/incorrect predictions)
    instances_to_explain = []
    instance_indices = []
    
    # Get first few instances from test set
    for i in range(min(5, len(X_test))):
        instances_to_explain.append(X_test.iloc[i].values)
        instance_indices.append(i)
    
    print(f"\nExplaining {len(instances_to_explain)} instances...")
    
    # Explain instances for all models
    comparison.explain_all_models(instances_to_explain, num_features=8, num_samples=2000)
    
    # Create visualizations
    print("\nGenerating LIME visualizations...")
    
    # Individual model explanations
    comparison.create_individual_plots(instance_idx=0)
    
    # Compare feature contributions across models
    comparison.compare_feature_importance(
        instance_idx=0,
        save_path='/home/akashmis/xai_credit_risk/figures/lime_comparison.png'
    )
    
    # Create summary report
    comparison.create_summary_report(
        save_path='/home/akashmis/xai_credit_risk/results/lime_report.txt'
    )
    
    # Save LIME results
    lime_results = {}
    for model_name, explanations in comparison.results.items():
        lime_results[model_name] = []
        for exp in explanations:
            lime_results[model_name].append({
                'predicted_class': exp['predicted_class'],
                'prediction_proba': exp['prediction_proba'],
                'feature_contributions': comparison.explainers[model_name].get_feature_contributions(exp)
            })
    
    joblib.dump(lime_results, '/home/akashmis/xai_credit_risk/results/lime_results.pkl')
    print(f"\nLIME results saved to 'results/lime_results.pkl'")
    
    return comparison

if __name__ == "__main__":
    comparison = main()