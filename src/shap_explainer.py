"""
SHAP (SHapley Additive exPlanations) Implementation for Credit Risk Models
This module provides comprehensive SHAP explanations for all trained models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

class SHAPExplainer:
    """
    Comprehensive SHAP explanation class for credit risk models
    """
    
    def __init__(self, model, X_train, feature_names=None):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names or X_train.columns.tolist()
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, explainer_type='auto'):
        """
        Create appropriate SHAP explainer based on model type
        """
        model_name = type(self.model).__name__.lower()
        
        if 'logistic' in model_name or 'linear' in model_name:
            self.explainer = shap.LinearExplainer(self.model, self.X_train)
        elif 'forest' in model_name or 'tree' in model_name or 'xgb' in model_name:
            self.explainer = shap.TreeExplainer(self.model)
        else:
            # Use KernelExplainer as fallback
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba, 
                shap.sample(self.X_train, 100)
            )
        
        print(f"Created {type(self.explainer).__name__} for {model_name}")
        return self.explainer
    
    def calculate_shap_values(self, X_test=None, sample_size=100):
        """
        Calculate SHAP values for test data
        """
        if self.explainer is None:
            self.create_explainer()
        
        if X_test is None:
            X_test = self.X_train
        
        # Sample data for faster computation
        if len(X_test) > sample_size:
            X_sample = shap.sample(X_test, sample_size)
        else:
            X_sample = X_test
        
        print(f"Calculating SHAP values for {len(X_sample)} samples...")
        self.shap_values = self.explainer.shap_values(X_sample)
        
        # Handle binary classification (return array of arrays)
        if isinstance(self.shap_values, list):
            self.shap_values = self.shap_values[1]  # Use positive class
        
        return self.shap_values, X_sample
    
    def plot_summary(self, save_path=None):
        """
        Create SHAP summary plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values, 
            self.X_train.iloc[:len(self.shap_values)], 
            feature_names=self.feature_names,
            plot_type="dot",
            show=False
        )
        plt.title('SHAP Summary Plot - Feature Impact on Predictions')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP summary plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_importance(self, save_path=None):
        """
        Create SHAP feature importance plot
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        # Calculate mean absolute SHAP values
        feature_importance = np.abs(self.shap_values).mean(0)
        
        # Debug shapes
        print(f"SHAP values shape: {self.shap_values.shape}")
        print(f"Feature importance shape: {feature_importance.shape}")
        print(f"Feature names length: {len(self.feature_names)}")
        
        # Ensure feature_importance is 1D
        if len(feature_importance.shape) > 1:
            feature_importance = feature_importance.flatten()
        
        # Take minimum length to avoid mismatch
        min_length = min(len(feature_importance), len(self.feature_names))
        feature_importance = feature_importance[:min_length]
        feature_names_subset = self.feature_names[:min_length]
        
        print(f"Using {min_length} features for plotting")
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_subset,
            'importance': feature_importance
        }).sort_values('importance', ascending=True)
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Mean |SHAP Value|')
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP feature importance plot saved to {save_path}")
        
        plt.show()
        return feature_importance_df
    
    def explain_individual(self, instance_idx=0, save_path=None):
        """
        Create individual prediction explanation
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        if instance_idx >= len(self.shap_values):
            instance_idx = 0
        
        plt.figure(figsize=(12, 6))
        
        # Handle multi-class SHAP values
        expected_val = self.explainer.expected_value
        if hasattr(expected_val, '__len__'):
            expected_val = expected_val[1]  # Use positive class for binary
        
        instance_shap_values = self.shap_values[instance_idx]
        if len(instance_shap_values.shape) > 1:
            instance_shap_values = instance_shap_values[:, 1]  # Use positive class
        
        shap.force_plot(
            expected_val,
            instance_shap_values,
            self.X_train.iloc[instance_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        plt.title(f'SHAP Explanation for Sample {instance_idx}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Individual SHAP explanation saved to {save_path}")
        
        plt.show()
    
    def plot_waterfall(self, instance_idx=0, save_path=None):
        """
        Create waterfall plot for individual explanation
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call calculate_shap_values() first.")
        
        if instance_idx >= len(self.shap_values):
            instance_idx = 0
        
        plt.figure(figsize=(12, 8))
        
        # Handle multi-class SHAP values
        expected_val = self.explainer.expected_value
        if hasattr(expected_val, '__len__'):
            expected_val = expected_val[1]  # Use positive class for binary
        
        instance_shap_values = self.shap_values[instance_idx]
        if len(instance_shap_values.shape) > 1:
            instance_shap_values = instance_shap_values[:, 1]  # Use positive class
        
        shap.waterfall_plot(
            shap.Explanation(
                values=instance_shap_values,
                base_values=expected_val,
                data=self.X_train.iloc[instance_idx].values,
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot for Sample {instance_idx}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP waterfall plot saved to {save_path}")
        
        plt.show()

class SHAPComparison:
    """
    Compare SHAP explanations across different models
    """
    
    def __init__(self):
        self.explainers = {}
        self.results = {}
    
    def add_model(self, model_name, model, X_train, feature_names=None):
        """
        Add a model for SHAP comparison
        """
        explainer = SHAPExplainer(model, X_train, feature_names)
        self.explainers[model_name] = explainer
        print(f"Added {model_name} to SHAP comparison")
    
    def calculate_all_shap_values(self, X_test=None, sample_size=100):
        """
        Calculate SHAP values for all models
        """
        for model_name, explainer in self.explainers.items():
            print(f"\nCalculating SHAP values for {model_name}...")
            shap_values, X_sample = explainer.calculate_shap_values(X_test, sample_size)
            self.results[model_name] = {
                'shap_values': shap_values,
                'X_sample': X_sample,
                'explainer': explainer
            }
    
    def compare_feature_importance(self, save_path=None):
        """
        Compare feature importance across models
        """
        importance_data = []
        
        for model_name, result in self.results.items():
            feature_importance = np.abs(result['shap_values']).mean(0)
            
            # Handle multi-dimensional importance
            if len(feature_importance.shape) > 1:
                feature_importance = feature_importance[:, 1]  # Use positive class
            
            for i, importance in enumerate(feature_importance):
                importance_data.append({
                    'model': model_name,
                    'feature': result['explainer'].feature_names[i],
                    'importance': float(importance)  # Ensure numeric type
                })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Create comparison plot
        plt.figure(figsize=(15, 10))
        
        # Get top features by average importance
        top_features = importance_df.groupby('feature')['importance'].mean().nlargest(10).index
        filtered_df = importance_df[importance_df['feature'].isin(top_features)]
        
        # Create grouped bar plot
        pivot_df = filtered_df.pivot(index='feature', columns='model', values='importance')
        pivot_df.plot(kind='bar', figsize=(15, 8))
        plt.title('Top 10 Features - SHAP Importance Comparison Across Models')
        plt.xlabel('Features')
        plt.ylabel('Mean |SHAP Value|')
        plt.xticks(rotation=45)
        plt.legend(title='Models')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SHAP feature importance comparison saved to {save_path}")
        
        plt.show()
        return importance_df
    
    def create_summary_report(self, save_path=None):
        """
        Create comprehensive SHAP summary report
        """
        print("="*80)
        print("SHAP EXPLANATIONS COMPARISON REPORT")
        print("="*80)
        
        for model_name, result in self.results.items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)
            
            # Top 5 features
            feature_importance = np.abs(result['shap_values']).mean(0)
            
            # Handle multi-dimensional importance
            if len(feature_importance.shape) > 1:
                feature_importance = feature_importance[:, 1]  # Use positive class
            
            # Ensure 1D
            if len(feature_importance.shape) > 1:
                feature_importance = feature_importance.flatten()
            
            importance_df = pd.DataFrame({
                'feature': result['explainer'].feature_names[:len(feature_importance)],
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            print("Top 5 Most Important Features:")
            for i, row in importance_df.head(5).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save report
        if save_path:
            with open(save_path, 'w') as f:
                f.write("SHAP EXPLANATIONS COMPARISON REPORT\n")
                f.write("="*80 + "\n")
                for model_name, result in self.results.items():
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    
                    feature_importance = np.abs(result['shap_values']).mean(0)
                    
                    # Handle multi-dimensional importance
                    if len(feature_importance.shape) > 1:
                        feature_importance = feature_importance[:, 1]
                    
                    # Ensure 1D
                    if len(feature_importance.shape) > 1:
                        feature_importance = feature_importance.flatten()
                    
                    importance_df = pd.DataFrame({
                        'feature': result['explainer'].feature_names[:len(feature_importance)],
                        'importance': feature_importance
                    }).sort_values('importance', ascending=False)
                    
                    f.write("Top 5 Most Important Features:\n")
                    for i, row in importance_df.head(5).iterrows():
                        f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
            
            print(f"\nSHAP report saved to {save_path}")

def main():
    """
    Demonstrate SHAP explanations for all models
    """
    # Load data and models
    test_data = joblib.load('/home/akashmis/xai_credit_risk/results/test_data.pkl')
    X_test = test_data['X_test']
    y_test = test_data['y_test']
    
    preprocessed_data = joblib.load('/home/akashmis/xai_credit_risk/data/preprocessed_data.pkl')
    X_train = preprocessed_data['X_train']
    feature_names = X_train.columns.tolist()
    
    print("="*80)
    print("SHAP EXPLANATIONS FOR CREDIT RISK MODELS")
    print("="*80)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Create SHAP comparison
    comparison = SHAPComparison()
    
    # Load and add models
    model_names = ['logistic_regression', 'random_forest', 'xgboost']
    for model_name in model_names:
        model = joblib.load(f'/home/akashmis/xai_credit_risk/models/{model_name}.pkl')
        comparison.add_model(model_name, model, X_train, feature_names)
    
    # Calculate SHAP values for all models
    comparison.calculate_all_shap_values(X_test, sample_size=50)  # Smaller sample for speed
    
    # Create visualizations
    print("\nGenerating SHAP visualizations...")
    
    # Individual model explanations
    for model_name, result in comparison.results.items():
        explainer = result['explainer']
        
        print(f"\n{model_name.upper()} SHAP Analysis:")
        
        # Summary plot
        explainer.plot_summary(
            save_path=f'/home/akashmis/xai_credit_risk/figures/shap_summary_{model_name}.png'
        )
        
        # Feature importance
        explainer.plot_feature_importance(
            save_path=f'/home/akashmis/xai_credit_risk/figures/shap_importance_{model_name}.png'
        )
        
        # Individual explanation
        explainer.explain_individual(
            instance_idx=0,
            save_path=f'/home/akashmis/xai_credit_risk/figures/shap_individual_{model_name}.png'
        )
    
    # Comparison across models
    comparison.compare_feature_importance(
        save_path='/home/akashmis/xai_credit_risk/figures/shap_comparison.png'
    )
    
    # Create summary report
    comparison.create_summary_report(
        save_path='/home/akashmis/xai_credit_risk/results/shap_report.txt'
    )
    
    # Save SHAP results
    shap_results = {}
    for model_name, result in comparison.results.items():
        shap_results[model_name] = {
            'shap_values': result['shap_values'],
            'feature_names': result['explainer'].feature_names
        }
    
    joblib.dump(shap_results, '/home/akashmis/xai_credit_risk/results/shap_results.pkl')
    print(f"\nSHAP results saved to 'results/shap_results.pkl'")
    
    return comparison

if __name__ == "__main__":
    comparison = main()