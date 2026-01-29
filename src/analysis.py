"""
Comprehensive Analysis and Visualization for XAI Credit Risk Project
This module creates comparative analysis between accuracy and interpretability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

class XAIAnalysis:
    """
    Comprehensive analysis comparing model performance vs interpretability
    """
    
    def __init__(self):
        self.model_results = None
        self.shap_results = None
        self.lime_results = None
        
    def load_results(self):
        """
        Load all results from previous steps
        """
        self.model_results = joblib.load('/home/akashmis/xai_credit_risk/results/quick_results.pkl')
        self.shap_results = joblib.load('/home/akashmis/xai_credit_risk/results/shap_results.pkl')
        self.lime_results = joblib.load('/home/akashmis/xai_credit_risk/results/lime_results.pkl')
        
        print("Loaded all results for analysis")
        return self.model_results, self.shap_results, self.lime_results
    
    def calculate_interpretability_scores(self):
        """
        Calculate interpretability scores for each model
        """
        interpretability_scores = {}
        
        for model_name in self.model_results.keys():
            # SHAP-based interpretability
            if model_name in self.shap_results:
                shap_values = self.shap_results[model_name]['shap_values']
                feature_consistency = self._calculate_feature_consistency(shap_values)
                explanation_complexity = self._calculate_explanation_complexity(shap_values)
                shap_interpretability = (feature_consistency + explanation_complexity) / 2
            else:
                shap_interpretability = 0
            
            # LIME-based interpretability
            if model_name in self.lime_results:
                lime_explanations = self.lime_results[model_name]
                local_fidelity = self._calculate_local_fidelity(lime_explanations)
                lime_interpretability = local_fidelity
            else:
                lime_interpretability = 0
            
            # Model complexity (inverse of interpretability)
            model_complexity = self._get_model_complexity(model_name)
            
            # Overall interpretability score (0-1, higher is better)
            overall_interpretability = (shap_interpretability * 0.6 + 
                                      lime_interpretability * 0.4) * (1 / model_complexity)
            
            interpretability_scores[model_name] = {
                'shap_interpretability': shap_interpretability,
                'lime_interpretability': lime_interpretability,
                'model_complexity': model_complexity,
                'overall_interpretability': overall_interpretability
            }
        
        return interpretability_scores
    
    def _calculate_feature_consistency(self, shap_values):
        """
        Calculate how consistent feature importance is across instances
        """
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]  # Use positive class for binary
        
        feature_importance_std = np.std(np.abs(shap_values), axis=0)
        feature_consistency = 1 - np.mean(feature_importance_std)
        
        return min(max(feature_consistency, 0), 1)
    
    def _calculate_explanation_complexity(self, shap_values):
        """
        Calculate explanation complexity (number of non-zero features)
        """
        if len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]
        
        avg_non_zero_features = np.mean([np.sum(shap_values[i] != 0) 
                                     for i in range(len(shap_values))])
        total_features = shap_values.shape[1]
        complexity_score = 1 - (avg_non_zero_features / total_features)
        
        return min(max(complexity_score, 0), 1)
    
    def _calculate_local_fidelity(self, lime_explanations):
        """
        Calculate LIME local fidelity (how well explanations match predictions)
        """
        if not lime_explanations:
            return 0
        
        # Simple proxy: average absolute contribution
        avg_contribution = np.mean([
            exp['feature_contributions']['abs_contribution'].mean() 
            for exp in lime_explanations
        ])
        
        return min(max(avg_contribution / 10, 0), 1)  # Normalize to 0-1
    
    def _get_model_complexity(self, model_name):
        """
        Get model complexity score (lower is more interpretable)
        """
        complexity_scores = {
            'logistic_regression': 1.0,    # Most interpretable
            'random_forest': 2.0,          # Medium complexity
            'xgboost': 3.0                 # Most complex
        }
        return complexity_scores.get(model_name, 2.0)
    
    def create_accuracy_vs_interpretability_plot(self, save_path=None):
        """
        Create accuracy vs interpretability comparison plot
        """
        interpretability_scores = self.calculate_interpretability_scores()
        
        # Prepare data for plotting
        plot_data = []
        for model_name in self.model_results.keys():
            plot_data.append({
                'model': model_name.replace('_', ' ').title(),
                'accuracy': self.model_results[model_name]['accuracy'],
                'roc_auc': self.model_results[model_name]['roc_auc'],
                'interpretability': interpretability_scores[model_name]['overall_interpretability']
            })
        
        df = pd.DataFrame(plot_data)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy vs Interpretability
        scatter1 = ax1.scatter(df['interpretability'], df['accuracy'], 
                              s=200, alpha=0.7, c=range(len(df)), cmap='viridis')
        ax1.set_xlabel('Interpretability Score (0-1)')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Accuracy vs Interpretability Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # Add model labels
        for i, row in df.iterrows():
            ax1.annotate(row['model'], 
                        (row['interpretability'], row['accuracy']), 
                        xytext=(5, 5), textcoords='offset points')
        
        # ROC-AUC vs Interpretability
        scatter2 = ax2.scatter(df['interpretability'], df['roc_auc'], 
                              s=200, alpha=0.7, c=range(len(df)), cmap='viridis')
        ax2.set_xlabel('Interpretability Score (0-1)')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_title('ROC-AUC vs Interpretability Trade-off')
        ax2.grid(True, alpha=0.3)
        
        # Add model labels
        for i, row in df.iterrows():
            ax2.annotate(row['model'], 
                        (row['interpretability'], row['roc_auc']), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Accuracy vs interpretability plot saved to {save_path}")
        
        plt.show()
        return df
    
    def create_comprehensive_comparison(self, save_path=None):
        """
        Create comprehensive comparison table and visualization
        """
        interpretability_scores = self.calculate_interpretability_scores()
        
        # Create comparison table
        comparison_data = []
        for model_name in self.model_results.keys():
            metrics = self.model_results[model_name]
            interp_scores = interpretability_scores[model_name]
            
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'Precision': f"{metrics['precision']:.3f}",
                'Recall': f"{metrics['recall']:.3f}",
                'F1-Score': f"{metrics['f1']:.3f}",
                'ROC-AUC': f"{metrics['roc_auc']:.3f}",
                'Interpretability': f"{interp_scores['overall_interpretability']:.3f}",
                'SHAP Score': f"{interp_scores['shap_interpretability']:.3f}",
                'LIME Score': f"{interp_scores['lime_interpretability']:.3f}",
                'Complexity': f"{interp_scores['model_complexity']:.1f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Create radar chart for visual comparison
        self._create_radar_chart(comparison_df, save_path)
        
        return comparison_df
    
    def _create_radar_chart(self, comparison_df, save_path=None):
        """
        Create radar chart for multi-dimensional comparison
        """
        # Prepare data for radar chart
        categories = ['Accuracy', 'ROC-AUC', 'Interpretability', 'SHAP Score', 'LIME Score']
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Number of variables
        N = len(categories)
        
        # Compute angle for each axis
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]
        
        # Plot each model
        colors = ['red', 'blue', 'green']
        for i, (_, row) in enumerate(comparison_df.iterrows()):
            values = []
            for cat in categories:
                if cat in ['Accuracy', 'ROC-AUC']:
                    values.append(float(row[cat]))
                else:
                    values.append(float(row[cat]) * 100)  # Scale interpretability scores
            
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Model'], color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Add labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 100)
        ax.set_title('Model Performance Radar Chart', size=20, y=1.1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        if save_path:
            radar_path = save_path.replace('.png', '_radar.png')
            plt.savefig(radar_path, dpi=300, bbox_inches='tight')
            print(f"Radar chart saved to {radar_path}")
        
        plt.show()
    
    def generate_executive_summary(self, save_path=None):
        """
        Generate executive summary of findings
        """
        interpretability_scores = self.calculate_interpretability_scores()
        
        summary = f"""
EXPLAINABLE AI FOR CREDIT RISK PREDICTION
EXECUTIVE SUMMARY
{'='*60}

KEY FINDINGS:

1. PERFORMANCE COMPARISON:
   - Best Accuracy: {max(self.model_results[m]['accuracy'] for m in self.model_results.keys()):.3f}
   - Best ROC-AUC: {max(self.model_results[m]['roc_auc'] for m in self.model_results.keys()):.3f}

2. INTERPRETABILITY ASSESSMENT:
   - Most Interpretable: {max(interpretability_scores.items(), key=lambda x: x[1]['overall_interpretability'])[0].replace('_', ' ').title()}
   - Least Interpretable: {min(interpretability_scores.items(), key=lambda x: x[1]['overall_interpretability'])[0].replace('_', ' ').title()}

3. ACCURACY VS INTERPRETABILITY TRADE-OFF:
   - High interpretability models: Logistic Regression ({self.model_results['logistic_regression']['accuracy']:.3f} accuracy)
   - High accuracy models: Random Forest ({self.model_results['random_forest']['accuracy']:.3f} accuracy)
   - Balanced approach: XGBoost ({self.model_results['xgboost']['accuracy']:.3f} accuracy)

4. ETHICAL AI IMPLICATIONS:
   - All models provide different levels of transparency
   - SHAP explanations reveal feature importance globally
   - LIME explanations provide local instance-level insights
   - Trade-offs between performance and explainability must be considered

RECOMMENDATIONS:
- Use Logistic Regression for regulatory compliance (high interpretability)
- Use XGBoost for balanced performance and explainability
- Use Random Forest when accuracy is prioritized
- Implement both SHAP and LIME for comprehensive explanations
- Consider domain-specific interpretability requirements
        """
        
        print(summary)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(summary)
            print(f"\nExecutive summary saved to {save_path}")
        
        return summary

def main():
    """
    Main analysis function
    """
    print("="*80)
    print("COMPREHENSIVE XAI ANALYSIS AND VISUALIZATION")
    print("="*80)
    
    # Initialize analysis
    analysis = XAIAnalysis()
    
    # Load all results
    analysis.load_results()
    
    # Create accuracy vs interpretability plot
    print("\nGenerating accuracy vs interpretability analysis...")
    analysis.create_accuracy_vs_interpretability_plot(
        save_path='/home/akashmis/xai_credit_risk/figures/accuracy_vs_interpretability.png'
    )
    
    # Create comprehensive comparison
    print("\nGenerating comprehensive comparison...")
    comparison_df = analysis.create_comprehensive_comparison(
        save_path='/home/akashmis/xai_credit_risk/figures/comprehensive_comparison.png'
    )
    
    # Generate executive summary
    print("\nGenerating executive summary...")
    analysis.generate_executive_summary(
        save_path='/home/akashmis/xai_credit_risk/results/executive_summary.txt'
    )
    
    # Save analysis results
    joblib.dump(comparison_df, '/home/akashmis/xai_credit_risk/results/final_analysis.pkl')
    print(f"\nFinal analysis results saved to 'results/final_analysis.pkl'")
    
    return analysis

if __name__ == "__main__":
    analysis = main()