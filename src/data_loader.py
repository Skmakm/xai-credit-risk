import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')

class CreditRiskDataLoader:
    
    def __init__(self):
        self.data = None
        self.target_column = None
        self.feature_columns = None
        
    def load_uci_credit_card(self):
        print("Loading UCI Credit Card Default dataset...")
        
        X, y = fetch_openml('default-of-credit-card-clients', version=1, as_frame=True, return_X_y=True)
        
        self.data = X.copy()
        self.data['default'] = y.astype(int)
        self.target_column = 'default'
        self.feature_columns = [col for col in X.columns if col != self.target_column]
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Target distribution: {self.data[self.target_column].value_counts().to_dict()}")
        
        return self.data
    
    def load_german_credit(self):
        print("Loading German Credit dataset...")
        
        X, y = fetch_openml('german-credit', version=1, as_frame=True, return_X_y=True)
        
        self.data = X.copy()
        self.data['credit_risk'] = (y == '2').astype(int)
        self.target_column = 'credit_risk'
        self.feature_columns = [col for col in X.columns if col != self.target_column]
        
        print(f"Dataset loaded successfully!")
        print(f"Shape: {self.data.shape}")
        print(f"Target distribution: {self.data[self.target_column].value_counts().to_dict()}")
        
        return self.data
    
    def explore_data(self, save_plots=True):
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
        
        print("\n" + "="*50)
        print("DATA EXPLORATION REPORT")
        print("="*50)
        
        print(f"\nDataset Shape: {self.data.shape}")
        print(f"Number of Features: {len(self.feature_columns)}")
        print(f"Target Column: {self.target_column}")
        
        print(f"\nData Types:")
        print(self.data.dtypes.value_counts())
        
        missing_values = self.data.isnull().sum()
        print(f"\nMissing Values:")
        if missing_values.sum() > 0:
            print(missing_values[missing_values > 0])
        else:
            print("No missing values found!")
        
        print(f"\nTarget Distribution:")
        target_dist = self.data[self.target_column].value_counts(normalize=True)
        for idx, val in target_dist.items():
            print(f"Class {idx}: {val:.2%} ({self.data[self.target_column].value_counts()[idx]} samples)")
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)
        
        print(f"\nNumerical Features Summary:")
        print(self.data[numerical_cols].describe())
        
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        if categorical_cols:
            print(f"\nCategorical Features:")
            for col in categorical_cols:
                print(f"{col}: {self.data[col].nunique()} unique values")
                print(f"  Values: {self.data[col].unique()[:10]}...")
        
        if save_plots:
            self._generate_exploration_plots(numerical_cols, categorical_cols)
        
        return {
            'shape': self.data.shape,
            'numerical_features': numerical_cols,
            'categorical_features': categorical_cols,
            'target_distribution': target_dist.to_dict(),
            'missing_values': missing_values.to_dict()
        }
    
    def _generate_exploration_plots(self, numerical_cols, categorical_cols):
        print("\nGenerating exploration plots...")
        
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Credit Risk Data Exploration', fontsize=16, y=1.02)
        
        ax1 = axes[0, 0]
        target_counts = self.data[self.target_column].value_counts()
        ax1.pie(target_counts.values, labels=[f'Class {i} ({v})' for i, v in enumerate(target_counts.values)], 
                autopct='%1.1f%%', startangle=90)
        ax1.set_title('Target Distribution')
        
        ax2 = axes[0, 1]
        if len(numerical_cols) > 1:
            corr_matrix = self.data[numerical_cols].corr()
            sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, ax=ax2)
            ax2.set_title('Feature Correlation Matrix')
        else:
            ax2.text(0.5, 0.5, 'Not enough numerical features\nfor correlation matrix', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Feature Correlation Matrix')
        
        ax3 = axes[1, 0]
        if len(numerical_cols) > 0:
            top_features = numerical_cols[:3]
            for feature in top_features:
                if feature in self.data.columns:
                    sns.kdeplot(data=self.data, x=feature, hue=self.target_column, 
                               ax=ax3, label=feature, alpha=0.7)
            ax3.set_title('Feature Distributions by Target')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'No numerical features found', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Feature Distributions')
        
        ax4 = axes[1, 1]
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            ax4.bar(range(len(missing_data)), missing_data.values)
            ax4.set_xticks(range(len(missing_data)))
            ax4.set_xticklabels(missing_data.index, rotation=45)
            ax4.set_title('Missing Values by Feature')
            ax4.set_ylabel('Count of Missing Values')
        else:
            ax4.text(0.5, 0.5, 'No missing values found', 
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Missing Values')
        
        plt.tight_layout()
        plt.savefig('/home/akashmis/xai_credit_risk/figures/data_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Exploration plots saved to 'figures/data_exploration.png'")
    
    def get_data_summary(self):
        if self.data is None:
            return "No data loaded"
        
        return {
            'dataset_shape': self.data.shape,
            'target_column': self.target_column,
            'n_features': len(self.feature_columns),
            'target_distribution': self.data[self.target_column].value_counts().to_dict(),
            'missing_values': self.data.isnull().sum().sum(),
            'data_types': self.data.dtypes.value_counts().to_dict()
        }

def main():
    loader = CreditRiskDataLoader()
    
    try:
        data = loader.load_uci_credit_card()
        exploration_results = loader.explore_data()
        
        print("\n" + "="*50)
        print("DATA SUMMARY")
        print("="*50)
        summary = loader.get_data_summary()
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error loading UCI Credit Card dataset: {e}")
        print("Trying German Credit dataset...")
        
        try:
            data = loader.load_german_credit()
            exploration_results = loader.explore_data()
            
            print("\n" + "="*50)
            print("DATA SUMMARY")
            print("="*50)
            summary = loader.get_data_summary()
            for key, value in summary.items():
                print(f"{key}: {value}")
                
        except Exception as e2:
            print(f"Error loading German Credit dataset: {e2}")
            return None
    
    return loader, data

if __name__ == "__main__":
    loader, data = main()