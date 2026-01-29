"""
Model Training Module for Credit Risk Prediction
This module handles training of Logistic Regression, Random Forest, and XGBoost models.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, roc_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

class ModelTrainer:
    """
    A comprehensive model training class for credit risk prediction
    """
    
    def __init__(self):
        self.models = {}
        self.best_params = {}
        self.training_history = {}
        
    def train_logistic_regression(self, X_train, y_train, cv_folds=5):
        """
        Train Logistic Regression with cross-validation and hyperparameter tuning
        """
        print("Training Logistic Regression...")
        
        # Define parameter grid
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }
        
        # Create model
        lr = LogisticRegression(random_state=42, class_weight='balanced')
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            lr, param_grid, cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.models['logistic_regression'] = grid_search.best_estimator_
        self.best_params['logistic_regression'] = grid_search.best_params_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            grid_search.best_estimator_, X_train, y_train, 
            cv=cv, scoring='roc_auc'
        )
        
        self.training_history['logistic_regression'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return grid_search.best_estimator_
    
    def train_random_forest(self, X_train, y_train, cv_folds=5):
        """
        Train Random Forest with cross-validation and hyperparameter tuning
        """
        print("Training Random Forest...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', None]
        }
        
        # Create model
        rf = RandomForestClassifier(random_state=42)
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf, param_grid, cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.models['random_forest'] = grid_search.best_estimator_
        self.best_params['random_forest'] = grid_search.best_params_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            grid_search.best_estimator_, X_train, y_train, 
            cv=cv, scoring='roc_auc'
        )
        
        self.training_history['random_forest'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return grid_search.best_estimator_
    
    def train_xgboost(self, X_train, y_train, cv_folds=5):
        """
        Train XGBoost with cross-validation and hyperparameter tuning
        """
        print("Training XGBoost...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [1, len(y_train[y_train==0])/len(y_train[y_train==1])]
        }
        
        # Create model
        xgb_model = xgb.XGBClassifier(
            random_state=42, objective='binary:logistic', eval_metric='logloss'
        )
        
        # Grid search with cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            xgb_model, param_grid, cv=cv, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Store best model
        self.models['xgboost'] = grid_search.best_estimator_
        self.best_params['xgboost'] = grid_search.best_params_
        
        # Cross-validation scores
        cv_scores = cross_val_score(
            grid_search.best_estimator_, X_train, y_train, 
            cv=cv, scoring='roc_auc'
        )
        
        self.training_history['xgboost'] = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV AUC: {grid_search.best_score_:.4f}")
        print(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return grid_search.best_estimator_
    
    def train_all_models(self, X_train, y_train, cv_folds=5):
        """
        Train all three models
        """
        print("="*60)
        print("TRAINING ALL MODELS")
        print("="*60)
        
        self.train_logistic_regression(X_train, y_train, cv_folds)
        print("\n" + "-"*40 + "\n")
        
        self.train_random_forest(X_train, y_train, cv_folds)
        print("\n" + "-"*40 + "\n")
        
        self.train_xgboost(X_train, y_train, cv_folds)
        
        return self.models
    
    def get_model(self, model_name):
        """
        Get a trained model by name
        """
        if model_name in self.models:
            return self.models[model_name]
        else:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
    
    def save_models(self, save_dir='/home/akashmis/xai_credit_risk/models'):
        """
        Save all trained models
        """
        for model_name, model in self.models.items():
            model_path = f"{save_dir}/{model_name}.pkl"
            joblib.dump(model, model_path)
            print(f"Model {model_name} saved to {model_path}")
        
        # Save training history
        history_path = f"{save_dir}/training_history.pkl"
        joblib.dump(self.training_history, history_path)
        print(f"Training history saved to {history_path}")

class ModelEvaluator:
    """
    Model evaluation class with comprehensive metrics
    """
    
    @staticmethod
    def evaluate_model(model, X_test, y_test, model_name="Model"):
        """
        Comprehensive model evaluation
        """
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        print(f"\n{model_name} Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:")
        print(cm)
        
        return {
            'metrics': metrics,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    @staticmethod
    def plot_roc_curves(models_dict, X_test, y_test, save_path=None):
        """
        Plot ROC curves for all models
        """
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc_score = roc_auc_score(y_test, y_pred_proba)
            
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    @staticmethod
    def compare_models(models_dict, X_test, y_test, save_path=None):
        """
        Compare all models with comprehensive metrics
        """
        results = {}
        
        for model_name, model in models_dict.items():
            evaluation = ModelEvaluator.evaluate_model(model, X_test, y_test, model_name)
            results[model_name] = evaluation['metrics']
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        
        print(f"\n{'='*50}")
        print("MODEL COMPARISON")
        print(f"{'='*50}")
        print(comparison_df.round(4))
        
        # Plot comparison
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, y=1.02)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [results[model][metric] for model in results.keys()]
            models = list(results.keys())
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Remove the last empty subplot
        axes[-1].remove()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
        
        return comparison_df

def main():
    """
    Demonstrate model training and evaluation
    """
    # Load preprocessed data
    preprocessed_data = joblib.load('/home/akashmis/xai_credit_risk/data/preprocessed_data.pkl')
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    
    print("="*60)
    print("MODEL TRAINING AND EVALUATION")
    print("="*60)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Training target distribution: {y_train.value_counts().to_dict()}")
    print(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    # Train models
    trainer = ModelTrainer()
    models = trainer.train_all_models(X_train, y_train, cv_folds=3)  # Use 3 folds for speed
    
    # Evaluate models
    evaluator = ModelEvaluator()
    results = evaluator.compare_models(
        models, X_test, y_test,
        save_path='/home/akashmis/xai_credit_risk/figures/model_comparison.png'
    )
    
    # Plot ROC curves
    evaluator.plot_roc_curves(
        models, X_test, y_test,
        save_path='/home/akashmis/xai_credit_risk/figures/roc_curves.png'
    )
    
    # Save models
    trainer.save_models()
    
    # Save evaluation results
    joblib.dump(results, '/home/akashmis/xai_credit_risk/results/model_evaluation_results.pkl')
    print(f"\nEvaluation results saved to 'results/model_evaluation_results.pkl'")
    
    return trainer, models, results

if __name__ == "__main__":
    trainer, models, results = main()