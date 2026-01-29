"""
Quick Model Training for Demonstration
Simplified version without extensive hyperparameter tuning
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

def quick_train_models():
    """
    Quick training of all models for demonstration
    """
    # Load preprocessed data
    preprocessed_data = joblib.load('/home/akashmis/xai_credit_risk/data/preprocessed_data.pkl')
    X_train = preprocessed_data['X_train']
    X_test = preprocessed_data['X_test']
    y_train = preprocessed_data['y_train']
    y_test = preprocessed_data['y_test']
    
    print("="*60)
    print("QUICK MODEL TRAINING")
    print("="*60)
    print(f"Training data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    models = {}
    
    # Logistic Regression
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    print("Logistic Regression trained!")
    
    # Random Forest
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    print("Random Forest trained!")
    
    # XGBoost
    print("\nTraining XGBoost...")
    scale_pos_weight = len(y_train[y_train==0])/len(y_train[y_train==1])
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, random_state=42, scale_pos_weight=scale_pos_weight
    )
    xgb_model.fit(X_train, y_train)
    models['xgboost'] = xgb_model
    print("XGBoost trained!")
    
    # Evaluate all models
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60)
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba)
        }
        
        results[name] = metrics
        print(f"\n{name.upper()}:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    # Save models and results
    import os
    os.makedirs('/home/akashmis/xai_credit_risk/models', exist_ok=True)
    os.makedirs('/home/akashmis/xai_credit_risk/results', exist_ok=True)
    
    for name, model in models.items():
        joblib.dump(model, f'/home/akashmis/xai_credit_risk/models/{name}.pkl')
    
    joblib.dump(results, '/home/akashmis/xai_credit_risk/results/quick_results.pkl')
    joblib.dump({'X_test': X_test, 'y_test': y_test}, '/home/akashmis/xai_credit_risk/results/test_data.pkl')
    
    print(f"\nModels and results saved!")
    return models, results, X_test, y_test

if __name__ == "__main__":
    models, results, X_test, y_test = quick_train_models()