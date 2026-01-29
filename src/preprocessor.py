"""
Data Preprocessing and Feature Engineering Pipeline for Credit Risk Prediction
This module handles data preprocessing, feature scaling, and train-test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class CreditRiskPreprocessor:
    """
    A comprehensive preprocessing pipeline for credit risk data.
    Handles numerical scaling, categorical encoding, and train-test splits.
    """
    
    def __init__(self, scaler_type='standard'):
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_columns = None
        self.target_column = None
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        
    def fit(self, data, target_column):
        """
        Fit the preprocessor to the training data
        """
        self.target_column = target_column
        self.feature_columns = [col for col in data.columns if col != target_column]
        
        # Identify numerical and categorical features
        self.numerical_features = data[self.feature_columns].select_dtypes(
            include=[np.number]).columns.tolist()
        self.categorical_features = data[self.feature_columns].select_dtypes(
            include=['object', 'category']).columns.tolist()
        
        print(f"Numerical features: {len(self.numerical_features)}")
        print(f"Categorical features: {len(self.categorical_features)}")
        
        # Create preprocessing steps
        preprocessing_steps = []
        
        # Add numerical scaling if we have numerical features
        if self.numerical_features:
            if self.scaler_type == 'standard':
                scaler = StandardScaler()
            elif self.scaler_type == 'robust':
                scaler = RobustScaler()
            elif self.scaler_type == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            preprocessing_steps.append(('num', scaler, self.numerical_features))
            self.scaler = scaler
        
        # Create column transformer
        if preprocessing_steps:
            self.preprocessor = ColumnTransformer(
                preprocessing_steps,
                remainder='passthrough'  # Keep categorical features as is
            )
            
            # Fit preprocessor
            X = data[self.feature_columns]
            self.preprocessor.fit(X)
            
            print(f"Preprocessor fitted with {self.scaler_type} scaling")
        else:
            print("No preprocessing steps applied")
    
    def transform(self, data):
        """
        Transform data using fitted preprocessor
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")
        
        X = data[self.feature_columns]
        X_transformed = self.preprocessor.transform(X)
        
        # Convert back to DataFrame
        feature_names = self._get_feature_names()
        X_transformed_df = pd.DataFrame(X_transformed, 
                                    columns=feature_names, 
                                    index=data.index)
        
        return X_transformed_df
    
    def fit_transform(self, data, target_column):
        """
        Fit preprocessor and transform data in one step
        """
        self.fit(data, target_column)
        return self.transform(data)
    
    def split_data(self, data, test_size=0.2, random_state=42, stratify=True):
        """
        Split data into train and test sets
        """
        X = data[self.feature_columns]
        y = data[self.target_column]
        
        stratify_param = y if stratify else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        print(f"Data split completed:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Target distribution in train: {y_train.value_counts().to_dict()}")
        print(f"Target distribution in test: {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def split_data_transformed(self, data, test_size=0.2, random_state=42, stratify=True):
        """
        Split data and return transformed features
        """
        # Set target column before splitting
        if self.target_column is None:
            raise ValueError("Target column not set. Call fit() first with target_column parameter.")
        
        X_train, X_test, y_train, y_test = self.split_data(
            data, test_size, random_state, stratify
        )
        
        # Fit on training data
        self.fit(pd.concat([X_train, X_test], ignore_index=True), self.target_column)
        
        # Transform both train and test
        X_train_transformed = self.transform(X_train)
        X_test_transformed = self.transform(X_test)
        
        return X_train_transformed, X_test_transformed, y_train, y_test
    
    def _get_feature_names(self):
        """
        Get feature names after transformation
        """
        if self.preprocessor is None:
            return self.feature_columns
        
        feature_names = []
        
        # Get numerical feature names (scaled)
        if self.numerical_features:
            feature_names.extend(self.numerical_features)
        
        # Get categorical feature names (unchanged)
        if self.categorical_features:
            feature_names.extend(self.categorical_features)
        
        return feature_names
    
    def get_preprocessing_info(self):
        """
        Get information about the preprocessing setup
        """
        return {
            'scaler_type': self.scaler_type,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'total_features': len(self.feature_columns),
            'preprocessor_fitted': self.preprocessor is not None
        }

class FeatureEngineer:
    """
    Feature engineering utilities for credit risk data
    """
    
    @staticmethod
    def create_payment_ratio_features(data):
        """
        Create payment ratio features for credit card data
        """
        engineered_data = data.copy()
        
        # Pay_1 to Pay_6 are payment status columns
        # Bill_AMT1 to Bill_AMT6 are bill amount columns  
        # Pay_AMT1 to Pay_AMT6 are payment amount columns
        
        for i in range(1, 7):
            pay_col = f'PAY_{i}'
            bill_col = f'BILL_AMT{i}'
            pay_amt_col = f'PAY_AMT{i}'
            
            if all(col in data.columns for col in [pay_col, bill_col, pay_amt_col]):
                # Payment to bill ratio
                ratio_col = f'PAY_BILL_RATIO_{i}'
                engineered_data[ratio_col] = np.where(
                    data[bill_col] != 0,
                    data[pay_amt_col] / data[bill_col],
                    0
                )
                
                # Limit extreme values
                engineered_data[ratio_col] = engineered_data[ratio_col].clip(
                    lower=engineered_data[ratio_col].quantile(0.01),
                    upper=engineered_data[ratio_col].quantile(0.99)
                )
        
        print(f"Created {6} payment ratio features")
        return engineered_data
    
    @staticmethod
    def create_lagged_features(data):
        """
        Create lagged features (differences between consecutive months)
        """
        engineered_data = data.copy()
        
        for i in range(1, 6):
            bill_col_current = f'BILL_AMT{i}'
            bill_col_next = f'BILL_AMT{i+1}'
            pay_col_current = f'PAY_AMT{i}'
            pay_col_next = f'PAY_AMT{i+1}'
            
            if all(col in data.columns for col in [bill_col_current, bill_col_next]):
                # Bill amount change
                diff_col = f'BILL_CHANGE_{i}'
                engineered_data[diff_col] = data[bill_col_next] - data[bill_col_current]
            
            if all(col in data.columns for col in [pay_col_current, pay_col_next]):
                # Payment amount change
                diff_col = f'PAY_CHANGE_{i}'
                engineered_data[diff_col] = data[pay_col_next] - data[pay_col_current]
        
        print(f"Created lagged features for bill and payment changes")
        return engineered_data
    
    @staticmethod
    def create_aggregate_features(data):
        """
        Create aggregate features
        """
        engineered_data = data.copy()
        
        # Total bills over 6 months
        bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in data.columns]
        if bill_cols:
            engineered_data['TOTAL_BILL_6M'] = data[bill_cols].sum(axis=1)
            engineered_data['AVG_BILL_6M'] = data[bill_cols].mean(axis=1)
            engineered_data['STD_BILL_6M'] = data[bill_cols].std(axis=1)
        
        # Total payments over 6 months
        pay_cols = [f'PAY_AMT{i}' for i in range(1, 7) if f'PAY_AMT{i}' in data.columns]
        if pay_cols:
            engineered_data['TOTAL_PAY_6M'] = data[pay_cols].sum(axis=1)
            engineered_data['AVG_PAY_6M'] = data[pay_cols].mean(axis=1)
            engineered_data['STD_PAY_6M'] = data[pay_cols].std(axis=1)
        
        print(f"Created aggregate features for bills and payments")
        return engineered_data
    
    @staticmethod
    def create_risk_indicators(data):
        """
        Create risk indicator features
        """
        engineered_data = data.copy()
        
        # Count of delayed payments (PAY_* > 0)
        pay_cols = [f'PAY_{i}' for i in range(1, 7) if f'PAY_{i}' in data.columns]
        if pay_cols:
            engineered_data['DELAYED_PAYMENTS_COUNT'] = (data[pay_cols] > 0).sum(axis=1)
            engineered_data['MAX_DELAY_MONTHS'] = data[pay_cols].max(axis=1)
        
        # Utilization indicators (if LIMIT_BAL is available)
        if 'LIMIT_BAL' in data.columns and bill_cols:
            bill_cols = [f'BILL_AMT{i}' for i in range(1, 7) if f'BILL_AMT{i}' in data.columns]
            for bill_col in bill_cols:
                util_col = bill_col.replace('BILL_AMT', 'UTILIZATION_')
                engineered_data[util_col] = np.where(
                    data['LIMIT_BAL'] > 0,
                    data[bill_col] / data['LIMIT_BAL'],
                    0
                )
            
            # Average utilization
            util_cols = [col for col in engineered_data.columns if 'UTILIZATION_' in col]
            if util_cols:
                engineered_data['AVG_UTILIZATION_6M'] = engineered_data[util_cols].mean(axis=1)
                engineered_data['MAX_UTILIZATION_6M'] = engineered_data[util_cols].max(axis=1)
        
        print(f"Created risk indicator features")
        return engineered_data

def main():
    """
    Demonstrate preprocessing and feature engineering
    """
    import sys
    sys.path.append('/home/akashmis/xai_credit_risk')
    from src.data_loader import CreditRiskDataLoader
    
    # Load data
    loader = CreditRiskDataLoader()
    data = loader.load_uci_credit_card()
    
    print("\n" + "="*50)
    print("PREPROCESSING PIPELINE DEMO")
    print("="*50)
    
    # Feature engineering
    print("Applying feature engineering...")
    fe = FeatureEngineer()
    data_engineered = fe.create_payment_ratio_features(data)
    data_engineered = fe.create_lagged_features(data_engineered)
    data_engineered = fe.create_aggregate_features(data_engineered)
    data_engineered = fe.create_risk_indicators(data_engineered)
    
    print(f"Original features: {data.shape[1]}")
    print(f"Engineered features: {data_engineered.shape[1]}")
    
    # Preprocessing
    preprocessor = CreditRiskPreprocessor(scaler_type='standard')
    
    # First set the target column
    preprocessor.target_column = 'default'
    preprocessor.feature_columns = [col for col in data_engineered.columns if col != 'default']
    
    X_train, X_test, y_train, y_test = preprocessor.split_data_transformed(
        data_engineered, test_size=0.2, random_state=42
    )
    
    print(f"\nPreprocessed training data shape: {X_train.shape}")
    print(f"Preprocessed test data shape: {X_test.shape}")
    
    # Save preprocessed data
    preprocessed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X_train.columns.tolist()
    }
    
    import joblib
    joblib.dump(preprocessed_data, '/home/akashmis/xai_credit_risk/data/preprocessed_data.pkl')
    joblib.dump(preprocessor, '/home/akashmis/xai_credit_risk/data/preprocessor.pkl')
    
    print("\nPreprocessed data saved to 'data/preprocessed_data.pkl'")
    print("Preprocessor saved to 'data/preprocessor.pkl'")
    
    # Display preprocessing info
    info = preprocessor.get_preprocessing_info()
    print(f"\nPreprocessing Info:")
    for key, value in info.items():
        print(f"{key}: {value}")
    
    return preprocessor, data_engineered

if __name__ == "__main__":
    preprocessor, engineered_data = main()