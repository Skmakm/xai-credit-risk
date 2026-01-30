"""
Data Preprocessing and Feature Engineering Pipeline — Cross-platform fixed
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# ✅ Cross-platform paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ==================================================
# PREPROCESSOR
# ==================================================

class CreditRiskPreprocessor:

    def __init__(self, scaler_type="standard"):
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.feature_columns = None
        self.target_column = None
        self.numerical_features = None
        self.categorical_features = None

    def _get_scaler(self):
        return {
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "minmax": MinMaxScaler()
        }.get(self.scaler_type, StandardScaler())

    # --------------------------------------------------

    def fit(self, X):

        self.numerical_features = X.select_dtypes(
            include=[np.number]).columns.tolist()

        self.categorical_features = X.select_dtypes(
            include=["object", "category"]).columns.tolist()

        transformer = ColumnTransformer([
            ("num", self._get_scaler(), self.numerical_features)
        ], remainder="passthrough")

        self.preprocessor = transformer.fit(X)

    # --------------------------------------------------

    def transform(self, X):

        Xt = self.preprocessor.transform(X)
        cols = self.numerical_features + self.categorical_features
        return pd.DataFrame(Xt, columns=cols, index=X.index)

    # --------------------------------------------------

    def split_data_transformed(self, data, test_size=0.2, random_state=42):

        if self.target_column is None:
            raise ValueError("target_column must be set")

        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            stratify=y,
            random_state=random_state
        )

        print("Data split completed:")
        print("Training:", len(X_train))
        print("Test:", len(X_test))

        # ✅ Fit ONLY on train (no leakage)
        self.fit(X_train)

        return (
            self.transform(X_train),
            self.transform(X_test),
            y_train,
            y_test
        )


# ==================================================
# FEATURE ENGINEERING
# ==================================================

class FeatureEngineer:

    @staticmethod
    def create_aggregate_features(data):

        df = data.copy()

        bill_cols = [c for c in df.columns if "BILL_AMT" in c]
        pay_cols  = [c for c in df.columns if "PAY_AMT" in c]

        if bill_cols:
            df["TOTAL_BILL_6M"] = df[bill_cols].sum(axis=1)

        if pay_cols:
            df["TOTAL_PAY_6M"] = df[pay_cols].sum(axis=1)

        return df


# ==================================================
# DEMO MAIN
# ==================================================

def main():

    from src.data_loader import CreditRiskDataLoader

    loader = CreditRiskDataLoader()
    data = loader.load_uci_credit_card()

    pre = CreditRiskPreprocessor("standard")
    pre.target_column = "default"

    X_train, X_test, y_train, y_test = pre.split_data_transformed(data)

    bundle = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    joblib.dump(bundle, DATA_DIR / "preprocessed_data.pkl")
    joblib.dump(pre, DATA_DIR / "preprocessor.pkl")

    print("Saved preprocessed data + preprocessor")

    return pre


if __name__ == "__main__":
    main()
