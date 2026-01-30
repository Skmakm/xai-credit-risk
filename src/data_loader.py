import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------
# ✅ Project paths (cross-platform safe)
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = BASE_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


class CreditRiskDataLoader:

    def __init__(self):
        self.data = None
        self.target_column = None
        self.feature_columns = None

    # --------------------------------------------------
    # DATASETS
    # --------------------------------------------------

    def load_uci_credit_card(self):
        print("Loading UCI Credit Card Default dataset...")

        X, y = fetch_openml(
            "default-of-credit-card-clients",
            version=1,
            as_frame=True,
            return_X_y=True
        )

        self.data = X.copy()
        self.data["default"] = y.astype(int)

        self.target_column = "default"
        self.feature_columns = [c for c in X.columns if c != "default"]

        print("Dataset loaded successfully!")
        print("Shape:", self.data.shape)
        print("Target distribution:",
              self.data[self.target_column].value_counts().to_dict())

        return self.data

    def load_german_credit(self):
        print("Loading German Credit dataset...")

        X, y = fetch_openml(
            "german-credit",
            version=1,
            as_frame=True,
            return_X_y=True
        )

        self.data = X.copy()
        self.data["credit_risk"] = (y == "2").astype(int)

        self.target_column = "credit_risk"
        self.feature_columns = [c for c in X.columns if c != "credit_risk"]

        print("Dataset loaded successfully!")
        print("Shape:", self.data.shape)

        return self.data

    # --------------------------------------------------
    # EXPLORATION
    # --------------------------------------------------

    def explore_data(self, save_plots=True):

        if self.data is None:
            raise ValueError("No data loaded.")

        print("\n" + "="*50)
        print("DATA EXPLORATION REPORT")
        print("="*50)

        print("Shape:", self.data.shape)
        print("Target:", self.target_column)

        numerical_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        if self.target_column in numerical_cols:
            numerical_cols.remove(self.target_column)

        categorical_cols = self.data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

        if save_plots:
            self._generate_exploration_plots(numerical_cols, categorical_cols)

        return {
            "shape": self.data.shape,
            "numerical": numerical_cols,
            "categorical": categorical_cols
        }

    # --------------------------------------------------
    # PLOTS
    # --------------------------------------------------

    def _generate_exploration_plots(self, numerical_cols, categorical_cols):

        print("Generating exploration plots...")

        plt.style.use("default")

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle("Credit Risk Data Exploration")

        # Target pie
        counts = self.data[self.target_column].value_counts()
        axes[0,0].pie(counts.values, labels=counts.index, autopct="%1.1f%%")
        axes[0,0].set_title("Target Distribution")

        # Correlation
        if numerical_cols:
            sns.heatmap(
                self.data[numerical_cols].corr(),
                cmap="coolwarm",
                ax=axes[0,1]
            )
            axes[0,1].set_title("Correlation Matrix")

        # KDE
        if numerical_cols:
            f = numerical_cols[0]
            sns.kdeplot(
                data=self.data,
                x=f,
                hue=self.target_column,
                ax=axes[1,0]
            )
            axes[1,0].set_title("Feature Distribution")

        # Missing
        missing = self.data.isnull().sum()
        axes[1,1].bar(range(len(missing)), missing.values)
        axes[1,1].set_title("Missing Values")

        plt.tight_layout()

        out = FIGURES_DIR / "data_exploration.png"
        plt.savefig(out, dpi=300, bbox_inches="tight")
        print("Saved:", out)

        plt.show()

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    def get_data_summary(self):

        if self.data is None:
            return {}

        return {
            "shape": self.data.shape,
            "target": self.target_column,
            "features": len(self.feature_columns)
        }


def main():
    loader = CreditRiskDataLoader()
    data = loader.load_uci_credit_card()
    loader.explore_data()
    return loader, data


if __name__ == "__main__":
    main()
