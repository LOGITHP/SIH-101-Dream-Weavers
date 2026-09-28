"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Data Preprocessing Pipeline for Official Competency Assessment

Handles:
- Missing-value imputation (median for numerical, mode for categorical)
- Duplicate detection and removal
- Categorical feature encoding (One-Hot Encoding with unknown handle)
- Numerical feature scaling (StandardScaler)
- Train-test stratified splitting
- Saving and loading preprocessing pipeline state
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


# Feature definitions
CATEGORICAL_FEATURES = ["role", "department", "target_role"]

NUMERICAL_FEATURES = [
    "experience_years",
    "statistical_analysis_score",
    "data_visualization_score",
    "python_score",
    "sql_score",
    "data_collection_score",
    "data_quality_score",
    "statistical_methods_score",
    "communication_score",
    "previous_training_count",
    "training_hours_completed",
    "assessment_score",
    "learning_progress"
]

TARGET_COLUMN = "competency_level"
COMPETENCY_CLASSES = ["Beginner", "Developing", "Proficient", "Advanced"]


class DataPreprocessor:
    """
    Robust Preprocessing Pipeline for Competency Assessment.
    Encapsulates imputation, encoding, scaling, and feature alignment.
    """

    def __init__(self):
        self.num_imputer = SimpleImputer(strategy="median")
        self.cat_imputer = SimpleImputer(strategy="most_frequent")
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.is_fitted = False
        self.feature_names = []

    def fit(self, df: pd.DataFrame):
        """
        Fits the imputation, encoding, and scaling components on training data.
        """
        # Ensure working copy without duplicates
        clean_df = df.drop_duplicates().copy()

        # Fit numerical imputer & scaler
        num_data = clean_df[NUMERICAL_FEATURES]
        imputed_num = self.num_imputer.fit_transform(num_data)
        self.scaler.fit(imputed_num)

        # Fit categorical imputer & encoder
        cat_data = clean_df[CATEGORICAL_FEATURES]
        imputed_cat = self.cat_imputer.fit_transform(cat_data)
        self.encoder.fit(imputed_cat)

        # Build feature names
        cat_feature_names = list(self.encoder.get_feature_names_out(CATEGORICAL_FEATURES))
        self.feature_names = NUMERICAL_FEATURES + cat_feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transforms input dataframe into model-ready feature matrix X.
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before calling transform!")

        data = df.copy()

        # Ensure all required columns exist (fill missing with NaN for imputer to handle)
        for col in NUMERICAL_FEATURES:
            if col not in data.columns:
                data[col] = np.nan
        for col in CATEGORICAL_FEATURES:
            if col not in data.columns:
                data[col] = "Unknown"

        # Numerical transform
        num_data = data[NUMERICAL_FEATURES]
        imputed_num = self.num_imputer.transform(num_data)
        scaled_num = self.scaler.transform(imputed_num)

        # Categorical transform
        cat_data = data[CATEGORICAL_FEATURES]
        imputed_cat = self.cat_imputer.transform(cat_data)
        encoded_cat = self.encoder.transform(imputed_cat)

        # Stack into single feature matrix
        X = np.hstack([scaled_num, encoded_cat])
        return X

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Fits and transforms in one step.
        """
        self.fit(df)
        return self.transform(df)

    def save(self, filepath: str):
        """
        Saves preprocessor state to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Preprocessor saved successfully to {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "DataPreprocessor":
        """
        Loads preprocessor state from disk.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at {filepath}")
        with open(filepath, "rb") as f:
            obj = pickle.load(f)
        return obj


def prepare_train_test_data(data_path="data/official_training_dataset.csv", test_size=0.2, random_state=42):
    """
    Loads dataset, handles duplicates, fits preprocessor, and performs
    stratified train-test split.
    """
    print(f"Loading official training dataset from {data_path}...")
    df = pd.read_csv(data_path)

    # 1. Duplicate check & removal
    initial_len = len(df)
    df = df.drop_duplicates(subset=["employee_id"])
    duplicates_removed = initial_len - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate records.")
    else:
        print("No duplicate records found.")

    # 2. Check missing values
    missing_summary = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES].isnull().sum()
    missing_cols = missing_summary[missing_summary > 0]
    if not missing_cols.empty:
        print("Missing values detected:")
        for col, cnt in missing_cols.items():
            print(f"  - {col}: {cnt} missing values (handled via median/mode imputation)")
    else:
        print("No missing values in primary features.")

    # 3. Stratified Train / Test Split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN]
    )
    print(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples (Test ratio: {test_size * 100:.0f}%)")

    # 4. Fit Preprocessor on Train set only (prevent data leakage)
    preprocessor = DataPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    return preprocessor, X_train, X_test, y_train, y_test, train_df, test_df


if __name__ == "__main__":
    preprocessor, X_tr, X_te, y_tr, y_te, tr_df, te_df = prepare_train_test_data()
    preprocessor.save("models/preprocessing.pkl")
    print(f"Feature matrix shape: X_train={X_tr.shape}, X_test={X_te.shape}")
    print(f"Total engineered features: {len(preprocessor.feature_names)}")
