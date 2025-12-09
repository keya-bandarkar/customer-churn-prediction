import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def split_features_target(df: pd.DataFrame):
    df = df.copy()

    # Target: map Yes/No to 1/0
    y = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop customerID and target
    X = df.drop(columns=["customerID", "Churn"], errors="ignore")

    return X, y

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ]
    )
    return preprocessor

def prepare_data_for_model(df: pd.DataFrame):
    X, y = split_features_target(df)
    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Fit preprocessor on train only
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    return X_train_prep, X_test_prep, y_train, y_test, preprocessor
