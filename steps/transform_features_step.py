import pandas as pd
from zenml import step
from src.transform_features import LogTransformer

@step
def transform_features_step(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    ZenML step to apply feature transformation based on skewness.
    Applies log1p to skewed numerical features (excluding 'key' and 'mode').
    """
    transformer = LogTransformer()
    transformer.fit(X_train)

    X_train_transformed = transformer.transform(X_train)
    X_test_transformed = transformer.transform(X_test)

    return X_train_transformed, X_test_transformed
