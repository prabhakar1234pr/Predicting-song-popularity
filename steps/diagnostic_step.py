# steps/diagnostics_step.py

import pandas as pd
from zenml import step

@step
def diagnostics_step(X: pd.DataFrame) -> None:
    """Step to print missing value counts in a DataFrame."""
    print("=== Missing Values ===")
    print(X.isna().sum())
