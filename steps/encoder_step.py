import pandas as pd
import logging
from typing import Tuple
from zenml import step
from src.encoder import CategoricalFeatureEncoder

logger = logging.getLogger(__name__)

@step
def encoder_step(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    ZenML step to encode categorical columns using OneHotEncoder.
    """
    encoder = CategoricalFeatureEncoder()
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)

    logger.info(f"✅ Encoding complete. Train shape: {X_train_encoded.shape}, Test shape: {X_test_encoded.shape}")
    return X_train_encoded, X_test_encoded
