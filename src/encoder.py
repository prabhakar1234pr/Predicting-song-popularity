import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class BaseEncoder(ABC):
    @abstractmethod
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        pass

class CategoricalFeatureEncoder(BaseEncoder):
    def __init__(self):
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.columns_to_encode = ["playlist_genre", "playlist_subgenre"]
        self.fitted = False

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("🎯 Fitting and transforming categorical columns using OneHotEncoder.")
        encoded = self.encoder.fit_transform(X[self.columns_to_encode])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.columns_to_encode), index=X.index)
        X = X.drop(columns=self.columns_to_encode)
        X_encoded = pd.concat([X, encoded_df], axis=1)
        self.fitted = True
        return X_encoded

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Encoder has not been fitted. Call fit_transform on training data first.")
        logger.info("🚀 Transforming categorical columns in test set using fitted OneHotEncoder.")
        encoded = self.encoder.transform(X[self.columns_to_encode])
        encoded_df = pd.DataFrame(encoded, columns=self.encoder.get_feature_names_out(self.columns_to_encode), index=X.index)
        X = X.drop(columns=self.columns_to_encode)
        X_encoded = pd.concat([X, encoded_df], axis=1)
        return X_encoded
