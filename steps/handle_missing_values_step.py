from zenml import step
import pandas as pd
from src.handle_missing_values import (
    DropMissingHandler, MedianFillHandler, FillConstantHandler, 
    MeanFillHandler, ModeFillHandler, MissingHandlerChooser
)
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@step
def handle_missing_step(df: pd.DataFrame, strategy_name: str = "median") -> pd.DataFrame:
    """
    ZenML step to handle missing values using the specified strategy.
    strategy_name: "drop", "median", "mean", "mode", or "constant"
    """
    logger.info(f" Handling missing values using strategy: {strategy_name}")

    if strategy_name == "drop":
        strategy = DropMissingHandler()

    elif strategy_name in ["median", "mean", "mode"]:
        strategy_map = {
            "median": MedianFillHandler(),
            "mean": MeanFillHandler(),
            "mode": ModeFillHandler()
        }
        strategy = strategy_map[strategy_name]

    elif strategy_name == "constant":
        strategy = FillConstantHandler(constant=0)

    else:
        raise ValueError(f" Unknown strategy '{strategy_name}'. Choose from: drop, median, mean, mode, constant.")

    handler = MissingHandlerChooser(strategy)
    df_cleaned = handler.apply(df)

    logger.info(" Missing value handling complete.")
    return df_cleaned
