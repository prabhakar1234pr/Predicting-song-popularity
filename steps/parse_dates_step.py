import pandas as pd
import logging
from zenml import step
from src.date_parsing import parse_release_date

logger = logging.getLogger(__name__)

@step
def parse_date_step(df: pd.DataFrame) -> pd.DataFrame:
    """
    ZenML step that parses the track_album_release_date into
    year, month, and day components.
    """
    return parse_release_date(df)
