import pandas as pd
from .config import DATA_RAW

def load_raw_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_RAW)
    return df
