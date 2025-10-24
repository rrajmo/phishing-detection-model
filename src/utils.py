from typing import Tuple
import pandas as pd

def load_data(filename: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    df = pd.read_csv(filename)
    X = df.drop(columns=["label"])
    y = df["label"]
    return df, X, y