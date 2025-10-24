from paths import get_path
from typing import List, Tuple, Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif, SelectPercentile
from utils import load_data
import matplotlib.pyplot as plt
import numpy as np
import logging

logger = logging.getLogger(__name__)

def get_config(config: dict) -> Tuple[Any, Any, Any]:
    paths = get_path(config)

    RAW_DATA_PATH = paths["raw_data"]
    PROCESSED_DATA_PATH = paths["processed_data"]
    MUTUAL_INFORMATION_SCORE_PLOT = paths["mutual_information_score_plot"]

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not present at {RAW_DATA_PATH}")
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    MUTUAL_INFORMATION_SCORE_PLOT.parent.mkdir(parents=True, exist_ok=True)

    return RAW_DATA_PATH, PROCESSED_DATA_PATH, MUTUAL_INFORMATION_SCORE_PLOT

class ColumnCleaner(BaseEstimator, TransformerMixin):
    def __init__(self, drop_columns: List[str]):
        self.drop_columns = drop_columns
        self.features = None
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ColumnCleaner":
        logger.info("ColumnCleaner fit started")
        logger.info("ColumnCleaner fit ended")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("ColumnCleaner transform started")
        drop_columns = [col for col in self.drop_columns if col in X.columns]
        self.features = [col for col in X.columns if col not in self.drop_columns]
        logger.info("ColumnCleaner transform ended")
        return X.drop(columns=drop_columns)

class MutualInfoClassification(BaseEstimator, TransformerMixin):
    def __init__(self, percentile: int = 85):
        self.selector = SelectPercentile(score_func=mutual_info_classif, percentile=percentile)
        self.features = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MutualInfoClassification":
        logger.info("MutualInfoClassification fit started")
        self.selector.fit(X, y)
        logger.info("MutualInfoClassification fit ended")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        logger.info("MutualInfoClassification transform started")
        self.features = X.columns.tolist()
        X_transform = self.selector.transform(X)
        logger.info("MutualInfoClassification transform ended")
        return X_transform
    
    def get_feature_names_out(self) -> List[str]:
        mask = self.selector.get_support()
        return [feature for feature, boolean in zip(self.features, mask) if boolean]
    
    def get_scores(self) -> np.ndarray:
        mask = self.selector.get_support()
        scores = self.selector.scores_
        return scores[mask]

def feature_pipeline() -> Pipeline:
    drop_columns = ["FILENAME", "URL", "Domain", "Title", "TLD"]

    pipeline = Pipeline(
        steps=[
            ("ColumnCleaner", ColumnCleaner(drop_columns=drop_columns)),
            ("FeatureSelection", MutualInfoClassification())
        ]
    )
    return pipeline

def fit_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    X_processed = pipeline.fit_transform(X, y)
    return X_processed

def extract_selected_features(pipeline: Pipeline) -> List[str]:
    selected_features = pipeline.named_steps["FeatureSelection"].get_feature_names_out()
    return selected_features

def create_processed_csv(file: str, X: pd.DataFrame, X_processed: np.ndarray, y: pd.Series, selected_features: List[str]) -> None:
    final_df = pd.DataFrame(X_processed, columns=selected_features)
    final_df = final_df.reset_index(drop=True)
    urls = X["URL"].reset_index(drop=True)
    labels = y.reset_index(drop=True)
    final_df["URL"] = urls
    final_df["label"] = labels
    final_df.to_csv(file, index=False)
    logger.info(f"Loaded processed data into {file}")

def create_mutual_information_score_plot(file: str, pipeline: Pipeline, selected_features: List[str]) -> None:
    feature_scores = pipeline.named_steps["FeatureSelection"].get_scores()
    sort_index = np.argsort(feature_scores)[::-1]
    sort_features = np.array(selected_features)[sort_index]
    sort_scores = feature_scores[sort_index]

    plt.figure(figsize=(25, 25))
    plt.barh(sort_features, sort_scores, color='blue')
    plt.xlabel("Mutual Information Scores")
    plt.ylabel("Features")
    plt.title("Top Mutual Information Scores")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(file, bbox_inches='tight')
    logger.info(f"Loaded mutual information score plot into {file}")

def process_dataset(config: dict) -> None:
    RAW_DATA_PATH, PROCESSED_DATA_PATH, MUTUAL_INFORMATION_SCORE_PLOT = get_config(config)

    logger.info(f"Processing data from {RAW_DATA_PATH}")
    _, X, y = load_data(RAW_DATA_PATH)

    pipeline = feature_pipeline()
    X_processed = fit_pipeline(pipeline, X, y)
    logger.info(f"Finished processing data from {RAW_DATA_PATH}")

    selected_features = extract_selected_features(pipeline)

    create_processed_csv(PROCESSED_DATA_PATH, X, X_processed, y, selected_features)
    create_mutual_information_score_plot(MUTUAL_INFORMATION_SCORE_PLOT, pipeline, selected_features)