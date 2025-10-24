from paths import get_path
from typing import Tuple, Any
from sklearn.model_selection import StratifiedKFold, train_test_split
from models.lightgbm_model import LightGBMClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from utils import load_data
import logging

logger = logging.getLogger(__name__)

def get_config(config: dict) -> Tuple[Any, Any, Any, Any, float, int, int, dict]:
    paths = get_path(config)

    PROCESSED_DATA_PATH = paths["processed_data"]
    TEST_DATA = paths["test_data"]
    MODEL_PATH = paths["model"]
    CV_METRICS = paths["cv_metrics"]
    test_size = paths["test_size"]
    cv_splits = paths["cv_splits"]
    random_state = paths["random_state"]
    parameters = paths["model_parameters"]

    TEST_DATA.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    CV_METRICS.parent.mkdir(parents=True, exist_ok=True)
    return PROCESSED_DATA_PATH, TEST_DATA, MODEL_PATH, CV_METRICS, test_size, cv_splits, random_state, parameters

def create_train_test_split(X_features: Any, y: Any, test_size: float, random_state: int) -> Tuple[Any, Any, Any, Any]:
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def save_test_split(file: str, X_test: Any, y_test: Any) -> None:
    df = pd.DataFrame(X_test)
    df["label"] = y_test
    df.to_csv(file, index=False)

def cross_validate_model(file: str, parameters: dict, X: Any, y: Any, n_splits: int, random_state: int) -> Any:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores = []
    X_features = X.drop(columns=["URL"])

    for fold, (train_index, val_index) in enumerate(skf.split(X_features, y)):
        logger.info(f"Training fold {fold + 1} started")
        X_train, X_val = X_features.iloc[train_index], X_features.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = LightGBMClassifier(parameters)
        model.fit(X_train, y_train)
        logger.info(f"Training fold {fold + 1} ended")
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        fold_scores.append(accuracy)
        
    average_score = sum(fold_scores) / n_splits
    with open(file, 'w') as f:
        f.write(f"Cross Validation Metrics\n\n")
        for i, accuracy in enumerate(fold_scores):
            f.write(f"Fold {i + 1} Accuracy: {accuracy:.5f}\n")
        f.write(f"Average accuracy over {n_splits} folds: {average_score:.4f}\n")

    logger.info(f"Loaded cross validation metrics into {file}")

    final_model = LightGBMClassifier(parameters)
    final_model.fit(X_features, y)
    return final_model

def save_model(file: str, model: LightGBMClassifier) -> None:
    logger.info(f"Saved model to {file}")
    model.save(file)

def train_model(config: dict) -> None:
    PROCESSED_DATA_PATH, TEST_DATA, MODEL_PATH, CV_METRICS, test_size, cv_splits, random_state, parameters = get_config(config)

    _, X, y = load_data(PROCESSED_DATA_PATH)

    X_train, X_test, y_train, y_test = create_train_test_split(X, y, test_size, random_state)
    save_test_split(TEST_DATA, X_test, y_test)

    final_model = cross_validate_model(CV_METRICS, parameters, X_train, y_train, cv_splits, random_state)

    save_model(MODEL_PATH, final_model)