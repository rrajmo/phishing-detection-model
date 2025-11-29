from paths import get_path
from typing import Tuple, Any
from models.lightgbm_model import LightGBMClassifier
from utils import load_data
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)
import matplotlib.pyplot as plt
import shap
import lightgbm as lgb

logger = logging.getLogger(__name__)

def get_config(config: dict) -> Tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    paths = get_path(config)

    TEST_DATA = paths["test_data"]
    MODEL_PATH = paths["model"]
    MODEL_METRICS = paths["model_metrics"]
    CONFUSION_MATRIX = paths["confusion_matrix"]
    ROC_CURVE = paths["roc_curve"]
    PRECISION_RECALL_CURVE = paths["precision_recall_curve"]
    SHAP_SUMMARY = paths["shap_summary"]
    LAST_TREE = paths["last_tree"]

    return TEST_DATA, MODEL_PATH, MODEL_METRICS, CONFUSION_MATRIX, ROC_CURVE, PRECISION_RECALL_CURVE, SHAP_SUMMARY, LAST_TREE

def load_model(model_file: str) -> Any:
    model = LightGBMClassifier()
    model.model = LightGBMClassifier.load(model_file)
    return model

def run_inference(model_file: str, test_data_file: str) -> Tuple[Any, Any, Any, Any]:
    model = load_model(model_file)
    X_test, y_test = load_data(test_data_file)
    logger.info("Predicting labels for test data started")
    y_predictions = model.predict(X_test.drop(columns=["URL"]))
    y_predictions_probabilities = model.predict_probabilities(X_test.drop(columns=["URL"]))
    logger.info("Predicting labels for test data ended")
    return X_test, y_test, y_predictions, y_predictions_probabilities

def model_metrics_file(file: str, y_test: Any, y_predictions: Any) -> None:
    accuracy = accuracy_score(y_test, y_predictions)
    precision = precision_score(y_test, y_predictions, zero_division=0)
    recall = recall_score(y_test, y_predictions, zero_division=0)
    f1 = f1_score(y_test, y_predictions, zero_division=0)
    
    with open(file, 'w') as f:
        f.write(f"Model Metrics\n\n")
        f.write(f"Model Accuracy: {accuracy:.5f}\n")
        f.write(f"Model Precision: {precision:.5f}\n")
        f.write(f"Model Recall: {recall:.5f}\n")
        f.write(f"Model F1 Score: {f1:.5f}\n")

    logger.info(f"Loaded model metrics into {file}")

def confusion_matrix_chart(file: str, y_test: Any, y_predictions: Any) -> None:
    cm = confusion_matrix(y_test, y_predictions)
    display = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    display.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title("Confusion Matrix")
    plt.savefig(file, bbox_inches='tight')
    plt.close()
    logger.info(f"Loaded confusion matrix into {file}")

def roc_curve_chart(file: str, y_test: Any, y_predictions_probabilities: Any) -> None:
    RocCurveDisplay.from_predictions(y_test, y_predictions_probabilities[:, 1])
    plt.title("Reciever Operating Characteristic Curve")
    plt.savefig(file, bbox_inches='tight')
    plt.close()
    logger.info(f"Loaded reciever operating characteristic curve into {file}")

def precision_recall_curve_chart(file: str, y_test: Any, y_predictions_probabilities: Any) -> None:
    PrecisionRecallDisplay.from_predictions(y_test, y_predictions_probabilities[:, 1])
    plt.title("Precision Recall Curve")
    plt.savefig(file, bbox_inches='tight')
    plt.close()
    logger.info(f"Loaded precision recall curve into {file}")

def shap_summary_chart(file: str, model_file: str, X_test: Any) -> None:
    model = load_model(model_file)
    explainer = shap.Explainer(model.model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(file, bbox_inches='tight')
    plt.close()
    logger.info(f"Loaded SHAP summary into {file}")

def last_tree_chart(file: str, model_file: str) -> None:
    model = load_model(model_file)
    booster = model.model.booster_
    number_of_trees = booster.num_trees()
    lgb.plot_tree(
        booster,
        tree_index=number_of_trees - 1,
        figsize=(25, 25),
        show_info=["split_gain", "internal_value", "internal_count", "leaf_count", "threshold", "decision_type"]
    )
    plt.savefig(file, bbox_inches='tight')
    plt.close()
    logger.info(f"Loaded last tree into {file}")

def evaluate_model(config: dict) -> None:
    TEST_DATA, MODEL_PATH, MODEL_METRICS, CONFUSION_MATRIX, ROC_CURVE, PRECISION_RECALL_CURVE, SHAP_SUMMARY, LAST_TREE = get_config(config)

    X_test, y_test, y_predictions, y_predictions_probabilities = run_inference(MODEL_PATH, TEST_DATA)
    
    model_metrics_file(MODEL_METRICS, y_test, y_predictions)
    confusion_matrix_chart(CONFUSION_MATRIX, y_test, y_predictions)
    roc_curve_chart(ROC_CURVE, y_test, y_predictions_probabilities)
    precision_recall_curve_chart(PRECISION_RECALL_CURVE, y_test, y_predictions_probabilities)
    shap_summary_chart(SHAP_SUMMARY, MODEL_PATH, X_test.drop(columns=["URL"]))
    last_tree_chart(LAST_TREE, MODEL_PATH)