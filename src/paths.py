from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]

def get_path(config):
   data = config["data"]
   train_test_split = config["train_test_split"]
   cross_validation = config["cross_validation"]
   model_parameters = config["model_parameters"]
   model = config["model"]
   model_metrics = config["model_metrics"]
   visualizations = config["visualizations"]
   return {
      "raw_data": BASE_DIR / data["raw_data"],
      "processed_data": BASE_DIR / data["processed_data"],
      "test_data": BASE_DIR / data["test_data"],
      "test_size": train_test_split["test_size"],
      "cv_splits": cross_validation["splits"],
      "cv_metrics": BASE_DIR / cross_validation["metrics"],
      "random_state": train_test_split["random_state"],
      "model_parameters": {
        "objective": model_parameters["objective"],
        "metric": model_parameters["metric"],
        "random_state": model_parameters["random_state"],
        "verbosity": model_parameters["verbosity"],
      },
      "model": BASE_DIR / model,
      "model_metrics": BASE_DIR / model_metrics,
      "mutual_information_score_plot": BASE_DIR / visualizations["mutual_information_score_plot"],
      "confusion_matrix": BASE_DIR / visualizations["confusion_matrix"],
      "roc_curve": BASE_DIR / visualizations["roc_curve"],
      "precision_recall_curve": BASE_DIR / visualizations["precision_recall_curve"],
      "shap_summary": BASE_DIR / visualizations["shap_summary"],
      "last_tree": BASE_DIR / visualizations["last_tree"]
   }
