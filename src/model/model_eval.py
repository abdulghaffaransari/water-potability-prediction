
import pandas as pd
import pickle
import yaml
import json
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from dvclive import Live
import dagshub
dagshub.init(repo_owner='abdulghaffaransari', repo_name='water-potability-prediction', mlflow=True)


def load_params(filepath: str) -> dict:
    """Load model parameters from a YAML file."""
    try:
        with open(filepath, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise Exception(f"Error loading parameters from {filepath}: {e}")

def load_model(model_path: str):
    """Load a trained model from a specified path."""
    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        raise Exception(f"Error loading model from {model_path}: {e}")

def load_data(data_path: str) -> pd.DataFrame:
    """Load training data from a CSV file."""
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")

def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target variable from the dataset."""
    try:
        X = data.drop(columns=['Potability'], axis=1)
        y = data['Potability']
        return X, y
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")

def evaluate_model(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate the model and return performance metrics."""
    predictions = model.predict(X)
    return {
        "accuracy": accuracy_score(y, predictions),
        "precision": precision_score(y, predictions),
        "recall": recall_score(y, predictions),
        "f1_score": f1_score(y, predictions)
    }

def log_results(model_name: str, model, model_results: dict, params: dict):
    """Log evaluation results with MLflow and DVC Live."""
    test_size = params['data_collection']['test_size']
    n_estimators = params['model_building']['n_estimators']  # Load n_estimators

    # Logging with MLflow
    with mlflow.start_run(run_name=f"{model_name} Evaluation"):
        mlflow.set_tracking_uri("https://dagshub.com/abdulghaffaransari/water-potability-prediction.mlflow")
        for metric, value in model_results.items():
            mlflow.log_metric(metric.capitalize(), value)
        mlflow.log_param("Test Size", test_size)
        mlflow.log_param("n_estimators", n_estimators)  # Log n_estimators
        mlflow.sklearn.log_model(model, f"{model_name}_model")

    # Logging with DVC Live
    with Live(save_dvc_exp=True) as live:
        for metric, value in model_results.items():
            live.log_metric(metric.capitalize(), value)
        live.log_param("Test Size", test_size)
        live.log_param("n_estimators", n_estimators)  # Log n_estimators


def save_metrics_to_json(metrics_path: str, rf_results: dict, gb_results: dict) -> None:
    """Save evaluation results to a JSON file."""
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({"Random Forest": rf_results, "Gradient Boosting": gb_results}, f)

def main():
    """Main function to execute the evaluation process."""
    try:
        params_path = "params.yaml"
        data_path = "./src/data/processed/test_processed.csv"
        rf_model_path = "./src/model/random_forest_model.pkl"
        gb_model_path = "./src/model/gradient_boosting_model.pkl"
        metrics_path = "reports/metrics.json"

        # Load parameters
        params = load_params(params_path)

        # Load data
        test_data = load_data(data_path)
        X_test, y_test = prepare_data(test_data)

        # Evaluate Random Forest model
        rf_model = load_model(rf_model_path)
        rf_results = evaluate_model(rf_model, X_test, y_test)
        log_results("Random Forest", rf_model, rf_results, params)

        # Evaluate Gradient Boosting model
        gb_model = load_model(gb_model_path)
        gb_results = evaluate_model(gb_model, X_test, y_test)
        log_results("Gradient Boosting", gb_model, gb_results, params)

        # Save the evaluation results to a JSON file
        save_metrics_to_json(metrics_path, rf_results, gb_results)

        print("Model evaluations completed successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
