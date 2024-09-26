# import pandas as pd
# import pickle
# from sklearn.ensemble import RandomForestClassifier
# import yaml

# def load_params(filepath : str) -> int:
#     try:
#         with open(filepath,"r") as file:
#             params = yaml.safe_load(file)
#         return params["model_building"]["n_estimators"]
#     except Exception as e:
#         raise Exception(f"Error Loading parameters from {filepath}:{e}")

# # Load the training data
# def load_data(data_path: str) -> pd.DataFrame:
#     try:
#         return pd.read_csv(data_path)
#     except Exception as e:
#         raise Exception(f"Error loading data from {data_path}: {e}")


# def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
#     try:
#         X = data.drop(columns=['Potability'], axis=1)
#         y = data['Potability']
#         return X, y
#     except Exception as e:
#         raise Exception(f"Error preparing data: {e}")


# # Train the RandomForestClassifier model
# def train_model(X: pd.DataFrame, y: pd.Series, n_estimators: int) -> RandomForestClassifier:
#     try:
#         clf = RandomForestClassifier(n_estimators=n_estimators)
#         clf.fit(X, y)
#         return clf
#     except Exception as e:
#         raise Exception(f"Error training model: {e}")



# # Save the trained model to a file using pickle
# def save_model(model: RandomForestClassifier, model_name: str) -> None:
#     try:
#         with open(model_name, "wb") as file:
#             pickle.dump(model, file)
#     except Exception as e:
#         raise Exception(f"Error saving model to {model_name}: {e}")


# def main():
#     try:
#         params_path = "params.yaml"
#         data_path = "./src/data/processed/train_processed.csv"
#         model_name = "./src/model/model.pkl"

#         n_estimators = load_params(params_path)
#         train_data = load_data(data_path)
#         X_train, y_train = prepare_data(train_data)

#         model = train_model(X_train, y_train, n_estimators)
#         save_model(model, model_name)
#         print("Model trained and saved successfully!")
#     except Exception as e:
#         print(f"An error occurred: {e}")

# if __name__ == "__main__":
#     main()























import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import BaseEstimator


def load_params(filepath: str) -> dict:
    """Load model parameters from a YAML file."""
    with open(filepath, "r") as file:
        try:
            params = yaml.safe_load(file)
            return params["model_building"]
        except Exception as e:
            raise Exception(f"Error loading parameters from {filepath}: {e}")


def load_data(data_path: str) -> pd.DataFrame:
    """Load training data from a CSV file."""
    try:
        return pd.read_csv(data_path)
    except Exception as e:
        raise Exception(f"Error loading data from {data_path}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and target variable from the dataset."""
    if 'Potability' not in data.columns:
        raise KeyError("Target column 'Potability' is missing from the data.")
        
    X = data.drop(columns=['Potability'], axis=1)
    y = data['Potability']
    return X, y


def create_model(model_type: str, params: dict) -> BaseEstimator:
    """Create a model based on the specified type and parameters."""
    if model_type == "random_forest":
        return RandomForestClassifier(n_estimators=params['n_estimators'], random_state=42)
    elif model_type == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=params['n_estimators'], random_state=42)
    else:
        raise ValueError("Unsupported model type!")


def train_model(model: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> BaseEstimator:
    """Train the specified model with the provided data."""
    try:
        model.fit(X, y)
        return model
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def save_model(model: BaseEstimator, model_path: str) -> None:
    """Save the trained model to a specified path."""
    with open(model_path, "wb") as file:
        try:
            pickle.dump(model, file)
        except Exception as e:
            raise Exception(f"Error saving model to {model_path}: {e}")


def main():
    """Main function to execute the model training and saving process."""
    params_path = "params.yaml"
    data_path = "./src/data/processed/train_processed.csv"
    rf_model_path = "./src/model/random_forest_model.pkl"
    gb_model_path = "./src/model/gradient_boosting_model.pkl"

    # Load parameters
    model_params = load_params(params_path)

    # Load and prepare data
    train_data = load_data(data_path)
    X_train, y_train = prepare_data(train_data)

    # Train and save models
    for model_type, model_path in [("random_forest", rf_model_path), ("gradient_boosting", gb_model_path)]:
        model = create_model(model_type, model_params)
        trained_model = train_model(model, X_train, y_train)
        save_model(trained_model, model_path)
        print(f"{model_type.replace('_', ' ').title()} model trained and saved successfully!")


if __name__ == "__main__":
    main()
