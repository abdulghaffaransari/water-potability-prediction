import numpy as np
import pandas as pd

import pickle
import json
from dvclive import Live
import yaml

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

def load_data(filepath : str) -> pd.DataFrame:
    try:
         return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}:{e}")

# test_data = pd.read_csv("./data/processed/test_processed.csv")


def prepare_data(data: pd.DataFrame) -> tuple[pd.DataFrame,pd.Series]:
    try:
        X = data.drop(columns=['Potability'],axis=1)
        y = data['Potability']
        return X,y
    except Exception as e:
        raise Exception(f"Error Preparing data:{e}")

# X_test = test_data.iloc[:, :-1].values
# y_test = test_data.iloc[:, -1].values


def load_model(filepath:str):
    try:
        with open(filepath,"rb") as file:
            model= pickle.load(file)
        return model
    except Exception as e:
        raise Exception(f"Error loading model from {filepath}:{e}")

# model = pickle.load(open("model.pkl","rb"))

def evaluation_model(model, X_test:pd.DataFrame, y_test:pd.Series) -> dict:
    try:
        params = yaml.safe_load(open("params.yaml","r"))
        
        test_size = params["data_collection"]["test_size"]
        n_estimators = params["model_building"]["n_estimators"]
        
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test,y_pred)
        pre = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)
        
        with Live(save_dvc_exp=True) as live:
            live.log_metric("Accuracy: ",acc)
            live.log_metric("Precision: ",pre)
            live.log_metric("recall: ",recall)
            live.log_metric("f1score: ",f1score)
            live.log_param("Test_Size: ",test_size)
            live.log_param("n_estimators: ",n_estimators)
            

        metrics_dict = {

            'acc':acc,
            'precision':pre,
            'recall' : recall,
            'f1_score': f1score
        }
        return metrics_dict
    except Exception as e:
        raise Exception(f"Error evaluating model : {e}")

# y_pred = model.predict(X_test)

# acc =accuracy_score(y_test,y_pred)
# pre = precision_score(y_test,y_pred)
# recall = recall_score(y_test,y_pred)
# f1score = f1_score(y_test,y_pred)


# metrics_dict = {

#         'acc':acc,
#         'precision':pre,
#         'recall' : recall,
#         'f1_score': f1score
#     }


def save_metrics(metrics:dict,metrics_path:str) -> None:
    try:
        with open(metrics_path,'w') as file:
            json.dump(metrics,file,indent=4)
    except Exception as e:
        raise Exception(f"Error saving metrics to {metrics_path}:{e}")
    
def main():
    try:
        test_data_path = "./src/data/processed/test_processed.csv"
        model_path = "./src/model/model.pkl"
        metrics_path = "./reports/metrics.json"

        test_data = load_data(test_data_path)
        X_test,y_test = prepare_data(test_data)
        model = load_model(model_path)
        metrics = evaluation_model(model,X_test,y_test)
        save_metrics(metrics,metrics_path)
    except Exception as e:
        raise Exception(f"An Error occurred:{e}")

if __name__ == "__main__":
    main()
