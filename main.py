"""
    Importing all required Libraries

"""
import os
import warnings
import sys

import pandas as pd
import numpy as np
from scipy import rand

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn

from urllib.parse import urlparse
import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def evaluate_model(actual, pred):
    rmse = np.sqrt(mean_absolute_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual,pred)

    return (rmse, mae, r2)

def train_model(data, alpha=0.5, l1_ratio=0.5):
    data_train, data_test = train_test_split(data, test_size=0.3, random_state=123)

    X_train = data_train.drop("quality",axis=1)
    y_train = data_train["quality"]
    X_test = data_test.drop("quality", axis=1)
    y_test = data_test["quality"]

    with mlflow.start_run():
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=420)
        lr.fit(X_train.values, y_train.values)

        predict_y = lr.predict(X_test.values)

        rmse, mae, r2 = evaluate_model(y_test.values, predict_y)

        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        # Model registry does not work with file store
        if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
            mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticnetWineModel")
        else:
            mlflow.sklearn.log_model(lr, "model")


def read_data():
    # Read the wine-quality csv file from the URL
    csv_url = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
    
    return data

def setting_up_experiment():
    remote_server_uri = "http://0.0.0.0:5050"
    mlflow.set_tracking_uri(remote_server_uri)
    exp_name = "ElasticNet_Wine"
    mlflow.set_experiment(exp_name)
    print(mlflow.tracking.get_tracking_uri())
    print("Experiment Ready")

if __name__ == "__main__":

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    setting_up_experiment()
    data = read_data()
    train_model(data=data, alpha=alpha, l1_ratio=l1_ratio)
