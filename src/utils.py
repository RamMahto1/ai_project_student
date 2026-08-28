import os
import pickle
import sys

import numpy as np
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_metrics(X_train, y_train, X_test, y_test, models, params):
    try:
        report = {}

        best_score = float("-inf")
        best_model_name = None
        best_model = None

        for model_name, model in models.items():

            logging.info(f"Model Evaluation: {model_name}")

            param = params.get(model_name)

            if param:
                gs = GridSearchCV(
                    model,
                    param_grid=param,
                    cv=3,
                    scoring="r2",
                    n_jobs=-1
                )

                gs.fit(X_train, y_train)

                model = gs.best_estimator_

            else:
                model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            r2score = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)

            report[model_name] = {
                "r2_score": r2score,
                "mean_squared_error": mse,
                "mean_absolute_error": mae,
                "root_mean_squared_error": rmse
            }

            logging.info(
                f"{model_name} - "
                f"R2: {r2score}, "
                f"MAE: {mae}, "
                f"MSE: {mse}, "
                f"RMSE: {rmse}"
            )

            if r2score > best_score:
                best_score = r2score
                best_model_name = model_name
                best_model = model

        return report, best_model_name, best_score, best_model

    except Exception as e:
        raise CustomException(e, sys)

def load_obj(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    