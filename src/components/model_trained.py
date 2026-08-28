from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


import os
import sys
import pandas as pd 
import numpy as np 
from src.utils import save_object,evaluate_metrics,load_obj



@dataclass

class ModelTrainerConfig:
    model_file_path_obj: str = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer = ModelTrainerConfig()


    def initiate_model_trainer(self,train_array, test_array):
        try:
            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]
            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            logging.info("initiate the model")
            models = {
                'LinearRegression':LinearRegression(),
                'Ridge':Ridge(),
                'Lasso':Lasso(),
                'DecisionTreeRegressor':DecisionTreeRegressor(),
                'RandomForestRegressor':RandomForestRegressor(),
                'AdaBoostRegressor':AdaBoostRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'XGBRegressor':XGBRegressor(),
                'CatBoostRegressor':CatBoostRegressor()
            }

            params = {
                'LinearRegression':{},
                'Ridge':{
                    'alpha':[0.1,0.01,0.2]
                },
                'Lasso':{
                    'alpha':[0.1,0.01,0.2]
                },
                'DecisionTreeRegressor':{
                    'max_features':[3,5,None],
                    'max_depth':[2,3,5]
                },
                'RandomForestRegressor':{
                    'n_estimators':[3,6,8]
                },
                'AdaBoostRegressor':{
                    'n_estimators':[10,30,50],
                    'learning_rate':[0.1,0.01,0.2],
                    'loss':['linear']

                },
                'CatBoostRegressor':{
                    'n_estimators':[100,200,300],
                    'max_depth':[2,4,5],
                    'learning_rate':[0.1,0.01,0.2]
                },
                'XGBRegressor':{
                    'n_estimators':[50,100,200],
                    'learning_rate':[0.1,0.01,0.3]
                },
                'GradientBoostingRegressor':{
                    'n_estimators':[10,30,50],
                    'learning_rate':[0.1,0.01,0.2]

                }}

            # Evaluate the model
            report, best_model_name, best_score, best_model = evaluate_metrics(
                  X_train, y_train, X_test, y_test, models, params)
            logging.info(f"Best model found: {best_model_name} with best score: {best_score}")

            save_object( 
                file_path=self.model_trainer.model_file_path_obj,
                obj=best_model
            )

            return report, best_model_name, best_model, best_score


        except Exception as e:
            raise CustomException(e,sys)
        