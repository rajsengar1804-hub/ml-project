import os 
import sys
from src.logger import logging 
from src.exception import CustomException
from src.utils import saveobject
import numpy as np 
import pandas as pd 
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import saveobject ,evaluate_models


@dataclass 
class ModelTrainerConfig :
    trained_model_file_path = os.path.join("artifacts","Model.pkl")


class ModelTrainer :
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()   # ✅ FIX 1

    def initiate_model_trainer(self, train_array, test_array):
        try :
            logging.info("Split training and test input data")

            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            models_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_score = max(sorted(models_report.values()))

            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f"model_name={best_model_name} and accuracy={best_model_score}")

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(
                f"Best found model on both training and testing dataset {best_model_name}"
            )

            best_model.fit(X_train, y_train)   

            saveobject(
                filepath=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )




            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as ex :
            raise CustomException(ex, sys)
