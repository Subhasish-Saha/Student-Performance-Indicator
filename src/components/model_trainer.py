import os 
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self,train_array:np.array,test_array:np.array):
        try:
            logging.info(f'Split training and test inout data.')
            X_train,y_train,X_test,y_test = (
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
                "Extratree Regressor": ExtraTreeRegressor(),
                "KNeighborsRegressor" : KNeighborsRegressor()
            }

            model_report:dict = evaluate_models(X_train=X_train, y_train = y_train,
                                               X_test = X_test, y_test = y_test,
                                               models = models)
            # Getting the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Getting the name of the best model from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Custom_Exception(f"No best model found")
            logging.info(f'Best model : {best_model} with score : {best_model_score} found on both training and testing set.')
            # dumping the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            logging.info(f'R2 Score : {r2_square}')

            return r2_square


        except Exception as e:
            raise Custom_Exception(e,sys)