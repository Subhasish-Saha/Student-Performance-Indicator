import os
import sys
import numpy as np
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score

from src.exception import Custom_Exception
from src.logger import logging

from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainerConfig()

    def inititate_model_trainer(self, train_array: np.array, test_array: np.array) -> float:
        '''
            This function splits the data into train and test sets, trains various machine learning models 
            on the training data, selects the best-performing model, and saves it. It then evaluates the 
            selected model on the test data and returns the R2 score.

            Parameters:
                train_array (np.array): An array containing the training data, including features and labels.
                test_array (np.array): An array containing the test data, including features and labels.

            Returns:
                float: The R2 score of the selected model on the test data.

            Raises:
                Custom_Exception: If no best model with a score greater than or equal to 0.6 is found.

            The function follows these steps:
            1. Splits the input data into training and test sets.
            2. Defines a set of machine learning models with default hyperparameters.
            3. Defines hyperparameter search spaces for some of the models.
            4. Evaluates the models on the training and test data, storing their performance in a dictionary.
            5. Selects the best-performing model based on the highest score.
            6. If the best model's score is less than 0.6, it raises a Custom_Exception.
            7. Saves the best model to a specified file path.
            8. Uses the best model to make predictions on the test data and calculates the R2 score.
            9. Returns the R2 score.
        '''
        try:
            logging.info(f"Split training and test inout data.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
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
                "KNeighborsRegressor": KNeighborsRegressor(),
            }
            params = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Extratree Regressor": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {  #'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {  #'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "KNeighborsRegressor": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    # 'loss':['linear','square','exponential'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                params=params,
            )
            # Getting the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Getting the name of the best model from the dictionary
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Custom_Exception(f"No best model found")
            logging.info(
                f"Best model : {best_model} with score : {best_model_score} found on both training and testing set."
            )
            # dumping the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            logging.info(f"R2 Score : {r2_square}")

            return r2_square

        except Exception as e:
            raise Custom_Exception(e, sys)
