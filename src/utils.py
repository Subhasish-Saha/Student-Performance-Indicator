import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from src.exception import Custom_Exception
from src.logger import logging


def save_object(file_path: str, obj: object) -> None:
    """
    Save the given object to a file at the specified file path.

    Args:
        file_path (str): The path to the file where the object will be saved.
        obj (object): The object to be saved.

    Raises:
        Custom_Exception: If an error occurs during the saving process, a Custom_Exception is raised.

    Note:
        This function creates any necessary directories in the file path if they don't exist.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise Custom_Exception(e, sys)


def load_object(file_path: str) -> pickle:
    """
    Load an object from a file.

    Args:
        file_path (str): The path to the file from which to load the object.

    Returns:
        object: The loaded object.

    Raises:
        Custom_Exception: If an error occurs while loading the object, a Custom_Exception is raised.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise Custom_Exception(e, sys)


def evaluate_models(
    X_train: np.array,
    y_train: np.array,
    X_test: np.array,
    y_test: np.array,
    models: dict,
    params: dict,
) -> float:
    """
    Evaluate machine learning models using hyperparameter tuning and return a dictionary
    of model names and their corresponding R2 scores on the test data.

    Parameters:
        X_train (np.array): The training data features.
        y_train (np.array): The training data labels.
        X_test (np.array): The test data features.
        y_test (np.array): The test data labels.
        models (dict): A dictionary of machine learning models, where keys are model names
                    and values are model objects.
        params (dict): A dictionary specifying hyperparameter search spaces for each model.

    Returns:
        dict: A dictionary containing model names as keys and their R2 scores on the test data as values.

    The function performs the following steps:
    1. Iterates through the provided list of models.
    2. Performs hyperparameter tuning for each model using GridSearchCV with cross-validation.
    3. Trains each model with the best hyperparameters on the training data.
    4. Calculates and stores the R2 score of each model on both the training and test data.
    5. Returns a dictionary where model names are keys and R2 scores on the test data are values.
    """
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]
            logging.info(
                f"Hyper parameter tuning for {list(models.values())[i]} started."
            )
            gs = GridSearchCV(model, param, cv=3, n_jobs=-1)
            gs.fit(X_train, y_train)
            model.set_params(**gs.best_params_)
            logging.info(
                f"Hyper parameter tuned for {list(models.values())[i]} and the best parameters are used for model training."
            )
            model.fit(X_train, y_train)

            # model.fit(X_train, y_train)  # Train model
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise Custom_Exception(e, sys)
