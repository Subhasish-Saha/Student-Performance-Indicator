import sys
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from dataclasses import dataclass

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
            Get a data transformation object that handles numerical and categorical columns.

            Returns:
                ColumnTransformer: A ColumnTransformer object that preprocesses numerical columns
                using median imputation and standard scaling, and preprocesses categorical columns
                using most frequent imputation, one-hot encoding, and standard scaling.

            Raises:
                Custom_Exception: If an error occurs while creating the data transformation object, a Custom_Exception is raised.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False)),
                ]
            )
            logging.info(
                f"Numerical Columns : {numerical_columns}, standard scaling completed"
            )
            logging.info(
                f"Categorical Columns : {categorical_columns}, encoding completed"
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise Custom_Exception(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str) -> tuple:
        """
            Read train and test data from CSV files, preprocess them using a data transformation object,
            and return the processed train and test arrays along with the file path to the preprocessing object.

            Args:
                train_path (str): The file path to the training data CSV file.
                test_path (str): The file path to the testing data CSV file.

            Returns:
                tuple: A tuple containing the following elements in order:
                    - numpy.ndarray: Processed training data array.
                    - numpy.ndarray: Processed testing data array.
                    - str: File path to the saved preprocessing object.

            Raises:
                Custom_Exception: If an error occurs during data transformation, a Custom_Exception is raised.

            Note:
                This function assumes that the target column name is "math_score" and that there are numerical
                columns named "writing_score" and "reading_score" in the data.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Read the train and test data.")
            logging.info(f"Obtaining the preprocessing object.")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                f"Applying the preprocessing object on the training dataframe and testing dataframe."
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info(f"Saving the preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise Custom_Exception(e, sys)
