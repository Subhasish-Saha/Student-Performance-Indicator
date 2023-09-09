import os
import sys
import pandas as pd
from src.exception import Custom_Exception
from src.logger import logging
from src.components.data_transformation import (
    DataTransformation,
    DataTransformationConfig,
)
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self) -> tuple[str, str]:
        """
            Read raw data from a specific data source, split it into a train set and test set, save them,
            and return the file paths to the train and test data.

            Returns:
                tuple[str, str]: A tuple containing the file paths to the following elements in order:
                    - str: File path to the training data.
                    - str: File path to the testing data.

            Raises:
                Custom_Exception: If an error occurs during data ingestion, a Custom_Exception is raised.
        """
        logging.info(f"Entered the data ingestion method")
        try:
            file_path = f"Data/student_data.csv"
            df = pd.read_csv(file_path)
            logging.info(
                f"Read the dataset as a dataframe from the local directory : {file_path}."
            )
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Ingestion of the data is complete.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise Custom_Exception(sys, e)


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    print(
        modeltrainer.inititate_model_trainer(train_array=train_arr, test_array=test_arr)
    )
