import sys
import os
import numpy as np
import pandas as pd
from src.exception import Custom_Exception
from src.utils import load_object


class PredictPipeline:
    def __init__(self) -> None:
        pass

    def predict(self,features):
        """
            Make predictions using a trained model on the provided features.

            Args:
                features (numpy.ndarray): Input features for prediction.

            Returns:
                float: Predicted value based on the input features.

            Raises:
                Custom_Exception: If an error occurs during prediction, a Custom_Exception is raised.
        """
        try:
            model_path = os.path.join('artifacts','model.pkl')
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model = load_object(file_path = model_path)
            preprocessor = load_object(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            raise Custom_Exception(e,sys)


class CustomData:
    def __init__(  self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    
    def get_data_as_data_frame(self):
        """
            Create a pandas DataFrame from the attributes of the CustomData object.

            Returns:
                pandas.DataFrame: A DataFrame containing the attributes as columns.

            Raises:
                Custom_Exception: If an error occurs during DataFrame creation, a Custom_Exception is raised.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            raise Custom_Exception(e,sys)