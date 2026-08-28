import os
import sys
import pandas as pd

from src.utils import load_obj
from src.exception import CustomException


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self, features):
        try:
            # Model path
            model_path = os.path.join(
                "artifacts",
                "model.pkl"
            )

            # Preprocessor path
            preprocessor_path = os.path.join(
                "artifacts",
                "preprocessor.pkl"
            )

            # Load model
            model = load_obj(model_path)

            # Load preprocessor
            preprocessor = load_obj(preprocessor_path)

            # Transform input data
            data_scaled = preprocessor.transform(features)

            # Make prediction
            prediction = model.predict(data_scaled)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        gender,
        race_ethnicity,
        parental_level_of_education,
        lunch,
        reading_score,
        writing_score
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):

        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [
                    self.parental_level_of_education
                ],
                "lunch": [self.lunch],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)