from src.logger import logging
from src.exception import CustomException
import sys
import pandas as pd
import os

class DataValidation:
    def __init__(self,train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data


    def initiate_data_validation(self):
        try:
            train_df = pd.read_csv(self.train_data)
            test_df = pd.read_csv(self.test_data)

            logging.info(f"train data shape: \n{train_df.shape}")
            logging.info(f"test data shape: \n{test_df.shape}")

            logging.info(f"train data null value: \n{train_df.isnull().sum()}")
            logging.info(f"test data null value: \n{test_df.isnull().sum()}")

            logging.info(f"train data null value: \n{train_df.duplicated().sum()}")
            logging.info(f"test data duplicate value: \n{test_df.duplicated().sum()}")

            logging.info(f"train data statatics test: \n{train_df.describe()}")
            logging.info(f"test data statatics test: \n{test_df.describe()}")

            expected_columns = [
        'gender',
        'race_ethnicity',
        'parental_level_of_education',  # ← comma
        'test_preparation_course',
        'lunch',
        'reading_score',
        'writing_score'
    ]

            missing_columns = [col for col in expected_columns if col not in train_df.columns]

            if missing_columns:
                raise CustomException(f"misssing column in train data{missing_columns}",sys)

        except Exception as e:
            raise CustomException(e,sys)