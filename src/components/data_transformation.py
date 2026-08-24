from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

from dataclasses import dataclass

import pandas as pd
import numpy as np
import os
import sys


@dataclass
class DataTransformationConfig:
    preprocessor_file_path_obj: str = os.path.join(
        "artifacts",
        "preprocessor.pkl"
    )


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_obj(self):

        try:
            numerical_features = [
                "reading_score",
                "writing_score"
            ]

            categorical_features = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch"
            ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent")
                    ),
                    (
                        "one_hot_encoder",
                        OneHotEncoder(handle_unknown="ignore")
                    )
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "num",
                        num_pipeline,
                        numerical_features
                    ),
                    (
                        "cat",
                        cat_pipeline,
                        categorical_features
                    )
                ]
            )

            logging.info(
                f"Numerical columns: {numerical_features}"
            )

            logging.info(
                f"Categorical columns: {categorical_features}"
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(
        self,
        train_path,
        test_path
    ):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data as DataFrames")

            logging.info("Getting preprocessing object")

            preprocessor_obj = self.get_data_transformer_obj()

            target_column = "math_score"

            input_feature_train_df = train_df.drop(
                columns=[target_column]
            )

            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(
                columns=[target_column]
            )

            target_feature_test_df = test_df[target_column]

            logging.info("Applying preprocessing to training data")

            input_feature_train_arr = preprocessor_obj.fit_transform(
                input_feature_train_df
            )

            logging.info("Applying preprocessing to test data")

            input_feature_test_arr = preprocessor_obj.transform(
                input_feature_test_df
            )

            train_arr = np.c_[
                input_feature_train_arr,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                np.array(target_feature_test_df)
            ]

            save_object(
                file_path=self.data_transformation_config.preprocessor_file_path_obj,
                obj=preprocessor_obj
            )

            logging.info("Preprocessor object saved successfully")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_file_path_obj
            )

        except Exception as e:

            logging.error(
                "Error occurred during data transformation"
            )

            raise CustomException(e, sys)