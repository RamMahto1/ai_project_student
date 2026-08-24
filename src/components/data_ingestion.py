from src.logger import logging
from src.exception import CustomException
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","data.csv")


class DataIngestion:
    def __init__(self):
        self.Ingestion_config = DataIngestionConfig()


    def initiate_data_ingestion(self):
        logging.info("data ingestion has started")
        try:
            df = pd.read_csv("notebook/stud.csv")
            logging.info("read the data as data frame")

            os.makedirs(os.path.dirname(self.Ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.Ingestion_config.raw_data_path,index=False,header=True)

            train_set, test_set = train_test_split(df,test_size=0.20, random_state=42)
            train_set.to_csv(self.Ingestion_config.train_data_path,index=False, header=True)
            test_set.to_csv(self.Ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data ingestion has completed")

            return(
                self.Ingestion_config.train_data_path,
                self.Ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)