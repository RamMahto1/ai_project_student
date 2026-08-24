from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion

# step: 1 Data Ingestion
data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()
logging.info("data ingestion completed")


# logging.info("logging started")

# try:
#     a = 0
#     b = 1
#     result= a/b
#     print(result,"zero divided by 1")
# except Exception as e:
#     raise CustomException(e,sys)
