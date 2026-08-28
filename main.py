from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.data_validation import DataValidation
from src.components.model_trained import ModelTrainer

# step: 1 Data Ingestion
data_ingestion = DataIngestion()
train_path, test_path = data_ingestion.initiate_data_ingestion()
logging.info("data ingestion completed")

# step: 2 Data Transformation
data_transformation = DataTransformation()
train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_path,test_path)

logging.info("data transformation has completed")

# step: 3 Data Validation
data_validation = DataValidation(train_path, test_path)
data_validation.initiate_data_validation()
logging.info("data validation has completed")

# step: 4 Model trained
model_trained=ModelTrainer()
best_model_path = model_trained.initiate_model_trainer(train_arr, test_arr)
logging.info(f"model saved at path:{best_model_path}")







# logging.info("logging started")

# try:
#     a = 0
#     b = 1
#     result= a/b
#     print(result,"zero divided by 1")
# except Exception as e:
#     raise CustomException(e,sys)
