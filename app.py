from fastapi import FastAPI
from pydantic import BaseModel
# from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils import load_obj
from src.exception import CustomException
import os
import sys

from src.pipeline.predict_pipeline import (
    PredictPipeline,
    CustomData
)

app = FastAPI()

predict_pipeline = PredictPipeline()


class StudentData(BaseModel):

    gender: str
    race_ethnicity: str
    parental_level_of_education: str
    lunch: str
    reading_score: float
    writing_score: float


@app.post("/predict")
def predict(data: StudentData):

    try:
        custom_data = CustomData(
            gender=data.gender,
            race_ethnicity=data.race_ethnicity,
            parental_level_of_education=data.parental_level_of_education,
            lunch=data.lunch,
            reading_score=data.reading_score,
            writing_score=data.writing_score
        )

        dataframe = custom_data.get_data_as_dataframe()

        prediction = predict_pipeline.predict(dataframe)

        return {
            "prediction": prediction.tolist()
        }

    except Exception as e:
        raise CustomException(e, sys)