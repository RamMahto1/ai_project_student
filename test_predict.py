from src.pipeline.predict_pipeline import PredictPipeline, CustomData


data = CustomData(
    gender="male",
    race_ethnicity="group A",
    parental_level_of_education="bachelor's degree",
    lunch="standard",
    reading_score=70,
    writing_score=70
)

features = data.get_data_as_dataframe()

print("Input Data:")
print(features)

pipeline = PredictPipeline()

prediction = pipeline.predict(features)

print("Predicted Math Score:", prediction[0])