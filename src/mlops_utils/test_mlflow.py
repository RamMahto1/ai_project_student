import pickle

from src.mlops_utils.model_registry import model_register


# Load trained model
with open("artifacts/model.pkl", "rb") as file:
    model = pickle.load(file)


# Register model
version = model_register(
    model=model,
    model_name="student_math_predictor",
    experiment_name="student_prediction",
    run_name="first_registry_test",
    alias="production"
)


print(f"Registered model version: {version}")