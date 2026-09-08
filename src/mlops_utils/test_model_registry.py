import os
import pickle

def test_model_file_exists():
    assert os.path.exists("artifacts/model.pkl")



def test_model_can_be_loaded():
    with open("artifacts/model.pkl", "rb") as file:
        model = pickle.load(file)

    assert model is not None