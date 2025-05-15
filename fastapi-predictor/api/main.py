from fastapi import FastAPI
from api.schemas import BloodTest
from api.predictor import Predictor

app = FastAPI()

# <- tutaj zmieniasz ścieżkę na models/anemia/modelXGBoost.pkl
predictor = Predictor(model_path="models/anemia/modelXGBoost.pkl")

@app.get("/")
def read_root():
    return {"message": "Medical Prediction API is running."}

@app.post("/predict")
def predict(blood_test: BloodTest):
    data = blood_test.dict()
    result = predictor.predict(data)
    return result
