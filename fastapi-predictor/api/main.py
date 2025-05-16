from fastapi import FastAPI
from api.schemas import BloodTest
from api.predictor import Predictor
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # <-- Dostosuj do portu Springa
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# <- tutaj zmieniasz ścieżkę na models/anemia/modelXGBoost.pkl
predictor = Predictor(model_path="models/anemia/modelXGBoost.pkl")

@app.get("/")
def read_root():
    return {"message": "Medical Prediction API is running."}

@app.post("/predict")
def predict(blood_test: BloodTest):

    data = blood_test.dict()
    print("Got: ", data)
    result = predictor.predict(data)
    return result
