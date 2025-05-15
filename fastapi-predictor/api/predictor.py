import os
import pickle
import joblib
import numpy as np

class Predictor:
    def __init__(self, model_path: str):
        preprocess_dir = os.path.join(os.path.dirname(__file__), "..", "preprocess", "anemia")
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

        self.scaler = joblib.load(os.path.join(preprocess_dir, "scaler.pkl"))
        self.pca = joblib.load(os.path.join(preprocess_dir, "pca.pkl"))

        # Mapowanie predykcji na nazwy klas
        self.type_mapping = {
            0: "Anemia Mikrocytarna",
            1: "Anemia Makrocytarna",
            2: "Anemia Normocytarna",
            3: "Anemia Hemolityczna",
            4: "Anemia Aplastyczna",
            5: "Healthy"
        }

    def predict(self, data: dict):
        # Przekształcanie wejścia do odpowiedniego formatu
        input_features = np.array([[data[feature] for feature in [
            'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC'
        ]]])

        # Skalowanie i redukcja PCA
        input_scaled = self.scaler.transform(input_features)
        input_pca = self.pca.transform(input_scaled)

        # Predykcja modelu
        prediction_idx = int(self.model.predict(input_pca)[0])
        probabilities = self.model.predict_proba(input_pca)[0]

        predicted_label = self.type_mapping.get(prediction_idx, "Unknown")
        probability_of_predicted_class = float(probabilities[prediction_idx])

        return {
            "prediction": predicted_label,
            "probability": probability_of_predicted_class,
            "probabilities": probabilities.tolist()
        }
