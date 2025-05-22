import os
import pickle
import re
import joblib
import numpy as np
import sys
from dotenv import load_dotenv
import google.generativeai as genai
import xgboost as xgb


class Predictor:
    def __init__(self, model_path: str):
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            self.gemini_model = None
            raise RuntimeError(f"Couldn't load gemini model")

        print(f"Attempting to initialize Predictor...")
        print(f"Predictor script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print(f"Current working directory: {os.getcwd()}")

        try:
            print(f"Attempting to load model from: {model_path}")

            MODEL_PATH = os.path.join(BASE_DIR, "models", "anemia", "modelXGBoost.json")
            self.model = xgb.XGBClassifier()
            self.model.load_model(MODEL_PATH)

            print("Model loaded successfully.")
            self.SCALER_PATH = os.path.join(BASE_DIR, "preprocess", "anemia", "scaler.pkl")


            print(f"Attempting to load scaler from calculated path: {self.SCALER_PATH}")

            if not os.path.exists(self.SCALER_PATH):
                print(f"Error: Scaler file not found at the calculated path {self.SCALER_PATH}")
                raise FileNotFoundError(f"Scaler file not found at {self.SCALER_PATH}. Ensure it's in the same directory as the Predictor script or adjust the path calculation.")


            self.scaler = joblib.load(self.SCALER_PATH)


            print(f"{type(self.scaler)=} loaded successfully from {self.SCALER_PATH}")
            print(f"Is self.scaler a scikit-learn transformer? {hasattr(self.scaler, 'transform')}")

            self.units = {
                    'RBC': '×10⁶/µL',
                    'HGB': 'g/dL',
                    'HCT': '%',
                    'MCV': 'fL',
                    'MCH': 'pg',
                    'MCHC': 'g/dL',
                    'RDW': '%',
                    'PLT': '×10³/µL',
                    'WBC': '×10³/µL'
                }

            self.type_mapping = {
                0: "Anemia Mikrocytarna",
                1: "Anemia Makrocytarna",
                2: "Anemia Hemolityczna",
                3: "Anemia Aplastyczna",
                4: "Anemia Normocytarna",
                5: "Healthy"
            }
            print("Predictor initialization complete.")

        except FileNotFoundError as e:
            print(f"Initialization failed: File not found error - {e}", file=sys.stderr)
            raise
        except pickle.UnpicklingError as e:
             print(f"Initialization failed: Loading error (pickle/joblib) - {e}", file=sys.stderr)
             print(f"Verify that '{self.SCALER_PATH}' and '{model_path}' are valid files saved with the correct library (pickle or joblib).", file=sys.stderr)
             raise
        except Exception as e:
            print(f"Initialization failed: An unexpected error occurred - {type(e).__name__}: {e}", file=sys.stderr)
            raise

    def predict(self, data: dict):
        print("Starting prediction...")
        try:
            input_features = np.array([[data[feature] for feature in [
                'RBC', 'HGB', 'HCT', 'MCV', 'MCH', 'MCHC', 'RDW', 'PLT', 'WBC'
            ]]])
            print(f"Input features shape: {input_features.shape}")

            input_scaled = self.scaler.transform(input_features)
            print("Data scaled successfully.")

            prediction_idx = int(self.model.predict(input_scaled)[0])
            probabilities = self.model.predict_proba(input_scaled)[0]
            print(f"{probabilities=}")
            print("Prediction performed successfully.")
            print(f"{prediction_idx=} prediction probability: {probabilities[prediction_idx]=}")

            predicted_label = self.type_mapping.get(prediction_idx, "Unknown")
            print(f"Predicted label: {predicted_label}")
            probability_of_predicted_class = float(probabilities[prediction_idx])
            print(f"Probability of predicted class: {probability_of_predicted_class}")
            result = {
                "prediction": predicted_label,
                "probability": probability_of_predicted_class,
                "probabilities": probabilities.tolist(),
                "epicrisis": self.generateMedicalEpicrisis(predicted_label,
                                                           probability_of_predicted_class,
                                                           input_features)
            }
            print(f"Prediction result: {result}")
            return result

        except Exception as e:
            print(f"Error during prediction: {type(e).__name__}: {e}", file=sys.stderr)
            raise

    def generateMedicalEpicrisis(self, predicted_label, probability_of_predicted_class, input_features) -> str:
        print(f"{input_features=}")
        RBC, HGB, HCT, MCV, MCH, MCHC, RDW, PLT, WBC = input_features[0]
        prompt = f"""
                                Jako hematolog, na podstawie danych pacjenta (RBC: {RBC} {self.units['RBC']}, HGB: {HGB} {self.units['HGB']}, 
                                HCT: {HCT} {self.units['HCT']}, MCV: {MCV} {self.units['MCV']}, MCH: {MCH} {self.units['MCH']}, 
                                MCHC: {MCHC} {self.units['MCHC']}, RDW: {RDW} {self.units['RDW']}, PLT: {PLT} {self.units['PLT']}, 
                                WBC: {WBC} {self.units['WBC']})
                                oraz przewidywania modelu AI: {'pacjent zdrowy' if predicted_label == 'Healthy' else 'pacjent choruje na ' + predicted_label} 
                                prawdopodobieństwo: {probability_of_predicted_class * 100:.2f}% napisz profesjonalną epikryzę medyczną w języku polskim.
                                """

        response = self.gemini_model.generate_content(prompt)

        clean_response = response.text.replace("**", "")
        clean_response = re.sub(r'\[.*?]', '', clean_response)

        return clean_response