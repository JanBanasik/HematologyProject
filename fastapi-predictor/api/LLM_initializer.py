from contextlib import asynccontextmanager
from predictor import Predictor
def load_your_llm_model():
    print("Ładowanie modelu LLM...")
    from transformers import pipeline # przykład
    try:
        model_id = "speakleash/Bielik-1.5B-v3"
        generator = pipeline("text-generation", model=model_id)
        print("Model LLM załadowany.")
        return generator
    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu LLM: {e}")
        return None

@asynccontextmanager
async def lifespan_manager(app):
    print("Rozpoczynanie cyklu życia aplikacji...")

    app.state.llm_generator = load_your_llm_model()

    print("Ładowanie modelu predykcyjnego...")
    model_path = "modelXGBoost.pkl"
    try:
        predictor_instance = Predictor(model_path=model_path)
        app.state.predictor = predictor_instance
        print("Model predykcyjny załadowany.")
    except Exception as e:
        print(f"BŁĄD: Nie udało się załadować modelu predykcyjnego z {model_path}: {e}")
        app.state.predictor = None
        raise RuntimeError(f"Fatal error: Could not load prediction model from {model_path}") from e

    yield

    print("Zamykanie cyklu życia aplikacji...")

    app.state.predictor = None

    app.state.llm_generator = None