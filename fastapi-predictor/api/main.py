from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from schemas import BloodTest
from fastapi.middleware.cors import CORSMiddleware
from LLM_initializer import lifespan_manager


class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str


app = FastAPI(lifespan=lifespan_manager) # Używamy menedżera z LLM_initializer

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # <-- Dostosuj do portu Springa
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
def predict(blood_test: BloodTest):
    predictor = app.state.predictor

    if predictor is None:
         raise HTTPException(status_code=500, detail="Prediction model is not available. Failed to load at startup.")

    data = blood_test.dict()
    print("Got: ", data)
    try:
        result = predictor.predict(data)
        return result
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/generate", response_model=ChatResponse)
async def generate_chat_reply(request: ChatRequest):

    generator = app.state.llm_generator

    if generator is None:
         raise HTTPException(status_code=500, detail="AI model is not available. Failed to load at startup.")

    user_message = request.message
    print(f"Received message from Spring Boot: {user_message}")
    try:
        output = generator(user_message)
        if isinstance(output, list) and len(output) > 0 and 'generated_text' in output[0]:
            generated_text = output[0]['generated_text']

            reply = generated_text.strip()

            if generated_text.startswith(user_message):
                 reply = generated_text[len(user_message):].lstrip()

            if not reply:
                 reply = "Przepraszam, nie udało mi się wygenerować odpowiedzi (pusta odpowiedź po oczyszczeniu)."


        else:
             reply = "Przepraszam, model wygenerował nieoczekiwaną odpowiedź."
             print(f"Warning: Unexpected LLM output format: {output}")


        print(f"Generated reply: {reply}")
        return ChatResponse(reply=reply)

    except Exception as e:
        print(f"Error generating reply: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {e}")

@app.get("/")
async def read_root():
    # Sprawdź oba modele w statusie
    llm_loaded = hasattr(app.state, 'llm_generator') and app.state.llm_generator is not None
    predictor_loaded = hasattr(app.state, 'predictor') and app.state.predictor is not None

    status_message = "FastAPI Service is running."
    if llm_loaded and predictor_loaded:
        status_message += " Both AI and Prediction models loaded."
    elif llm_loaded:
        status_message += " AI model loaded, but Prediction model failed."
    elif predictor_loaded:
        status_message += " Prediction model loaded, but AI model failed."
    else:
        status_message += " Both AI and Prediction models failed to load."

    return {"message": status_message}