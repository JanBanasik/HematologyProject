# BloodAI – Artificial Intelligence for Anemia Diagnosis

[![Project Status](https://img.shields.io/badge/Status-In_Development-yellow)](https://github.com/JanBanasik/HematologyProject)  
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/JanBanasik/HematologyProject/blob/main/LICENSE)

## Project Overview

**BloodAI** is a web-based platform designed to assist doctors and patients in the diagnosis of anemia. It combines advanced AI models with an intuitive user interface to predict anemia types based on blood test results and offers an interactive medical chatbot.

The project is composed of two main components:

1. **Backend (Spring Boot):** Handles business logic, user management, database operations (MySQL), and communication with both the frontend and the AI service.
2. **AI Service (Python/FastAPI):** Provides endpoints for anemia prediction (using a trained XGBoost model), chatbot responses (via the Polish LLM *Bielik*), and medical summary generation (*Gemini 1.5 Flash*).

Currently, the frontend is implemented with Thymeleaf, but a migration to React is planned to enable a more decoupled architecture.

---

## 📸 Screenshots

### 🧪 Anemia Prediction Interface
![image](https://github.com/user-attachments/assets/a3c0d8ac-6e64-4c4b-af5c-446607dbe044)

### 💬 AI Medical Chatbot
![image](https://github.com/user-attachments/assets/985a98ae-307f-48fc-a157-01850cd1f20b)


### 📝 Generated Medical Summary (Epicrisis)
![image](https://github.com/user-attachments/assets/02eb62f1-acd9-4de5-93f1-8361f045cb41)


---

## Features

- **Anemia Type Prediction:** Identifies the most probable type of anemia (e.g., Microcytic, Macrocytic, Hemolytic, Aplastic, Normocytic) or classifies as *Healthy* based on key blood parameters.
- **Medical Chatbot:** An interactive assistant based on the Polish LLM *Bielik*, capable of answering questions related to hematology and anemia.
- **Medical Summary Generator:** Automatically creates clinical summaries (epicrises) based on blood test results and predictions, powered by *Gemini 1.5 Flash*.
- **Prediction History:** Logged-in users can browse a history of their past predictions.
- **User Authentication:** Secure registration and login system for personalized experiences.
- **User-Friendly Interface:** Clean and accessible web UI (currently Thymeleaf, migrating to React soon).

## Tech Stack

### Backend (Spring Boot)

- Java, Spring Boot
- Spring Security, Spring Data JPA, Hibernate
- MySQL
- Maven
- REST APIs via `RestTemplate`
- Thymeleaf (templating engine)

### AI Service (Python/FastAPI)

- Python, FastAPI
- XGBoost (for anemia prediction)
- CVAE (for synthetic training data generation)
- Hugging Face Transformers (Bielik LLM)
- Google Generative AI / Gemini 1.5 Flash (for epicrises)
- `python-dotenv`, `scikit-learn`, `pandas`, `numpy`, `torch`, `joblib`

### Frontend

- Thymeleaf (currently)
- HTML, CSS, JavaScript  
- React (planned)

## Installation Guide

To run the project locally, you need to set up both the Spring Boot backend and the Python AI service.

### 1. MySQL Database Setup

1. Install MySQL server.
2. Create a new database for the project.
3. Update the `application.properties` file in the Spring Boot project with your database URL, username, and password.  
   *(Spring Data JPA will automatically generate the required tables on first run.)*

### 2. AI Service Setup (Python/FastAPI)

1. Clone the repository:
    ```bash
    git clone https://github.com/JanBanasik/HematologyProject.git
    cd HematologyProject/fastapi-predictor
    ```
2. Create and activate a Python virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Linux/macOS
    .venv\Scripts\activate     # On Windows
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements-cuda.txt
    ```
    or:
    ```bash
    pip install -r requirements-cpu.txt
    ```
    (depending on your system and GPU availability)

4. **Download AI models:**  
   Ensure the XGBoost model (`modelXGBoost.json`) and scaler (`scaler.pkl`) are in the correct directories (`models/anemia`, `preprocess/anemia`).  
   The Bielik model (`speakleash/Bielik-1.5B-v3`) will be downloaded automatically by the `transformers` library on first use.

5. **API Key Setup:**  
   Create a `.env` file in the `fastapi-predictor` directory and add your Gemini API key:
    ```dotenv
    GEMINI_API_KEY=your_gemini_api_key_here
    ```
   *(You can obtain a key from [Google AI Studio](https://makersuite.google.com/app).)*

6. Run the FastAPI service:
    ```bash
    uvicorn main:app --reload
    ```
   The service will be available at `http://127.0.0.1:8000`.

### 3. Backend Setup (Spring Boot)

1. Navigate to the Spring Boot project directory.
2. Install Maven dependencies:
    ```bash
    mvn clean install
    ```
3. Run the Spring Boot app:
    ```bash
    mvn spring-boot:run
    ```
   It will be accessible by default at `http://localhost:8080`.

## How to Use

1. Open your browser and go to `http://localhost:8080`.
2. Register and log in.
3. Click on "Anemia Prediction" to access the blood test input form.
4. Enter the required parameters and submit the form.
5. View the predicted anemia type and generated epicrisis.
6. Explore your prediction history.
7. Use the chatbot (bottom-right chat bubble) to ask medical questions.

## Project Structure

```
HematologyProject/
├── fastapi-predictor/         # AI Service (Python/FastAPI)
│   ├── api/                   # API endpoints and AI logic
│   │   ├── LLM_initializer.py
│   │   ├── main.py
│   │   ├── predictor.py
│   │   └── ...
│   ├── data/                  # Synthetic data
│   ├── models/                # Trained ML models
│   ├── preprocess/            # Scaler and preprocessing files
│   └── ...
├── src/                       # Backend (Spring Boot)
│   └── main/
│       └── java/com/agh/anemia/
│           ├── controller/
│           ├── dto/
│           ├── model/
│           ├── repository/
│           ├── service/
│           └── ...
│       └── resources/         # Config, templates, static assets
└── ...
```

## Future Plans

- Migration to a React frontend
- Further AI model improvements
- More advanced chatbot interactions
- Integration of data visualizations

## Contact

For questions or collaboration inquiries, feel free to reach out:

- **Jan Banasik** – jan.jerzy.banasik@gmail.com  
- **Antoni Pater** – antonipaterbusiness@gmail.com  

Project developed as part of the **AI Lab AGH Kraków** student scientific society.
