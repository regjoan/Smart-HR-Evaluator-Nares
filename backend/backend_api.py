from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app = FastAPI(title="Smart HR Evaluator API")

# -----------------------------
# Load Model
# -----------------------------
model_path = os.path.join("data", "model.pkl")

try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# -----------------------------
# Pydantic Schema
# -----------------------------
class EmployeeData(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

class EvaluationResult(BaseModel):
    performance_score: int
    recommendation: str


# -----------------------------
# Predict Endpoint
# -----------------------------
@app.post("/predict", response_model=EvaluationResult)
def predict(employee: EmployeeData):
    try:
        input_data = [[
            employee.feature1,
            employee.feature2,
            employee.feature3,
            employee.feature4
        ]]

        performance_score = int(model.predict(input_data)[0])

        recommendation = (
            "Promotion Recommended" if performance_score > 80 else
            "Needs Improvement" if performance_score > 50 else
            "Underperforming - Consider Training"
        )

        return EvaluationResult(
            performance_score=performance_score,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")


# -----------------------------
# Same logic for /evaluate
# -----------------------------
@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(employee: EmployeeData):
    try:
        input_data = [[
            employee.feature1,
            employee.feature2,
            employee.feature3,
            employee.feature4
        ]]

        performance_score = int(model.predict(input_data)[0])

        recommendation = (
            "Promotion Recommended" if performance_score > 80 else
            "Needs Improvement" if performance_score > 50 else
            "Underperforming - Consider Training"
        )

        return EvaluationResult(
            performance_score=performance_score,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation error: {e}")
