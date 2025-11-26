from fastapi import FastAPI, HTTPException
from .schemas import Features, PredictionResponse
import joblib
import os
import numpy as np
from fastapi import FastAPI

app = FastAPI()

# Load model
model_path = "data/model.pkl"
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")


@app.post("/predict", response_model=PredictionResponse)
def predict(features: Features):

    input_data = np.array([[
        features.feature1,
        features.feature2,
        features.feature3,
        features.feature4
    ]])

    try:
        pred = float(model.predict(input_data)[0])

        # create simple recommendation logic
        recommendation = (
            "High performer" if pred >= 0.5 
            else "Needs improvement"
        )

        return PredictionResponse(
            performance_score=pred,
            recommendation=recommendation
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")
