from pydantic import BaseModel

class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

class PredictionResponse(BaseModel):
    performance_score: float
    recommendation: str
