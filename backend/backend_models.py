from pydantic import BaseModel
from typing import Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import joblib

# Input schema for prediction
class Features(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    # Add more features as needed
    
    
    # Generate sample data
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    # Train a model
    model = RandomForestClassifier()
    model.fit(X, y)

    # Save the model
    os.makedirs("data", exist_ok=True)
    joblib.dump(model, "data/model.pkl")
    print("Model trained and saved successfully!")

    # Load the model and make a prediction
    model = joblib.load("data/model.pkl")
    input_data = [[85.0, 70.0, 90.0, 80.0]]
    prediction = model.predict(input_data)
    print(f"Prediction: {prediction}")

    model_path = os.path.join("data", "model.pkl")
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}. Please check the path.")
    except Exception as e:
        raise RuntimeError(f"Error loading the model: {e}")

# Response schema for prediction results
class PredictionResponse(BaseModel):
    performance: int
    confidence: Optional[float] = None  # Optional confidence score

# Example schema for database models (if needed)
class EmployeePerformance(BaseModel):
    employee_id: int
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    performance: int