import os
import joblib
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data
X, y = make_classification(
    n_samples=100,
    n_features=4,
    n_informative=3,
    random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=50, max_depth=10)
model.fit(X, y)

# Save model
os.makedirs("data", exist_ok=True)
joblib.dump(model, "data/model.pkl")

print("Model trained and saved to data/model.pkl")
