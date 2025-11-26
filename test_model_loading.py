import joblib
import time

model_path = "data/model.pkl"

start_time = time.time()
model = joblib.load(model_path)
end_time = time.time()

print(f"Model loaded in {end_time - start_time:.2f} seconds")