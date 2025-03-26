# import pickle
# import numpy as np
# import uvicorn
# from fastapi import FastAPI
# from pydantic import BaseModel
# from pathlib import Path

# # Define the model file path
# MODEL_PATH = Path("C:/Users/ankit/Downloads/breast_cancer/model.pkl")  # Update the path accordingly

# # Load the trained model
# with open(MODEL_PATH, "rb") as f:
#     model = pickle.load(f)

# # Initialize FastAPI app
# app = FastAPI()

# # Define request body
# class InputData(BaseModel):
#     features: list[float]  # Example: [1.2, 3.4, 5.6, ...]

# @app.post("/predict")
# def predict(data: InputData):
#     # Convert input to numpy array and reshape for model
#     input_array = np.array(data.features).reshape(1, -1)
    
#     # Make prediction
#     prediction = model.predict(input_array)

#     return {"prediction": prediction.tolist()}  # Convert NumPy to list for JSON response

# # Run Uvicorn server when script is executed
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)



from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from typing import List

# ✅ Fix file path
model_path = r"C:/Users/ankit/Downloads/breast_cancer/model.pkl"

# ✅ Load the trained model
with open(model_path, "rb") as file:
    model = pickle.load(file)
print("✅ Model loaded successfully!")

# ✅ Define request model
class InputData(BaseModel):
    features: List[float]  # Expecting a list of numerical features

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Allow CORS (for frontend requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (change this in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

print(f"✅ Model expects {model.n_features_in_} features")

@app.get("/")
def home():
    return {"message": "Breast Cancer Prediction API is running!"}

@app.post("/predict")
def predict(data: InputData):
    if len(data.features) != model.n_features_in_:
        return {"error": f"Expected {model.n_features_in_} features, but got {len(data.features)}"}
    
    input_array = np.array(data.features, dtype=np.float64).reshape(1, -1)
    prediction = model.predict(input_array)
    
    return {"prediction": int(prediction[0])}

