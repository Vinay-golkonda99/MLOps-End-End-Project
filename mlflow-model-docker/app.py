import mlflow.pyfunc
from fastapi import FastAPI, Request
import pandas as pd

app = FastAPI()

# Load model from the downloaded MLflow model directory
model = mlflow.pyfunc.load_model("downloaded_model")

@app.get("/")
def root():
    return {"message": "MLflow Model API is running"}

@app.post("/predict")
async def predict(request: Request):
    input_json = await request.json()

    # Expecting input as {"data": [{"MedInc": 8.3, "HouseAge": 21.0, ...}]}
    try:
        data = input_json["data"]
        df = pd.DataFrame(data)
    except Exception as e:
        return {"error": f"Invalid input format: {str(e)}"}

    # Run prediction
    try:
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}
