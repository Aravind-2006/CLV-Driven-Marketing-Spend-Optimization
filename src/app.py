from fastapi import FastAPI
import xgboost as xgb
import numpy as np
from pydantic import BaseModel
from src.business_layer import evaluate_customer_spend

app = FastAPI(title="CLV Marketing Spend Optimization API")

# load the trained model
model = xgb.XGBRegressor()
model.load_model("model.json")


@app.get("/")
def home():
    return {"message": "CLV Prediction API is running"}


class Customer(BaseModel):
    recency: float
    frequency: float


@app.post("/predict")
def predict(customer: Customer):
    input_data = np.array([[customer.recency, customer.frequency]])
    log_prediction = model.predict(input_data)
    prediction = np.expm1(log_prediction[0])
    return {"predicted_monetary_value": float(prediction)}


@app.post("/segment")
def segment(customer: Customer):
    input_data = np.array([[customer.recency, customer.frequency]])
    log_prediction = model.predict(input_data)
    prediction = float(np.expm1(log_prediction[0]))
    result = evaluate_customer_spend(prediction)
    return result