from fastapi import FastAPI,status,HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import joblib

app = FastAPI(
    title="Deploy Delivery ETA Prediction",
    version="0.0.1"
)

## print

# -------------------------------------------------
# Load AI Model
# -------------------------------------------------
model = joblib.load("model/LSTM_Model_V1.pkl")

@app.post("/api/v1/predict-delivery-ETA", tags=["Delivery-ETA"])
async def predict(
        Delivery_person_Age: float,
        Delivery_person_Ratings: float,
        distance: float
):
    dictionary = {
        'Delivery_person_Age': Delivery_person_Age,
        'Delivery_person_Ratings': Delivery_person_Ratings,
        'distance': distance
    }

    try:
        df = pd.DataFrame(dictionary, index=[0])
        prediction = model.predict(df)
        return JSONResponse(
            content=prediction.tolist(),
            status_code=status.HTTP_200_OK
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )