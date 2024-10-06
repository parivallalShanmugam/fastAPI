from fastapi import FastAPI
import numpy as np
import joblib
from pydantic import BaseModel

app = FastAPI()

model_rf = joblib.load('D:/guvi/FASTAPI_V1/fastAPI/api/random_forest_model.pkl')
vectorizer = joblib.load('D:/guvi/FASTAPI_V1/fastAPI/api/tfidf_vectorizer.pkl')

class InputText(BaseModel):
    text: str

@app.post('/predict')
async def predict(input:InputText):
    text = input.text
    transformed_text = vectorizer.transform([text])
    prediction = model_rf.predict(transformed_text)
    prediction = int(prediction[0])
    if prediction == 1:
        sentiment = "positive"
    else:
        sentiment = "negative"
    return {"Prediction":sentiment}
