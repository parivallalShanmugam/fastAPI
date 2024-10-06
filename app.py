from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel
import os

app = FastAPI()

# Load the trained model and vectorizer from the models directory
try:
    model_rf = joblib.load(os.path.join('models', 'random_forest_model.pkl'))
    vectorizer = joblib.load(os.path.join('models', 'tfidf_vectorizer.pkl'))
except Exception as e:
    raise RuntimeError(f"Error loading model or vectorizer: {e}")

class InputText(BaseModel):
    text: str

@app.post('/predict')
async def predict(input: InputText):
    text = input.text

    # Handle empty text input
    if not text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty")

    try:
        transformed_text = vectorizer.transform([text])
        prediction = model_rf.predict(transformed_text)
        prediction = int(prediction[0])

        if prediction == 1:
            sentiment = "positive"
        else:
            sentiment = "negative"

        return {"Prediction": sentiment}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
