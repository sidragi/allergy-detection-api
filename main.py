from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image
import io
import tensorflow as tf
import keras

app = FastAPI()

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the Keras 3 model
model = keras.saving.load_model("allergyDetection.h5")

@app.get("/")
async def root():
    return {"message": "Allergy Detection API is up."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type.split('/')[0] != 'image':
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((256, 256))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0][0]

        if prediction < 0.1:
            result = "Severe allergy. Please visit a doctor immediately."
        elif prediction < 0.3:
            result = "Moderate allergy. Please consult a doctor."
        elif prediction < 0.5:
            result = "Mild allergy. Monitor the condition."
        elif prediction < 0.7:
            result = "Low chance of allergy. Keep an eye on symptoms."
        else:
            result = "No signs of allergy."

        return {"result": result, "confidence": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
