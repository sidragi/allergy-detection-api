from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import tensorflow as tf
import keras
import io

app = FastAPI()

# CORS middleware taken from template
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# model made on google colab with dataset of 800 and trained on keras 3+
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
        image_bytes = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image decoding failed")

        resize = tf.image.resize(img, (256, 256))
        conf = model.predict(np.expand_dims(resize / 255.0, 0))[0][0]

        if conf < 0.1:
            result = "Severe allergy. Please visit a doctor immediately."
        elif conf < 0.3:
            result = "Moderate allergy. Please consult a doctor."
        elif conf < 0.5:
            result = "Mild allergy. Monitor the condition."
        elif conf < 0.7:
            result = "Low chance of allergy. Keep an eye on symptoms."
        else:
            result = "No signs of allergy."

        return {"result": result, "confidence": float(conf)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
