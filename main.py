from fastapi import FastAPI, UploadFile, File, HTTPException #fast api to make the api then theres file uploding, handelling and http 
from fastapi.middleware.cors import CORSMiddleware #need to read more about it, have taken from template example
import numpy as np #the model accepts the input as an numpy array of size 1
import cv2 #to convert the model
import tensorflow as tf # to handle keras lib
import keras #keras lib main thing where the the model is trained on
import io
from pydantic import BaseModel

app = FastAPI() # app init

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

@app.get("/") #to turn on the api
async def root():
    return {"message": "Allergy Detection API is up."}

@app.post("/predict") #to get post requests using preidct can be changed also 
async def predict(file: UploadFile = File(...)): #takes file input
    if file.content_type.split('/')[0] != 'image': #checks if image or not
        raise HTTPException(status_code=400, detail="Invalid image file")#error

    try: #error handelling
        image_data = await file.read()
        image_bytes = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)#cv2 reqads the image

        if img is None:
            raise HTTPException(status_code=400, detail="Image decoding failed")

        resize = tf.image.resize(img, (256, 256)) #resizes the image 
        sev = float(model.predict(np.expand_dims(resize / 255.0, 0))[0][0]) #puts the image into a numpy array to input into the model

        if sev < 0.1: #cetrain conditions based on model output, its vague 
            result = "Severe allergy. Please visit a doctor immediately."
        elif sev < 0.3:
            result = "Moderate allergy. Please consult a doctor."
        elif sev < 0.5:
            result = "Mild allergy. Monitor the condition."
        elif sev < 0.7:
            result = "Low chance of allergy. Keep an eye on symptoms."
        else:
            result = "No signs of allergy."
#since the close 0 is for allergy and 1 is for not allergy that means we have to flip the values in order to make sense logically for percentage
        percentage = round((1 - sev) * 100, 2)

        return {"result": result, "severity": float(1-sev), "severity percentage": percentage}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
class ImageURL(BaseModel):
  url: str

@app.post("/predict_url")
async def predict_url(data: ImageURL):
    try:
        response = requests.get(data.url)
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to fetch image from URL")

        image_bytes = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Image decoding failed")

        resize = tf.image.resize(img, (256, 256))
        sev = float(model.predict(np.expand_dims(resize / 255.0, 0))[0][0])

        if sev < 0.1:
            result = "Severe allergy. Please visit a doctor immediately."
        elif sev < 0.3:
            result = "Moderate allergy. Please consult a doctor."
        elif sev < 0.5:
            result = "Mild allergy. Monitor the condition."
        elif sev < 0.7:
            result = "Low chance of allergy. Keep an eye on symptoms."
        else:
            result = "No signs of allergy."

        percentage = round((1 - sev) * 100, 2)

        return {"result": result, "severity": float(1-sev), "severity percentage": percentage}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"URL Prediction failed: {str(e)}")
