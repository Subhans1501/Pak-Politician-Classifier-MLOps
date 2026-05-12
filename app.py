from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
from PIL import Image
import io
import os
app = FastAPI(title="Pakistani Politician Classifier API", version="1.0")
MODEL_PATH = "saved_models/ResNet50_politicians.h5"

if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH,compile=False)
else:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}. Ensure 'dvc pull' was executed.")

CLASSES = [
    "ali_wazir", "asif_ali_zardari", "benazir_bhutto", "bilawal_bhutto",
    "fawad_chaudhry", "hamza_shahbaz", "imran_khan", "jahangir_tareen",
    "lt_gen_ahmed_sharif_chaudhry", "maryam_nawaz", "maulana_fazlur_rehman",
    "murad_ali_shah", "nawaz_sharif", "pervez_musharraf", "shehbaz_sharif", "shireen_mazari"
]

@app.get("/")
def health_check():
    return {"status": "Active", "model_loaded": model is not None}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image = image.resize((224, 224))
        
        img_array = tf.keras.preprocessing.image.img_to_array(image)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array) 
        predictions = model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])

        return JSONResponse({
            "politician": CLASSES[predicted_class_idx],
            "confidence_score": round(confidence * 100, 2)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")