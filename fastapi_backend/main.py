from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
import os
import logging
from PIL import Image
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define sign language signs based on your model's classes
SIGNS = ['HELLO', 'YES', 'NO', 'I LOVE YOU', 'GOOD', 'THANK YOU', 'Cute', "What", "Who"]

# Load the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "ai", "ai", "models", "hand_gesture_model.keras")
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

def preprocess_image(image):
    # Resize image to match model input size
    image = cv2.resize(image, (64, 64))
    # Normalize pixel values
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
async def root():
    return {"message": "Hand Gesture Recognition API is running"}

@app.post("/process_frame")
async def process_frame(frame_data: dict):
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data['frame'].split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Preprocess the image
        processed_image = preprocess_image(frame)
        
        # Make prediction
        prediction = model.predict(processed_image)
        predicted_sign = SIGNS[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))
        
        return {
            "sign": predicted_sign,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}