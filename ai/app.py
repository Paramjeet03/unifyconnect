from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import base64
import io
import logging
from PIL import Image
import traceback
import os
import ssl

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# SSL Configuration
ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
ssl_context.load_default_certs()

try:
    # Initialize MediaPipe
    logger.info("Initializing MediaPipe...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    logger.info("MediaPipe initialized successfully")

    # Define sign language signs based on your model's classes
    SIGNS = ['HELLO', 'YES', 'NO', 'I LOVE YOU', 'GOOD', 'THANK YOU', 'Cute', "What", "Who"]
    logger.info(f"Loaded signs: {SIGNS}")

    # Load the existing model
    logger.info("Loading hand gesture model...")
    model_path = os.path.join('ai', 'models', 'hand_gesture_model.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Try loading with custom_objects
        model = tf.keras.models.load_model(model_path, compile=False)
        logger.info("Model loaded successfully with compile=False")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        # Create a new model with the correct input shape
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(42,)),  # Updated input shape
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(len(SIGNS), activation='softmax')
        ])
        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
        logger.info("Created new model with correct input shape")

    # Warm up the model
    logger.info("Warming up the model...")
    dummy_input = np.random.rand(1, 42)  # Updated input shape
    model.predict(dummy_input)
    logger.info("Model warmed up successfully")

except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    logger.error(traceback.format_exc())
    raise

def extract_hand_features(landmarks):
    """Extract features from hand landmarks."""
    try:
        if not landmarks:
            return None
        
        # Convert landmarks to numpy array
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # Calculate palm center
        palm_center = np.mean(points[[0, 5, 9, 13, 17]], axis=0)
        
        # Calculate hand scale
        hand_scale = np.linalg.norm(points[0] - points[9])
        
        # Normalize points relative to palm center and scale
        normalized_points = (points - palm_center) / hand_scale
        
        # Flatten the points to match the model's input shape (42 features)
        flattened_points = normalized_points.flatten()
        
        return flattened_points
    except Exception as e:
        logger.error(f"Error in extract_hand_features: {str(e)}")
        return None

def process_frame(frame_data):
    """Process a single frame and return sign language prediction."""
    try:
        # Decode base64 image
        img_data = base64.b64decode(frame_data.split(',')[1])
        img = Image.open(io.BytesIO(img_data))
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        
        # Convert to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            # Get the first detected hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract features
            features = extract_hand_features(hand_landmarks)
            
            if features is not None:
                # Make prediction using the loaded model
                prediction = model.predict(features.reshape(1, 42))
                predicted_sign = SIGNS[np.argmax(prediction[0])]
                confidence = np.max(prediction[0])
                
                return {
                    'sign': predicted_sign,
                    'confidence': float(confidence)
                }
        
        return {'sign': None, 'confidence': 0.0}
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return {'sign': None, 'confidence': 0.0}

@app.route('/')
def serve_client():
    """Serve the video call client HTML file."""
    try:
        return send_from_directory('.', 'video_call_client.html')
    except Exception as e:
        logger.error(f"Error serving client: {str(e)}")
        return str(e), 500

@app.route('/process_frame', methods=['POST'])
def handle_frame():
    """Handle incoming video frames for sign language recognition."""
    try:
        data = request.json
        if 'frame' not in data:
            return jsonify({'error': 'No frame data provided'}), 400
        
        result = process_frame(data['frame'])
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error handling frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server with HTTPS...")
        app.run(host='0.0.0.0', port=5000, ssl_context='adhoc')
    except Exception as e:
        logger.error(f"Error starting server: {str(e)}")
        logger.error(traceback.format_exc()) 