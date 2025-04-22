from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Update CORS configuration to be more explicit
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)

# Add CORS headers to all responses
@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# For debugging - root endpoint
@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running", "endpoints": ["/predict"]}), 200

# Load the model
try:
    model = tf.keras.models.load_model('sign_word_model.h5')
    print("Model loaded successfully. Input shape:", model.input_shape, "Output shape:", model.output_shape)
    # Define vocabulary based on your gesture-to-word mapping
    gesture_to_word = {
        'A': 'No',
        'B': 'Yes',
        'O': 'Good',
        'D': 'Thanku',
        'L': 'Hellow',
        'W': 'Bye'
    }
    num_classes = model.output_shape[-1]
    if len(gesture_to_word) != num_classes:
        print(f"Warning: Vocabulary size ({len(gesture_to_word)}) does not match output classes ({num_classes})")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    # Handle preflight requests
    if request.method == 'OPTIONS':
        return '', 204  # Simplified OPTIONS response

    try:
        print("Received prediction request")
        
        # Check if model is loaded
        if model is None:
            return jsonify({'char': '', 'word': 'Model Error', 'sentence': 'Model not loaded properly'}), 200
            
        # Check request content
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({'char': '', 'word': 'Error', 'sentence': 'Invalid request format'}), 400
            
        data = request.get_json()
        print("Request data type:", type(data))
        
        if not data or 'video' not in data:
            print("Missing video data in request")
            return jsonify({'char': '', 'word': 'Error', 'sentence': 'No video data provided'}), 400
            
        # Process image
        try:
            img_data = data['video'].split(',')[1]  # Remove data URL prefix
            img_data = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_data)).convert('L')  # Convert to grayscale
            img = img.resize((64, 64), Image.LANCZOS)  # Resize to 64x64
            img = np.array(img) / 255.0  # Normalize to [0, 1]
            img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions
            
            # Make prediction
            prediction = model.predict(img)
            predicted_index = np.argmax(prediction[0])
            
            # Map to word
            gestures = list(gesture_to_word.keys())
            if 0 <= predicted_index < len(gestures):
                word = gesture_to_word[gestures[predicted_index]]
                sentence = f"Detected: {word}"
            else:
                word = 'Unknown'
                sentence = 'Unknown gesture'
                
            print(f"Prediction complete: {word}")
            return jsonify({'char': '', 'word': word, 'sentence': sentence}), 200
        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({'char': '', 'word': 'Error', 'sentence': f'Image processing error: {str(e)}'}), 200
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)