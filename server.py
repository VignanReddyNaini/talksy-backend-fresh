from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "http://localhost:8000"}})  # Allow localhost:8000
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

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data or 'video' not in data:
            return jsonify({'char': '', 'word': '', 'sentence': ''}), 200
        # Decode base64 image
        img_data = data['video'].split(',')[1]  # Remove data URL prefix
        img_data = base64.b64decode(img_data)
        img = Image.open(BytesIO(img_data)).convert('L')  # Convert to grayscale
        img = img.resize((64, 64), Image.LANCZOS)  # Resize to 64x64
        img = np.array(img) / 255.0  # Normalize to [0, 1]
        img = np.expand_dims(img, axis=(0, -1))  # Add batch and channel dimensions: (1, 64, 64, 1)
        # Prediction
        prediction = model.predict(img)
        predicted_index = np.argmax(prediction[0])
        # Map index to gesture (assuming indices 0 to 5 correspond to A, B, O, D, L, W)
        gestures = list(gesture_to_word.keys())
        if 0 <= predicted_index < len(gestures):
            word = gesture_to_word[gestures[predicted_index]]
            sentence = f"Detected: {word}"
        else:
            word = 'Unknown'
            sentence = 'Unknown gesture'
        return jsonify({'char': '', 'word': word, 'sentence': sentence}), 200
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)