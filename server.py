from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://talksy-frontend.vercel.app"}}, supports_credentials=True)

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', 'https://talksy-frontend.vercel.app')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

@app.route('/', methods=['GET'])
def index():
    return jsonify({"status": "API is running", "endpoints": ["/predict"]}), 200

try:
    model = tf.keras.models.load_model('sign_word_model.h5')
    print("Model loaded successfully. Input shape:", model.input_shape, "Output shape:", model.output_shape)
    gesture_to_word = {
        'A': 'No', 'B': 'Yes', 'O': 'Good', 'D': 'Thanku', 'L': 'Hellow', 'W': 'Bye'
    }
    num_classes = model.output_shape[-1]
    if len(gesture_to_word) != num_classes:
        print(f"Warning: Vocabulary size ({len(gesture_to_word)}) does not match output classes ({num_classes})")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        print("Received prediction request")
        if model is None:
            return jsonify({'char': '', 'word': 'Model Error', 'sentence': 'Model not loaded properly'}), 200
        if not request.is_json:
            print("Request is not JSON")
            return jsonify({'char': '', 'word': 'Error', 'sentence': 'Invalid request format'}), 400
        data = request.get_json()
        if not data or 'video' not in data:
            print("Missing video data")
            return jsonify({'char': '', 'word': 'Error', 'sentence': 'No video data provided'}), 400
        try:
            img_data = data['video'].split(',')[1]
            img_data = base64.b64decode(img_data)
            img = Image.open(BytesIO(img_data)).convert('L')
            img = img.resize((64, 64), Image.LANCZOS)
            img = np.array(img) / 255.0
            img = np.expand_dims(img, axis=(0, -1))
            prediction = model.predict(img)
            predicted_index = np.argmax(prediction[0])
            print(f"Prediction input shape: {img.shape}, output: {prediction}")
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