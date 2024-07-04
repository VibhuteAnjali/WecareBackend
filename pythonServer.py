
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from test_model import predict_health_status
from PIL import Image
import numpy as np
app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['avatar']
        image = process_image(file)
        prediction = predict_health_status(image)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)})

def process_image(file):
    try:
        # Read image data from the file object
        img = Image.open(file.stream)
        img = np.array(img)

        # Convert the image to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Verify image shape
        print("Original Image Shape:", img.shape)

        # Resize the image
        img = cv2.resize(img, (200, 200))

        return img
    except Exception as e:
        print("Error processing image:", e)
        return None



def start():
    return "Server Started"
if __name__ == '__main__':
    app.run(debug=True)
