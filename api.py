from waitress import serve
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the trained model
model = load_model(os.path.join("mount", "mech2.h5"))

@app.route("/")
def home():
    return "Use the /classify POST route with an image file named 'image'"

@app.route('/classify', methods=['POST'])
def predict():
    file = request.files['image']

    filename = secure_filename(file.filename)
    file.save(filename)


    img = image.load_img(filename,target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255
    x = np.expand_dims(x,axis=0)
    predicted = model.predict(x)
    a = np.argmax(predicted)
    confidence = np.max(predicted)
    classes = ["Bolt", "Locating Pin", "Nut", "Washer"]

    return jsonify({"category": classes[a], "confidence": float(confidence)})

if __name__ == "__main__":
    HOST_IP = "0.0.0.0"
    HOST_PORT = 8000
    print(f"Attempting to serve on {HOST_IP}:{HOST_PORT}")
    serve(app, host=HOST_IP, port=HOST_PORT)
    
