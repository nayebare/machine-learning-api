
# machine-api.py
# (c) 2019 fieldcloud SAS. all rights reserved
# VERSION:
# DATE: TOD
# AUTHOR: micheal.nayebare@gmail.com
# back-end api file for machine-learning-model


import flask
import io
import string
import time
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, jsonify, request 
from flask_cors import CORS, cross_origin

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('resnet50_food_model')

def prepare_image(img):
    img = Image.open(io.BytesIO(img))
    img = img.resize((224, 224))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

def predict_result(img):
    return 1 if model.predict(img)[0][0] > 0.5 else 0

@app.route("/", methods=['GET'])
def index():
    return("Machine Learning Inference") 
    
#send sms
@app.route("/predict", methods=['POST'])
def infer_image():
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')

    if not file:
        return

    # Read the image
    img_bytes = file.read()

    # Prepare the image
    img = prepare_image(img_bytes)

    # Return on a JSON format
    return jsonify(prediction=predict_result(img))


if __name__ == '__main__':
    app.run(port=5000, host='0.0.0.0', debug=True)
