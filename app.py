from __future__ import division, print_function
# coding=utf-8
import sys
import logging
import os
import glob
import re
import numpy as np
import tensorflow as tf
# Keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)
# Configure logging
logging.basicConfig(level=logging.DEBUG)  # Set to DEBUG level
logger = app.logger

# Model saved with Keras model.save()
MODEL_PATH =r'C:\Users\HP\model_inception.h5'

# Load your trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    ## Scaling
    x=x/255
    x=np.expand_dims(x, axis=0)
   # x = preprocess_input(x)
    preds=model.predict(x)
    preds=np.argmax(preds, axis=1)
    if preds==0:
        preds="The leaf is diseased cotton leaf"
    elif preds==1:
        preds="The leaf is diseased cotton plant"
    elif preds==2:
        preds="The leaf is fresh cotton leaf"
    else:
        preds="The leaf is fresh cotton plant"
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    try:
        if request.method == 'POST':
            # Log the request method
            logger.debug("POST request received") 
            # Get the file from post request
            f = request.files['file']
            # Log file details
            logger.debug(f"File received: {f.filename}")
            # Save the file to ./uploads
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            # Ensure the uploads directory exists
            os.makedirs(os.path.join(basepath, 'uploads'), exist_ok=True)
            f.save(file_path)
            # Log the file path
            logger.debug(f"File saved to: {file_path}")
            # Make prediction
            preds = model_predict(file_path, model)
            # Log the prediction result
            logger.debug(f"Prediction result: {preds}")
            return preds
        else:
            logger.debug("GET request received")
    except Exception as e:
        # Log any exception that occurs
        logger.error(f"Error occurred: {str(e)}", exc_info=True)
        return str(e), 500
if __name__ == '__main__':
    app.run(debug=True)
if __name__ == '__main__':
    app.run(port=5001,debug=True)