import sys
import os
import glob
import re
import numpy as np
import json

# Keras
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

MODEL_PATH = 'models/Clasificador_VGG16_20196351.h5'

# Load trained model and configurations
model = load_model(MODEL_PATH)
model._make_predict_function()
model_config = json.load(open('models/config.json'))
print('Model loaded. Start serving...')

# Define labels of the trained model
LABELS = np.array([['cardboard','glass','metal','paper','plastic']])


# Define prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(model_config['img_height'],model_config['img_width']))

    # Preprocessing the image
    x = image.img_to_array(img)
    #x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x /= 255.

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    #x = preprocess_input(x)

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        pred = model_predict(file_path, model)

        # Decode prediction
        label = LABELS[pred==np.max(pred)][0]
        result = label + ": " + str(pred[pred==np.max(pred)][0])

        # Return prediction if probability higher than threshold
        if np.max(pred) < model_config['threshold']:
        	result = 'No puedo identificar correctamente :('

        # Delete saved image
        os.remove(file_path)

        return result
    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()
