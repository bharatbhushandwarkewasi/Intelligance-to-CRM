# -*- coding: utf-8 -*-
"""
Created on  sep 22 00:58:31 2022

@author: bhushan
"""

import flask
from flask import Flask, request , jsonify, render_template, url_for, render_template
import jinja2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle


# loading the model from disk
model = keras.models.load_model("CRM_LSTM_model.h5")
CRM_tokenizer = pickle.load(open("CRM_model_tokenizer.pkl", "rb" ))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index_2.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
            message  = str(request.form['message'])
            data = [message]
            seq = CRM_tokenizer.texts_to_sequences(data)
            padded = pad_sequences(seq, maxlen=100)
            
            results = model.predict(padded)
            my_prediction = np.argmax(results)

            if my_prediction == int(1):
                output = 1
            else:
                output = 0
                
    return render_template('after.html',data = output)

if __name__ == "__main__":
    app.run(host = '0.0.0.0',port= 8080)

