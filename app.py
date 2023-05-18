from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import json
import pickle

app = Flask(__name__)

# Load model
ml_model = pickle.load(open('lr_model.pkl', 'rb'))
scaler = pickle.load(open('scaling.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # print(type(data))
    # print(data.values())
    # print(np.array(list(data.values())))
    new_data = np.array(list(data.values()))
    new_data = new_data.reshape(1,-1)
    transformed_data = scaler.transform(new_data)
    print(new_data)
    print(transformed_data)
    predicted = ml_model.predict(transformed_data)
    print(predicted)

    predicted_class = None
    if predicted == 0:
        predicted_class = 'Benign'
    if predicted == 1:
        predicted_class = 'Malignant'

    return predicted_class


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    new_data = np.array(data)
    new_data = new_data.reshape(1, -1)
    transformed_data = scaler.transform(new_data)
    predicted = ml_model.predict(transformed_data)
    predicted_class = None
    if predicted == 0:
        predicted_class = 'Benign'
    if predicted == 1:
        predicted_class = 'Malignant'
    return render_template("home.html", prediction_text = "Your cancer class is {}".format(predicted_class))

if __name__=="__main__":
    app.run(debug=True)