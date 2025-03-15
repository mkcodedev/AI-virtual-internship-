import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas as pd
import os
from flask import Flask, request, jsonify, render_template, url_for, send_from_directory

app = Flask(__name__)

# Load the model and preprocessing files
model = pickle.load(open('traffic_volume.pkl', 'rb'))
scaler = pickle.load(open('scale.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))
imputer = pickle.load(open('imputer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # rendering the home page

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        # reading the inputs given by the user
        input_feature = [float(x) for x in request.form.values()]
        features = [np.array(input_feature)]
        names = ['holiday', 'temp', 'rain', 'snow', 'weather', 'year', 'month', 'day',
                'hours', 'minutes', 'seconds']
        
        # Create a DataFrame with the input values
        data = pd.DataFrame(features, columns=names)
        
        # Transform the data and make predictions
        data_scaled = scaler.transform(data)
        prediction = model.predict(data_scaled)
        
        print(prediction)
        
        # Format the prediction with commas for thousands
        formatted_prediction = "{:,.0f}".format(prediction[0])
        
        # Create a string representation of the input parameters
        input_params = ", ".join([f"{name}: {value}" for name, value in zip(names, input_feature)])
        
        # Determine if traffic volume is high or low (threshold at 3000 vehicles)
        if prediction[0] > 3000:
            return render_template("chance.html", 
                                  prediction_value=formatted_prediction,
                                  input_parameters=input_params)
        else:
            return render_template("noChance.html", 
                                  prediction_value=formatted_prediction,
                                  input_parameters=input_params)
    
    return render_template("index.html")

@app.route('/visualize')
def visualize():
    # Check if visualization files exist
    feature_importance_exists = os.path.exists('static/feature_importance.png')
    actual_vs_predicted_exists = os.path.exists('static/actual_vs_predicted.png')
    
    return render_template('visualize.html', 
                          feature_importance_exists=feature_importance_exists,
                          actual_vs_predicted_exists=actual_vs_predicted_exists)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

if __name__ == "__main__":
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False) 