# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 20:44:19 2023

@author: Dell
"""

import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

#create flask app
app = Flask(__name__)

#load pickle model 
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict", methods = ["POST"])
def predict():
    # Retrieve form values
    pregnancies = float(request.form['pregnancies'])
    glucose = float(request.form['glucose'])
    blood_pressure = float(request.form['blood_pressure'])
    skin_thickness = float(request.form['skin_thickness'])
    insulin = float(request.form['insulin'])
    bmi = float(request.form['bmi'])
    diabetes_pedigree_function = float(request.form['diabetes_pedigree_function'])
    age = float(request.form['age'])

    # Create a 2D input array with a single row
    input_array = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])

    # Make the prediction using your model
    prediction = model.predict(input_array)
    
    return render_template("result.html", prediction_text = prediction)
    
   
if __name__ == "__main__":
    app.run(debug=True)
    

        
    

