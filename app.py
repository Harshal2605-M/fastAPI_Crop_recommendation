# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 18:01:55 2025

@author: harsh
"""

import joblib
from fastapi import FastAPI
from pydantic import BaseModel
import json
import numpy as np

app = FastAPI()

class Model_Input(BaseModel):
    #ph	rainfall	label	soil_type	water_presence
    N : float
    P : float
    K : float
    temperature : float
    humidity : float
    ph : float
    rainfall : float
    soil_type : str
    water_presence : str
    
recommendation_model = joblib.load(open('crop_model.pkl', 'rb'))
soil_encoder = joblib.load(open("soil_encoder.pkl", "rb"))
water_encoder = joblib.load(open("water_encoder.pkl", "rb"))
crop_encoder = joblib.load(open("crop_encoder.pkl", "rb"))

@app.post('/crop_model')

def crop_reccom(input_parameters : Model_Input) :
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    N = input_dictionary['N']
    P = input_dictionary['P']
    K = input_dictionary['K']
    temperature = input_dictionary['temperature']
    humidity = input_dictionary['humidity']
    ph = input_dictionary['ph']
    rainfall = input_dictionary['rainfall']
    soil_type = input_dictionary['soil_type']
    water_presence = input_dictionary['water_presence']
    
    soil_encoded = soil_encoder.transform([soil_type])[0]
    water_encoded = water_encoder.transform([water_presence])[0]
    
    
    
    input_list = [[N, P, K, temperature, humidity, ph, rainfall, soil_encoded, water_encoded]]
    
    probabilities = recommendation_model.predict_proba(input_list)[0]

   # Get indices of top 3 probabilities
    top3_indices = np.argsort(probabilities)[-3:][::-1]

   # Decode crop names and probabilities
    top3_crops = [
        {
           "crop": crop_encoder.inverse_transform([i])[0],
           "probability": round(float(probabilities[i] * 100), 2)
        }
        for i in top3_indices
    ]

    return {
       
       "top_3_recommendations": top3_crops
   }
    
    
    
    
    
    
    
    
    
    