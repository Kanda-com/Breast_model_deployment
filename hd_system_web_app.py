# -*- coding: utf-8 -*-
"""
Created on Thu May 29 13:51:14 2025

@author: user
"""

import numpy as np
import pickle
import streamlit as st
import pandas as pd
import requests
url = 'https://github.com/Kanda-com/Breast_model_deployment/blob/main/final_model.sav'
loaded_model = requests.get(url)

with open('final_model.sav', 'wb') as f:
    pickle.dump(loaded_model, f)
with open('final_model.sav', 'rb') as f:
    loaded_model = pickle.load(f)
    
def Breast_disease_prediction(input_data):
    input_data_as_numpy_array=np.array(input_data)
    input_data.reshaped= input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data.reshaped)

    if prediction [0] == 0:
        return "Cancerous patient"
    else:
        return "Noncancerous patient"
def main():
    st.title("Breast Cancer ML Prediction model")
    mean_radius = st.text_imput("Enter the mean radius ")
    mean_texture = st.text_input("Enter the mean texture")
    mean_perimeter = st.text_input("mean perimeter")
    mean_area = st.text_input("mean area")
    mean_smoothness = st.text_input("mean smoothness")
    
    mean_radius = pd.to_numeric(mean_radius, errors='coerce')
    mean_texture = pd.to_numeric(mean_texture, errors='coerce')
    mean_perimeter = pd.to_numeric(mean_perimeter, errors='coerce')
    mean_area =pd.to_numeric(mean_area, errors='coerce')
    mean_smoothness =pd.to_numeric(mean_smoothness, errors='coerce')
    
    diagnosis = ''
    
    if  st.button("PREDICT"):
        diagnosis = Breast_disease_prediction([mean_radius,mean_texture,mean_perimeter,mean_area,mean_smoothness])
        st.success(diagnosis)

    if __name__ == '__main__':
        main()
