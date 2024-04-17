# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:54:36 2024

@author: Siddhi
"""

import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

encoder=LabelEncoder()
# loading the saved model
#loaded_model = pickle.load(open('D:/Work/Machine Learning/Deploying Machine Learning model/trained_model.sav', 'rb'))
#C:\Users\askpr\ExR\Project_Deployment\Weekday
loaded_model = pickle.load(open('C:/Users/Siddhi/datascience/excelr project/random_regressor.sav', 'rb'))
# creating a function for Prediction

def ran_regressor(inputdata):
    
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(inputdata)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    return prediction

    
  
    
  
def main():
    
    
    # giving a title
    st.title('Co2 Emission Web App')
    
    
    # getting the input data from the user
    
    Make_Type=st.text_input('Make Type: #luxury,premium,General,Sports')
    Make_Type=encoder.fit_transform([Make_Type])[0]
    Vehicle_Class_Type=st.text_input('Type of vehicle:#Hatchback,SUV,Sedan,Truck')
    Vehicle_Class_Type=encoder.fit_transform([Vehicle_Class_Type])[0]
    engine_size = st.text_input('Engine size')
    cylinders= st.text_input('cylinder')
    transmission= st.text_input('Transmission: #automatic With shift select,manual,automated manual,continuously variable,automatic')
    transmission=encoder.fit_transform([transmission])[0]
    fuel_type = st.text_input('Fuel Type:# Premium Gasoline,diesel,regular gasoline,ethanol,natural gas')
    fuel_type=encoder.fit_transform([fuel_type])[0]
    fuel_consumption_city= st.text_input('Fuel consumption in city')
    fuel_consumption_hwy= st.text_input('fuel consumption on highway')
    fuel_consumption_comb_100km = st.text_input('Fuel consumption per 100 km')
    fuel_consumption_comb_mpg= st.text_input('Consumption in mpg')
    
    
    
    # code for Prediction
    result= ''
    
    # creating a button for Prediction
    
    if st.button('Test Result'):
        result= ran_regressor([Make_Type,Vehicle_Class_Type,engine_size , cylinders,transmission,fuel_type,fuel_consumption_city,fuel_consumption_hwy,fuel_consumption_comb_100km,fuel_consumption_comb_mpg])
        
        
    st.success(result)
    
    
    
    
    
if __name__ == '__main__':
    main()