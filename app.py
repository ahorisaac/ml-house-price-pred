# -- importing the dependencies
import streamlit as st
st.set_page_config(layout="wide")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

# -- import the boston house price dataset
house_price_dataset = sklearn.datasets.load_boston()

# -- load the dataset to pandas DataFrame
house_price_dataframe = pd.DataFrame(house_price_dataset.data, columns=house_price_dataset.feature_names)

# -- add the target (price) column to the DataFrame
house_price_dataframe['price'] = house_price_dataset.target

# -- split the data and the target
X = house_price_dataframe.drop(['price'], axis=1)
y = house_price_dataframe['price']

# -- split the data into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=2)

# -- model training 
# -- XGBoost Regressor 
model = XGBRegressor()
model.fit(X_train, y_train)

# -- the predict house price function 
def predict_house_price(crim, zn, indus, chas, nox, rm, age, dis, rad, tax, ptratio, b, lstat,):
    input_data = (float(crim), float(zn), float(indus), 
                  float(chas), float(nox), float(rm), 
                  float(age), float(dis), float(rad), 
                  float(tax), float(ptratio), float(b), 
                  float(lstat))
    
    # -- convert the input data into a numpy array 
    input_data_as_np_array = np.asarray(input_data)
    
    # -- reshape the array as we are predicting for only one instance 
    input_data_reshaped = input_data_as_np_array.reshape(1, -1)
    
    prediction = model.predict(input_data_reshaped)
    
    return prediction[0]

# -- ml web app UI
st.columns(5)[2].image("./images/icon.png", width=106)
st.columns(3)[1].subheader("House Price Prediction Form")

with st.form("house_price_pred_form", clear_on_submit=False):
    col1, col2, col3 = st.columns(3)
            
    # -- 1st row
    crim_input = col1.number_input("CRIM", help="per capita crime rate by town")
    
    zn_input = col2.number_input("ZN", help="proportion of residential land zoned for lots  over 25,000 sq.ft")
    
    indus_input = col3.number_input("INDUS", help="proportion of non-retail business acres per town")
    
    # -- 2nd row
    chas_input = col1.number_input("CHAS", help="Charles River dummy variable (1 if tract bounds river; 0 otherwise)")
    
    nox_input = col2.number_input("NOX", help="nitric oxides concentration (parts per 10 million)")

    rm_input = col3.number_input("RM", help="average number of rooms per dwelling")
    
    # -- 3rd row
    age_input = col1.number_input("AGE", help="proportion of owner-occupied units built prior to 1940")
    
    dis_input = col2.number_input("DIS", help="weighted distances to five Boston employment centres")
    
    rad_input = col3.number_input("RAD", help="index of accessibility to radial highways")
    
    # -- 4th row
    tax_input = col1.number_input("TAX", help="full-value property-tax per $10,000")
    
    ptratio_input = col2.number_input("PTRATIO", help="pupil-teacher ratio by town")
    
    b_input = col3.number_input("B", help="1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town")
    
    # -- 5th row
    lstat_input = col1.number_input("LSTAT", help="\% \lower status of the population")
    
    # -- prediction, form submit button 
    submitted = col2.form_submit_button("Predict", type="primary", help="click to predict house price")

    if(submitted):
        col3.title(str(predict_house_price(crim_input, zn_input, indus_input,  
                                           chas_input, nox_input, rm_input,
                                           age_input, dis_input, rad_input, 
                                           tax_input, ptratio_input, b_input,
                                           lstat_input)), anchor="output-str")