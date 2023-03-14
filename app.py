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
