# -*- coding: utf-8 -*-

#Import Libraries
from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
app = Flask(__name__)

############################################################################
df = pd.read_csv("clean_data_1.csv")
X = df.drop('price', axis=1).values
y= df["price"].values

# data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 51)
# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
#X_train= sc.transform(X_train)
#X_test = sc.transform(X_test)
 
model = joblib.load("bengaluru_house_price_model.pkl") 
def predict_house_price(area_type,location,bath,balcony,total_sqft_int,bhk,availability):
 
    x =np.zeros(7) # create zero numpy array
    
    def choose_area(area_type):
        if area_type == 'Super built-up Area': 
            return 0
        elif area_type == 'Built-up Area': 
            return 1
        elif area_type == 'Plot Area': 
            return 2
        else:
            return 3
        
    def loc(location):
        if location == 'Whitefield': 
            return 0
        elif location == 'Sarjapur Road': 
            return 1
        elif location == 'Electronic City': 
            return 2
        elif location == 'Kanakpura Road': 
            return 3
        else:
            return 4
       
        
    def availabe(availability):  
        if availability == 'Ready To Move':
            return 0
        elif availability == 'Immediate Possession':
            return 1
        else:
            return 2
  
    # adding feature's value according to their column index
    x[0] = choose_area(area_type)
    x[1] = loc(location)   
    x[2] = bath
    x[3] = balcony
    x[4] = total_sqft_int
    x[5] = bhk
    x[6] = availabe(availability)
    
    x= sc.transform([x])
    return model.predict(x) # return the predicted value
    
##############################################################################

@app.route('/')
def home():
    return render_template('index.html')
 
# get user input -- predict the output -- return to user
@app.route('/predict',methods=['POST', 'GET'])
def predict():
    
    #return "hello"   
     
    #take data from the form and store in each feature    
    input_features = [x for x in request.form.values()]
    
    area_type = input_features[0]
    location = input_features[1]
    bath = input_features[2]
    balcony = input_features[3]
    total_sqft_int = input_features[4]
    bhk = input_features[5]
    availability = input_features[6]
        
    # predict the price of house by calling model.py
    predicted_price = predict_house_price(area_type,location,bath,balcony,total_sqft_int,bhk,availability)       
    #return "world"
 
    # render the html page and show the output
    return render_template('index.html', prediction_text='Predicted House Price is {} lakhs'.format(predicted_price))
 
     
if __name__ == "__main__":
    app.run()
    