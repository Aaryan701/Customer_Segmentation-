import streamlit as st
import pandas as pd
import numpy as np
import joblib

kmeans_model=joblib.load("Customer_Segmentation_KMeans_Model.pkl")
scaler_model=joblib.load("Customer_Segmentation_Scaler.pkl")

st.title("Customer Segmentation App")
st.write("Enter the customer details to predict the segment")

age=st.number_input("Age",min_value=18,max_value=100,value=30)
income=st.number_input("Income",min_value=0,value=50000)
total_spending=st.number_input("spending_score",min_value=0,value=50)
Customer_Since=st.number_input("Customer Since (years)",min_value=0,value=5)
recency=st.number_input("Recency (days since known purchase)",min_value=180,value=300)
num_web_purchases=st.number_input("Number of Web Visits",min_value=0,value=10)
num_stores_purchases=st.number_input("Num of Stores Purchased From",min_value=0,value=5)



input_data=pd.DataFrame({
    "Age":[age],
    "Income":[income],
    "spending_score":[total_spending],
    "Customer_Since":[Customer_Since],
    "Recency":[recency],
    "NumWebPurchases":[num_web_purchases],
    "NumStorePurchases":[num_stores_purchases]
    })

scaled_input=scaler_model.transform(input_data)

if st.button("Predict Segment"):
    cluster=kmeans_model.predict(scaled_input)
    st.success(f"The predicted customer segment is: Cluster {cluster}")
