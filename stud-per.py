import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://karanAbhinav:Abhi1234@cluster0.hadebom.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db = client['student']
collection = db["student_pred"]

def load_model():
    with open("student_lr_final_model.pkl", 'rb') as file :
        model,scaler,le = pickle.load(file)
    return model,scaler,le

def preprocessing_input_data(data, scaler, le):
    data["Extracurricular Activities"] = le.transform([data["Extracurricular Activities"]])[0]
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

def predict_data(data):
    model,scaler,le = load_model()
    processed_data = preprocessing_input_data(data, scaler, le)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Student Performance Prediction")
    st.write("Enter Your Data to Get a Prediction For Your Performance")

    Hours_Studied = st.number_input("Hours Studied",min_value = 1, max_value = 10, value = 5)
    Previous_Scores = st.number_input("Previous Scores",min_value = 40, max_value = 100, value = 70)
    Extracurricular = st.selectbox("Extracurricular Activities", ["Yes","No"])
    Sleep_Hours = st.number_input("Sleep Hours",min_value = 4, max_value = 10, value = 7)
    Sample_QP_Solved = st.number_input("Sample Question Papers Practiced",min_value = 0, max_value = 10, value = 5)

    if st.button("Predict_Your_Score"):
        user_data = {
            "Hours Studied":Hours_Studied,
            "Previous Scores":Previous_Scores,
            "Extracurricular Activities":Extracurricular,
            "Sleep Hours":Sleep_Hours,
            "Sample Question Papers Practiced":Sample_QP_Solved
        }
        prediction = predict_data(user_data)
        st.success(f"Your Prediction result is {prediction}")
        user_data['prediction'] = round(float(prediction[0]),2)
        user_data = {key: int(value) if isinstance(value, np.integer) else float(value) if isinstance(value, np.floating) else value for key, value in user_data.items()}
        collection.insert_one(user_data)
        

if __name__ == "__main__":
    main()

