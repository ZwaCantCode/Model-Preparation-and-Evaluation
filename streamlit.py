import streamlit as st
import pandas as pd
import pickle 
import os

directory = r'C:\Kuliah\ASDOS ML\Model Preparation and Evaluation'
model=os.path.join(directory,'rf_heartDisease_model.pkl')

if os.path.exists(model):
    try:
        with open(model,'rb') as f:
            loaded_model = pickle.load(f)
            
        rf_model = loaded_model[0]
        
        st.title("Prediksi Potensi Penyakit Jantung")
        st.write("Aplikasi ini berguna untuk membantu menenali potensi penyakit jantung pada manusia berusia 21 - 79 tahun")
        Age = st.slider("Age",21,79)
        sex = st.selectbox("Gender",["F","M"])
        
    except Exception as e :
        st.error("Terjadi kesalahan {E}")
        
else:
    print("File model tidak ditemukan dalam direktori")   