import streamlit as st
import pandas as pd
import pickle 
import os
from sklearn.preprocessing import OneHotEncoder 

encoder = OneHotEncoder(sparse_output=False)
model='rf_heartDisease_model.pkl'

with open(model,'rb') as f:
    loaded_model = pickle.load(f)
    
rf_model = loaded_model[0]

st.title("Prediksi Potensi Penyakit Jantung")
st.write("Aplikasi ini berguna untuk membantu menenali potensi penyakit jantung pada manusia berusia 21 - 79 tahun")
Age = st.slider("Age",21,79)
sex = st.selectbox("Gender",["F","M"])
ChestPainType = st.selectbox("Chest Pain Type",["ATA","ASY"])    
RestingBP = st.number_input("Resting Blood Preassure",0,200)
Cholesterol = st.number_input("Cholesterol", 0,603)
FastingBS = st.selectbox("Fasting BS",["1","0"])
RestingECG = st.selectbox("Resting ECG",["Normal","ST","LVH"])
MaxHR = st.number_input("Max Heart Rate",60,202)
ExcerciseAngina = st.radio("Excercise Angina",["Y","N"])
ST_Slope=st.selectbox("ST_Slope",["Up","Flat","Down"])
Old_peak = st.slider("Old Peak",-3.0,7.0,step=0.1)

#ChestPain = pd.DataFrame({'ChestPainType'}[ChestPainType])

df_result=pd.DataFrame({'ChestPainType':[ChestPainType]})
df_result['RestingECG'] = RestingECG
df_result['ExcerciseAngina'] = ExcerciseAngina
df_result['ST_Slope']=ST_Slope

input_encoded = encoder.fit_transform(df_result)
input_onehot_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())

input_data = [input_onehot_df['onehotencoder__ChestPainType_ASY', 'onehotencoder__ChestPainType_ATA',
       'onehotencoder__ExerciseAngina_N', 'onehotencoder__ExerciseAngina_Y',
       'onehotencoder__ST_Slope_Flat', 'onehotencoder__ST_Slope_Up'],MaxHR,Old_peak]

if st.button("Prediksi"):
    rf_model_prediction = rf_model.predict(input_data)
    outcome= {0:'Tidak Berpotensi sakit jantung', 1:'Berpotensi sakit jantung'}
    st.write(f"Orang tersebut diprediksi **{outcome[rf_model_prediction[0]]}** oleh RF")
