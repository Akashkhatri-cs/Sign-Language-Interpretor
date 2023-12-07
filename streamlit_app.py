# app.py

import streamlit as st
import requests

st.title('Sign Language Recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:
    if st.button('Predict'):
        files = {'file': uploaded_file.getvalue()}
        response = requests.post('http://74.12.105.219:8090/predict', files=files)
        result = response.json()
        st.write('Predicted Letter: ', result['predicted_letter'])
