import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import load_model

# Function to preprocess text
def preprocessing(text):
    text = text.strip()
    text = text.lower()
    return text
    
def clean_text(text):
    text = text.strip()                         #Removes Extra White Spaces
    text = text.lower()                         #lowercase
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)     #Removes special characters
    return text

# Load your trained vectorizer
import joblib
vectorizer = joblib.load("fakenews_vectorizer.pkl")
title_vectorizer = joblib.load("title_vectorizer.pkl")
text_vectorizer = joblib.load("text_vectorizer.pkl")

# Streamlit app
st.title("TruthGuard Fake News Detector")

# Text input
article_title = st.text_input("Enter the title of the news article:")
user_input = st.text_area("Enter the text of the news article:", height=400)

chosen_model = st.selectbox("Select the model you want to use:", ["CNN", "Logistic Regression"])

if chosen_model == 'CNN':
    # Load the Keras model (adjust path as needed)
    model = load_model("fakenews_cnn_model.h5")
else:
    # Load the logistic regression model
    model = joblib.load("fakenews_model.pkl")

if st.button("Predict", use_container_width=True):


    if chosen_model == 'CNN':
        # Preprocess the input
        processed_title = clean_text(article_title)
        processed_text = clean_text(user_input)
        # Transform the input using the loaded vectorizer
        vectorized_title = title_vectorizer.transform([processed_title]).toarray()
        vectorized_text = text_vectorizer.transform([processed_text]).toarray()

        # Make a prediction
        prediction_percent = model.predict([vectorized_title, vectorized_text])
        
        prediction = 0 if prediction_percent < 0.5 else 1
    else:
        # Preprocess the input
        processed_input = preprocessing(article_title+' '+user_input)
        # Transform the input using the loaded vectorizer
        vectorized_input = vectorizer.transform([processed_input])
        
        scaler = joblib.load("fakenews_model_scaled.pkl")
        scaled_input = scaler.transform(vectorized_input)
        # Make a prediction
        prediction = model.predict(scaled_input)

    # Display the prediction
    if prediction == 0:
        st.success("The news is predicted to be REAL.")
    else:
        st.error("The news is predicted to be FAKE.")
    if chosen_model == 'CNN':
        prediction_percent = round((prediction_percent[0][0])*100, 2)
        st.write(f"FAKE Probability: {prediction_percent} %") 