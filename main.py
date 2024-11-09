import streamlit as st
import joblib

model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

st.title("Sentiment Analysis")

user_input = st.text_input("Enter text for sentiment analysis:")

if user_input:
    input_tfidf = tfidf.transform([user_input])
    prediction = model.predict(input_tfidf)
    sentiment = prediction[0]  

    st.write("Predicted Sentiment:", sentiment)
