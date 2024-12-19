import streamlit as st
from transformers import pipeline

# Load the sentiment analysis pipeline
classifier = pipeline('sentiment-analysis', model='krishnamishra8848/movie_sentiment_analysis')

# Mapping labels to human-readable sentiments
label_mapping = {"LABEL_0": "negative", "LABEL_1": "positive"}

# Streamlit app title
st.title("Movie Review Sentiment Analysis Transformer")

# Input text from the user
user_input = st.text_area("Enter your movie review:")

# Predict sentiment when the user clicks the button
if st.button("Predict"):
    if user_input.strip():
        # Perform sentiment analysis
        result = classifier(user_input)
        sentiment = label_mapping[result[0]['label']]
        confidence = result[0]['score'] * 100

        # Display results
        if sentiment == "positive":
            st.markdown(f"<h1 style='color: green;'>Sentiment: {sentiment}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: red;'>Sentiment: {sentiment}</h1>", unsafe_allow_html=True)

        # Display confidence in a progress bar
        st.markdown("### Confidence Score")
        st.progress(int(confidence))

        st.write(f"Confidence: {confidence:.2f}%")
    else:
        st.error("Please enter a valid review before predicting.")
