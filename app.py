import streamlit as st
import pickle
import re
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os # Added to check if files exist

# --- NLTK Resource Download ---
def download_nltk_resources():
    """Checks and downloads required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('tokenizers/punkt_tab') 
        nltk.data.find('corpora/stopwords')
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('punkt')
        nltk.download('punkt_tab')
        nltk.download('stopwords')
        nltk.download('wordnet')

download_nltk_resources()

# --- Define Model and Tokenizer Paths ---
MODEL_PATH = 'sentiment_lstm_model.keras'
TOKENIZER_PATH = 'tokenizer.pkl'

# --- Load Saved Model and Tokenizer (Cached) ---
@st.cache_resource
def load_assets():
    """
    Loads and caches the trained model and tokenizer.
    This function runs only once when the app starts.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}")
        return None, None
    if not os.path.exists(TOKENIZER_PATH):
        st.error(f"Tokenizer file not found at {TOKENIZER_PATH}")
        return None, None
        
    print("Loading model and tokenizer...")
    try:
        model = load_model(MODEL_PATH)
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
        print("Loading complete.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading assets: {e}")
        return None, None

model, tokenizer = load_assets()

# --- Preprocessing Functions & Assets ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def removing(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = re.sub(r'\@\w+|\#','', text)  # Remove @mentions and hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters & numbers
    text = re.sub(r'\n','',text)
    text = re.sub(r'\xa0','',text)
    return text

def preprocess_text(text):
    """
    The complete preprocessing pipeline for a single raw text string.
    """
    text = text.lower()
    text = removing(text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens


# --- Prediction Pipeline ---
SENTIMENT_MAP = {1: "Positive", 0: "Negative", 2: "Neutral"}
MAXLEN = 60 # From your notebook

def predict_sentiment(raw_text, model, tokenizer):
    """
    Takes a raw text string and returns its predicted sentiment.
    """
    if not raw_text or model is None or tokenizer is None:
        return None
        
    try:
        # 1. Preprocess the raw text
        clean_tokens = preprocess_text(raw_text)
        
        # 2. Convert tokens to integer sequence
        seq = tokenizer.texts_to_sequences([clean_tokens])
        
        # 3. Pad the sequence
        padded_seq = pad_sequences(seq, maxlen=MAXLEN, dtype='int32', padding='post', truncating='post', value=0)
        
        # 4. Make prediction
        prediction = model.predict(padded_seq)
        
        # 5. Get the class with the highest probability
        predicted_class_index = np.argmax(prediction[0])
        
        # 6. Map index back to sentiment label
        sentiment = SENTIMENT_MAP[predicted_class_index]
        
        return sentiment
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

# --- Streamlit UI ---

# Main Page
st.title("LSTM-Based Sentiment Analyzer")
st.write("Enter a comment below to analyze its sentiment (Positive, Negative, or Neutral).")


st.text_area("Enter your comment:", height=150, key="text_input_area")

# Analyze button
if st.button("Analyze Sentiment"):
    comment_to_analyze = st.session_state.text_input_area
    
    if comment_to_analyze:        
        # Run the pipeline
        with st.spinner("Analyzing..."):
            sentiment = predict_sentiment(comment_to_analyze, model, tokenizer)
        
        if sentiment:
            st.subheader("Predicted Sentiment")
            st.write(f"**{sentiment}**")
        else:
            st.warning("Could not analyze sentiment. Model assets might be missing or an error occurred.")
    else:
        st.warning("Please enter a comment to analyze.")

# Add a "Clear" button
if st.button("Clear"):
    # 4. The clear button now just sets the text_input_area's state to empty
    st.session_state.text_input_area = ""
    st.rerun()

# --- Sidebar Information ---
st.sidebar.title("About the Model")
st.sidebar.markdown("""
This model is a **Recurrent Neural Network (RNN)** designed for multi-class sentiment analysis. It was trained on a dataset of Reddit comments.
""")

st.sidebar.subheader("Project Highlights")
st.sidebar.markdown(f"""
- **Data Source:** Reddit API (scraped from subreddits like r/AskReddit, r/worldnews, etc.)
- **Data Labeling:** Programmatically labeled using **VADER** (a rule-based sentiment tool).
- **Data Balancing:** Handled class imbalance using **RandomUnderSampler** to create a balanced dataset (12,685 samples per class).
- **Test Accuracy:** Achieved **~81% accuracy** on the held-out test set.
""")

st.sidebar.subheader("Model Architecture")
st.sidebar.markdown(f"""
A Sequential Keras model:
1.  **Embedding Layer:** vocab_size=15000, output_dim=128, input_length=60
2.  **SpatialDropout1D:** rate=0.2
3.  **LSTM Layer:** units=64, dropout=0.2, recurrent_dropout=0.2
4.  **Dense Layer (Output):** units=3, activation='softmax'
""")

st.sidebar.subheader("Preprocessing Pipeline")
st.sidebar.markdown("""
Raw text is cleaned before prediction:
1.  Lowercase text
2.  Remove URLs, mentions, and special characters (Regex)
3.  Tokenize text (NLTK word_tokenize)
4.  Remove Stopwords (NLTK)
5.  Lemmatize tokens (NLTK WordNetLemmatizer)
""")

