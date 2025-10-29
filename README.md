***LSTM-Based Sentiment Analyzer***

This project is an end-to-end multi-class sentiment analysis application. It uses a Recurrent Neural Network (RNN) with an LSTM layer to classify text into three categories: Positive, Negative, or Neutral.

The model was trained on a dataset of Reddit comments and is deployed as an interactive web app using Streamlit and Hugging Face Spaces.

Live Demo

You can try the app live on Hugging Face Spaces!

Try the Sentiment Analyzer Here : https://huggingface.co/spaces/Rajenderreddy2003/LSTM-Based-Sentiment-Analyzer

**Tech Stack & Workflow**

This project is built in two main phases:

**1. Model Training (Sentiment_Analysis_Using_LSTM.ipynb)**

Data Acquisition: Scraped thousands of comments from the Reddit API using the requests library (from subreddits like r/AskReddit, r/worldnews, etc.).

Data Labeling: Programmatically labeled the entire dataset using VADER (a rule-based sentiment tool) to create the three target classes.

Text Preprocessing: Cleaned and normalized the text data using NLTK and Regex. This pipeline includes:

Lowercasing

Removing URLs, mentions, and special characters

Tokenization (word_tokenize)

Stopword removal

Lemmatization (WordNetLemmatizer)

Data Balancing: Addressed significant class imbalance using RandomUnderSampler (imblearn) to create a balanced dataset (12,685 samples per class).

Model Building: Built and trained a Sequential model in TensorFlow/Keras.

**Model Architecture:**

Embedding Layer: (Vocab Size: 15,000, Max Sequence Length: 60, Output Dim: 128)

SpatialDropout1D: (Rate: 0.2)

LSTM Layer: (Units: 64, Dropout: 0.2, Recurrent Dropout: 0.2)

Dense Output Layer: (Units: 3, Activation: 'softmax')

Results: The trained model achieved ~81% accuracy on the held-out test set.

**2. Web Application & Deployment (app.py)**

**Framework**: Streamlit

**Deployment**: Hugging Face Spaces

**Pipeline**: The app loads the saved sentiment_lstm_model.keras model and tokenizer.pkl file. It uses the exact same NLTK preprocessing pipeline from training to process live, dynamic user input before feeding it to the model for prediction.

**Caching**: Streamlit's @st.cache_resource is used to load the model and tokenizer only once, ensuring fast predictions.

**Project Files**:

app.py: The main Streamlit application file.

Sentiment_Analysis_Using_LSTM.ipynb: The Jupyter Notebook containing all data scraping, cleaning, labeling, and model training.

sentiment_lstm_model.keras: The saved, trained TensorFlow/Keras model.

tokenizer.pkl: The saved Keras Tokenizer (vocabulary).

requirements.txt: A list of all required Python packages for the app.
