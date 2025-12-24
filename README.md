# LSTM-Based Sentiment Analyzer

An **end-to-end multi-class sentiment analysis project** built using a **deep stacked LSTM neural network**, trained on Reddit comments, and deployed as an interactive **Streamlit web application** on **Hugging Face Spaces**.

**Live Deployment:**
[https://huggingface.co/spaces/Rajenderreddy2003/LSTM-Based-Sentiment-Analyzer](https://huggingface.co/spaces/Rajenderreddy2003/LSTM-Based-Sentiment-Analyzer)

---

## Project Overview

This project demonstrates a **complete NLP pipeline**:

* Text tokenization and padding
* Deep LSTM-based sentiment modeling
* Regularization using Dropout and EarlyStopping
* Real-time inference via Streamlit

The system predicts **Positive, Negative, or Neutral** sentiment for user-provided text.

---

## Repository Structure & File Responsibilities

### 1️⃣ `Sentiment_Analysis_Using_LSTM.ipynb`

This notebook contains the **entire training workflow**:

* Loading Reddit comment data
* Train–test split
* Tokenization using `Tokenizer` with OOV handling
* Sequence padding (fixed length = 60)
* Deep stacked LSTM model construction
* Model training with EarlyStopping
* Model evaluation on test data
* Saving trained model and tokenizer for deployment

This file is responsible for **model experimentation and training**.

---

### 2️⃣ `app.py` – Deployment & Inference

The Streamlit application handles **model loading, preprocessing, and prediction**.

Key points:

* Loads the trained `.keras` model and `tokenizer.pkl`
* Uses the **same max sequence length (60)** as training
* Applies identical text preprocessing to ensure consistency
* Predicts sentiment using softmax probabilities

The UI allows users to enter text and instantly view predicted sentiment.

---

### 3️⃣ `reddit_comments_multi_subs.csv`

* Raw dataset containing Reddit comments
* Used as the primary text source for training
* Later labeled into sentiment classes

---

### 4️⃣ `requirements.txt`

Contains all libraries required to run training and deployment, including:
TensorFlow, Keras, NLTK, scikit-learn, imbalanced-learn, Streamlit, and supporting data libraries.

---

## Model Architecture (Verified from Training Code)

The sentiment classifier is a **deep stacked LSTM model** designed to capture complex contextual patterns in text.

### Architecture Used

1. **Embedding Layer**

   * Vocabulary Size: 15,000
   * Embedding Dimension: 64
   * Input Length: 60 tokens

2. **LSTM Layer (1st – Stacked)**

   * Units: 64
   * Dropout: 0.4
   * Recurrent Dropout: 0.2
   * `return_sequences=True`

3. **LSTM Layer (2nd)**

   * Units: 32
   * Dropout: 0.2

4. **Fully Connected Dense Layers**

   * Dense(64, ReLU) → Dropout(0.5)
   * Dense(32, ReLU) → Dropout(0.3)
   * Dense(32, ReLU) → Dropout(0.3)
   * Dense(32, ReLU) → Dropout(0.2)

5. **Output Layer**

   * Dense(3)
   * Activation: Softmax
   * Outputs probabilities for:

     * Positive
     * Negative
     * Neutral

---

## Regularization & Optimization

To prevent overfitting on noisy Reddit text:

* **Dropout** is applied aggressively across LSTM and Dense layers

* **EarlyStopping** is used:

  * Monitors validation loss
  * Patience = 3 epochs
  * Restores best model weights

* **Loss Function:** Sparse Categorical Crossentropy

* **Optimizer:** Adam

---

## Data Processing Pipeline

1. Tokenization with OOV token handling
2. Conversion of text to integer sequences
3. Padding/truncation to 60 tokens (post-padding)
4. Feeding padded sequences into the LSTM model

---

## Model Performance

* Evaluated on unseen test data
* Achieves strong performance despite noisy social media text
* Regularization enables better generalization

---

## Deployment

* Framework: **Streamlit**
* Platform: **Hugging Face Spaces**
* Real-time sentiment prediction via web UI
---

## Local Setup

```bash
# Clone repository
git clone https://github.com/your-username/LSTM-Based-Sentiment-Analyzer.git
cd LSTM-Based-Sentiment-Analyzer

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---
## Author

**Rajender Reddy**
Aspiring Data Scientist | Machine Learning & NLP Enthusiast
