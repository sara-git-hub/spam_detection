import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text_func(text):
    text_1=str(text)
    text_1=text_1.lower()

    # Tokenisation (décompose le texte en tokens)
    tokens = word_tokenize(text_1)
    
    # Supprimer la ponctuation et les caractères spéciaux
    tokens = [re.sub(f'[{re.escape(string.punctuation)}]', '', token) for token in tokens]
    tokens = [token for token in tokens if token]  # supprimer les tokens vides après suppression ponctuation

    if stop_words is None:
        raise ValueError
    
    # Supprimer les stop_words
    tokens = [token for token in tokens if token not in stop_words]
    
    # Appliquer le stemming
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    #Join tolkens
    processed_text_str = ' '.join(stemmed_tokens)
    
    return processed_text_str

# ---------------------
# Chargement du pipeline
# ---------------------
@st.cache_resource()
def load_model():
    return joblib.load("model.pkl")

model = load_model()

st.title("Détection de Spam par Machine Learning")

st.write("""
Ce mini outil permet de tester votre message et de savoir s'il est classé comme **spam** ou **ham**.
""")

# Entrée utilisateur
message = st.text_area("Votre message :", "")

if st.button("Analyser"):
    # Application du prétraitement sur le texte utilisateur
    processed = preprocess_text_func(message)
    prediction = model.predict([processed])
    probabilities = model.predict_proba([processed])
    proba_pred = probabilities[0][prediction[0]]
    label = "Spam" if prediction[0] == 1 else "Ham"
    st.success(f"Résultat : **{label}**")
    st.write(f"Probabilité associée : {proba_pred*100:.4f}%")