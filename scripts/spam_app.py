import streamlit as st
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import joblib

st.set_page_config(layout="wide")

nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

data=pd.read_csv(r"C:\Users\lenovo\Documents\Spam_Detection\data\DataSet_Emails.csv",index_col=0)
data=data.dropna(subset='text')

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


st.sidebar.title("Spam detection")
app_select=st.sidebar.selectbox("menu",["Analyse exploratoire","Résultats","Prediction"])

if app_select=="Analyse exploratoire":
    st.title("🔎 Analyse exploratoire: ")

    # Affichage des données tabulaires
    st.subheader("Aperçu des données brutes:")
    st.dataframe(data)

    # Affichage des principales statistiques
    st.subheader("Statistiques descriptives:")
    st.write(data.describe())


    st.subheader("Graphiques:")
    st.image(r'C:\Users\lenovo\Documents\Spam_Detection\plots\Nb_spam_ham.png',width=600)

    st.subheader("Nuages de mots pour les spam et les ham:")
    st.image(r'C:\Users\lenovo\Documents\Spam_Detection\plots\nuage_de_mots.png',use_container_width=True)

elif app_select=="Résultats":
    st.title("🚀 Résultats obtenus: ")

    st.markdown("""
    ### Performance des Modèles
    | Modèle                 | CV F1 (moyen) | F1 test|
    |----------------------  |-------        |------- |
    | **SVC**                | 0.988         | 0.988  |
    | Decision Tree          | 0.954         | 0.957  |
    | Multinomial            | 0.980         | 0.980  |
    """)
    
    st.write("""
    Le modèle retenu est le Support Vector Class avec un **F1 score** de **0.988**
    """)

    st.subheader("Graphiques des résultats")
    st.image(r'C:\Users\lenovo\Documents\Spam_Detection\plots\Matrice_de_confusion.png')
    st.image(r'C:\Users\lenovo\Documents\Spam_Detection\plots\Courbe_apprentissage_SVC.png')


elif app_select== "Prediction" :
    st.title("Détection de Spam ")
    st.image(r'C:\Users\lenovo\Documents\Spam_Detection\plots\spam.jpg',width=300)

    @st.cache_resource()
    def load_model():
        return joblib.load(r'C:\Users\lenovo\Documents\Spam_Detection\model\model.pkl')

    model = load_model()

    

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