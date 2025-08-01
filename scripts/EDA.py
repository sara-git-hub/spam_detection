import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# Fonction pour l'analyse exploratoire
def analyse_exploratoire(data):
    print(f"notre data contient {data.shape[0]} lignes et {data.shape[1]} colonnes")
    print()
    print("Data info :")
    print(data.info())
    print()
    print("Statistiques:")
    print(data.describe())
    print()
    print("Le nombres de données manquantes par colonnes:")
    print(data.isna().sum())
    print()
    print(f"nombres de doublons {data.duplicated().sum()}")

# Fonction pour afficher les graphiques
def graphique(data,label):

    plt.figure(figsize=(8, 4))
    plt.title("Repartition Spam/Ham")
    counts = data[label].value_counts().sort_index()
    ax = sns.barplot(x=counts.values, y=counts.index, orient='h', width=0.3)

    for i, value in enumerate(counts.values):
        ax.text(value, i, f' {value}', ha='left', va='center', fontsize=10)

    plt.yticks([0, 1], ['Ham', 'Spam'])
    plt.ylim(-0.6, 1.6)

    plt.xlabel('')
    ax.set_xticks([])
    ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    ax.tick_params(left=False)

    plt.tight_layout()
    plt.savefig(r'C:\Users\lenovo\Documents\Spam_Detection\plots\Nb_spam_ham.png')
    plt.show()

#Nettoyage des données
def clean_data(data):
    df=data.copy()
    df=df.dropna(subset='text')
    df = df[df['text'].str.strip() != '']
    return df

# Créer le nuage de mots
def nuage_mots(data,text):

    df=data.copy()
    
    spam_texts = df[df['label'] == 1][text]
    ham_texts = df[df['label'] == 0][text]

    spam_text = ' '.join(spam_texts)
    ham_text = ' '.join(ham_texts)
    
    wc_spam = WordCloud(width=800, height=400, background_color='white').generate(spam_text)
    wc_ham = WordCloud(width=800, height=400, background_color='white').generate(ham_text)

    # Afficher les deux nuages de mots côte à côte
    plt.figure(figsize=(24,12))

    plt.subplot(1,2,1)
    plt.imshow(wc_spam, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots - Spam')

    plt.subplot(1,2,2)
    plt.imshow(wc_ham, interpolation='bilinear')
    plt.axis('off')
    plt.title('Nuage de mots - Ham')

    plt.savefig(r'C:\Users\lenovo\Documents\Spam_Detection\plots\nuage_de_mots.png', dpi=300, bbox_inches='tight', quality=100)

    plt.show()


