# Système Intelligent de Filtrage des Emails pour la Sécurité des Communications

## 📌 Description du Projet
Ce projet vise à développer un système de classification automatique des emails pour identifier les messages spam et non-spam :
- **Base de données d'e-mails**
- **Modèle avancé** : Support Vector Class avec optimisation des hyperparamètres

**Approche technique** :
- Pretraitement des données
- Comparaison de plusieurs modèles (Decision Tree classifier, Naïve Bayes classifier, Support Vector Class)
- Optimisation via RandomizedSearch et CV

## 📂 Structure des Fichiers

├── data/

│ └── dataSet_Emails.csv # Données brutes

├── scripts/

│ ├── EDA.py # Fpnctions pour l'analyse exploratoire

│ ├── main.ipynb # Fichier principal regroupant l'analyse exploratoire, les modèles et les résultats

│ ├── spam_app.py # Fichier de création d'une application streamlit 

│ └── test.ipynb # Test sur les data pour évaluer le modèle

├── models/

│ └── model.pkl # Modèle sauvegardé (SVC)

└── README.md

  - Jira: 
 
 ## 🛠️ Technologies Utilisées
- **Langage**: Python
- **Librairies principales**:
  - `pandas`, `numpy` (traitement des données)
  - `scikit-learn` (modèles de ML)
  - `matplotlib`, `seaborn` (visualisation)
  - `nltk`, `wordcloud` (traitement et visualisation nlp)
  - `streamlit` (création d'application)


  📝 Méthodologie
- **Prétraitement** 
  - Nettoyage des valeurs manquantes


- **Modeles et Optimisation**
  - Validation croisée (5 folds)
  - Recherche d'hyperparamètres à l'aide de RandomizedSearch et CV
  - Choix du meilleur modèle (SVC)


  ## 🚀 Résultats Clés
### Performance des Modèles
| Modèle                 | CV F1 (moyen) | F1 test|
|----------------------  |-------        |------- |
| **SVC**                | 0.988         | 0.988  |
| Decision Tree          | 0.954         | 0.957  |
| Multinomial            | 0.980         | 0.980  |