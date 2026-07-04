# Projet 11 - Analyse des Discours Politiques

**Master 1 IA & Data Science — INSI Madagascar**

Analyse de sentiment sur des tweets politiques. Modèle entraîné sur 1,695,833 tweets politiques issus de 3 datasets Kaggle.

---

## Résultats

| Modele | F1 Test | Accuracy Test | CV F1 |
|---|---|---|---|
| **LinearSVC** | **0.8902** | **0.8904** | **0.8835** |
| Logistic Regression | 0.8821 | 0.8824 | 0.8739 |
| Naive Bayes | 0.7224 | 0.7226 | 0.7214 |
| Random Forest | 0.6994 | 0.7021 | 0.6962 |
| Gradient Boosting | 0.6168 | 0.6171 | 0.6153 |

**Meilleur modele : LinearSVC — F1 = 0.8902**

---

## Datasets

| Dataset | Source Kaggle | Tweets | Domaine |
|---|---|---|---|
| DS1 | Global Political Tweets (kaushiksuresh147) | 238,422 | Politique mondiale |
| DS2 | Political Sentiment Analysis (subhajournal) | 767 | Conflit Ukraine |
| DS3 | US Election 2020 Tweets (manchunhui) | 1,456,644 | Election Trump/Biden |

---

## Architecture

```
NLP_Projet/
    DS1/ DS2/ DS3/                    donnees brutes (non versionnees)
    outputs/                          CSV nettoyes (non versionnes)
    models/
        best_model.pkl                LinearSVC entraine
        tfidf_vectorizer.pkl          TF-IDF bigrammes 50k features
    reports/
        comparaison_modeles.png       graphiques des 5 modeles
        rapport_modelisation.txt      metriques detaillees
        comparaison_modeles.csv       tableau comparatif
    mlruns/                           runs MLflow (non versionnes)
    app_streamlit.py                  Dashboard Streamlit
    setup_models.py                   telechargement auto du modele
    Dockerfile.streamlit              image Docker Streamlit
    docker-compose.yml                build local
    docker-compose.hub.yml            pull depuis Docker Hub
    requirements.txt                  dependances Python
    deploy.ps1                        script build + push + GitHub
```

---

## Lancement rapide

### Option 1 — Venv local (recommande pour le developpement)

```powershell
# Setup unique : cree le venv + installe tout + ressources NLTK
powershell -ExecutionPolicy Bypass -File setup_venv.ps1

# Ensuite ouvrir 2 terminaux dans NLP_Projet/

# Terminal 1 - Dashboard Streamlit
venv\Scripts\streamlit run app_streamlit.py

# Terminal 2 - MLflow UI
venv\Scripts\mlflow ui --backend-store-uri ./mlruns
```

### Option 2 — Docker Hub (aucun build, aucune installation Python)

```bash
docker-compose -f docker-compose.hub.yml up
```

### Option 3 — Docker build local

```bash
docker-compose up --build
```

**Services :**

| Service | URL |
|---|---|
| Dashboard Streamlit | http://localhost:8501 |
| MLflow UI | http://localhost:5000 |

---

## Images Docker Hub

```bash
docker pull votre_username/projet11-streamlit:latest
```

---

## Pipeline complet

```
Tweets bruts (3 datasets Kaggle)
    -> Nettoyage : URLs, mentions, hashtags, emojis
    -> Tokenisation + Stopwords + Lemmatisation (NLTK)
    -> Labellisation automatique VADER
    -> TF-IDF bigrammes (50 000 features)
    -> 5 classifieurs entraines + Cross-Validation 5-fold
    -> Tracking MLflow + Model Registry
    -> Meilleur modele : LinearSVC (F1 = 0.8902)
    -> Dashboard Streamlit + Docker
```

---

## Technologies

| Categorie | Outils |
|---|---|
| NLP et ML | scikit-learn, NLTK, VADER, TF-IDF |
| MLOps | MLflow tracking + Model Registry |
| Dashboard | Streamlit |
| Docker | Dockerfile.api, Dockerfile.streamlit, docker-compose |
| GitHub | versionnement du code |
| Environnement | Google Colab, Google Drive |

---
