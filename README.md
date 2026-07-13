# Projet 11 - Analyse des discours politiques

Auteur : Vina RAHARITSIFA - M1 I2AD, INSI

Analyse de sentiment sur des tweets politiques en anglais. Le modele a ete entraine sur Google Colab avec trois datasets Kaggle, puis sauvegarde dans `models/` pour servir une API FastAPI et un dashboard Streamlit.

## Resultats

| Modele | F1 test | Accuracy test | CV F1 |
|---|---:|---:|---:|
| LinearSVC | 0.8902 | 0.8904 | 0.8835 |
| Logistic Regression | 0.8821 | 0.8824 | 0.8739 |
| Naive Bayes | 0.7224 | 0.7226 | 0.7214 |
| Random Forest | 0.6994 | 0.7021 | 0.6962 |
| Gradient Boosting | 0.6168 | 0.6171 | 0.6153 |

Meilleur modele : `LinearSVC` avec TF-IDF bigrammes, 50 000 features.

## Donnees

| Dataset | Source | Tweets apres nettoyage |
|---|---|---:|
| DS1 - Global Political Tweets | Kaggle, kaushiksuresh147 | 238 422 |
| DS2 - Political Sentiment Analysis | Kaggle, subhajournal | 767 |
| DS3 - US Election 2020 Tweets | Kaggle, manchunhui | 1 456 644 |

Total preprocessing : 1 695 833 tweets.  
Total utilise pour la modelisation apres filtrage et echantillonnage stratifie : 600 000 tweets.

## Structure

```text
NLP_Projet/
  src/nlp_project/              code partage API / preprocessing inference
  tests/                        tests pytest
  models/                       modele et vectoriseur sauvegardes
  reports/                      rapports, figures et captures de livraison
  outputs/                      sorties locales du preprocessing
  notebooks/
    projet11_pipeline.ipynb       notebook d'exploration / preprocessing Colab
    projet11_modelisation.ipynb   notebook de modelisation Colab
  app_streamlit.py              dashboard Streamlit connecte a l'API FastAPI via API_URL
  Dockerfile.api                image quantumspider777/projetnlp-api
  Dockerfile.streamlit          image quantumspider777/projetnlp-streamlit
  docker-compose.yml            lancement local API + Streamlit + MLflow
  docker-compose.hub.yml        lancement depuis images Docker Hub
```

Les donnees brutes et les gros CSV intermediaires ne sont pas versionnes.

## Lancement local

```powershell
powershell -ExecutionPolicy Bypass -File setup_venv.ps1
venv\Scripts\activate
```

API FastAPI :

```powershell
uvicorn src.nlp_project.api:app --reload
```

Dashboard Streamlit seul, avec fallback modele local :

```powershell
streamlit run app_streamlit.py
```

Dashboard Streamlit connecte a l'API :

```powershell
$env:API_URL="http://localhost:8000"
streamlit run app_streamlit.py
```

MLflow :

```powershell
mlflow ui --backend-store-uri ./mlruns
```

URLs :

| Service | URL |
|---|---|
| API | http://localhost:8000 |
| Documentation API | http://localhost:8000/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |

## Docker

Build local :

```bash
docker-compose up --build
```

Images Docker Hub deja preparees :

```bash
docker-compose -f docker-compose.hub.yml up
```

## Tests

```powershell
pytest
```

## Livrables

| Livrable | Emplacement |
|---|---|
| Repository GitHub complet | depot du projet |
| README clair | `README.md` |
| Notebook d'exploration | `notebooks/projet11_pipeline.ipynb` |
| Code organise dans `src/` | `src/nlp_project/` |
| Modele sauvegarde | `models/best_model.pkl`, `models/tfidf_vectorizer.pkl` |
| API fonctionnelle | `src/nlp_project/api.py` |
| Tests pytest | `tests/` |
| Capture MLflow | `reports/captures/mlflow.png` |
| Capture FastAPI `/docs` | `reports/captures/fastapi_docs.png` |
| Rapport court Word | `reports/rapport_projet11.docx` |

## Pipeline

```text
Tweets bruts
  -> nettoyage, deduplication, suppression RT
  -> tokenisation, stopwords, lemmatisation
  -> labellisation VADER
  -> TF-IDF bigrammes
  -> comparaison de 5 modeles
  -> tracking MLflow
  -> sauvegarde LinearSVC + vectoriseur
  -> API FastAPI et dashboard Streamlit
```
