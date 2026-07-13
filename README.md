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
uvicorn src.nlp_project.api:app --reload --port 8001
```

Dashboard Streamlit seul, avec fallback modele local :

```powershell
streamlit run app_streamlit.py
```

Dashboard Streamlit connecte a l'API :

```powershell
$env:API_URL="http://localhost:8001"
streamlit run app_streamlit.py
```

MLflow :

```powershell
mlflow ui --backend-store-uri ./mlruns
```

URLs :

| Service | URL |
|---|---|
| API | http://localhost:8001 |
| Documentation API | http://localhost:8001/docs |
| Streamlit | http://localhost:8501 |
| MLflow | http://localhost:5000 |

## Docker

Build local :

```bash
docker-compose up --build
```

L'API est exposee sur `http://localhost:8001` pour eviter les conflits avec un autre service deja lance sur le port `8000`. Dans Docker, Streamlit continue a appeler l'API via `http://api:8000` sur le reseau interne.

Images Docker Hub deja preparees :

```bash
docker-compose -f docker-compose.hub.yml up
```

## Tests

```powershell
pytest
```

## Modelisation avancee

Le script `src/train.py` permet de relancer la modelisation avec une source de donnees configurable, sans modifier le code :

```powershell
python src/train.py --data-source "C:\chemin\dataset.csv" --sample-per-class 200000
```

Il est aussi possible d'utiliser une variable d'environnement ou un identifiant Google Drive compatible `gdown` :

```powershell
$env:DATASET_SOURCE="gdown://FILE_ID"
python src/train.py --skip-tuning
```

Variantes implementees :

- `linearsvc_baseline` : LinearSVC seul, reference actuelle.
- `ensemble_vote_pondere` : vote pondere LinearSVC + Logistic Regression.
- `routage_confiance` : Logistic Regression appelee seulement quand la marge LinearSVC est faible.
- `linearsvc_tuned` : recherche sur `C` et `ngram_range`.

Les runs sont traces dans MLflow avec un backend local SQLite par defaut :

```powershell
python src/train.py --data-source "C:\chemin\dataset.csv" --mlflow-uri "sqlite:///mlflow.db"
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

Les rapports des variantes sont sauvegardes dans `reports/modelisation_variantes/`. L'option `--export-best` remplace `models/best_model.pkl` et `models/tfidf_vectorizer.pkl` en gardant le format attendu par l'API.

## Livrables

| Livrable | Emplacement |
|---|---|
| README  | `README.md` |
| Notebook | `notebooks/projet11_pipeline.ipynb - projet11_modelisation.ipynb` |
| Code organise dans `src/` | `src/nlp_project/` |
| Modele sauvegarde | `models/best_model.pkl`, `models/tfidf_vectorizer.pkl` |
| API fonctionnelle | `src/nlp_project/api.py` |
| Tests pytest | `tests/` |
| Capture MLflow | `reports/captures/mlflow.png` |
| Capture FastAPI `/docs` | `reports/captures/fastapi_docs.png` |
| Rapport | `reports/rapport_projet11.docx` |

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
