Write-Host ""
Write-Host "Projet 11 - Setup environnement virtuel" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/4] Verification de Python..." -ForegroundColor Yellow

try {
    $pythonVersion = python --version 2>&1
    Write-Host "  $pythonVersion detecte" -ForegroundColor Green
} catch {
    Write-Host "  ERREUR : Python non installe." -ForegroundColor Red
    Write-Host "  Telechargez Python 3.11 : https://www.python.org/downloads/"
    exit 1
}

Write-Host ""
Write-Host "[2/4] Creation du venv..." -ForegroundColor Yellow

if (Test-Path "venv") {
    Write-Host "  venv/ deja existant - on le reutilise" -ForegroundColor Green
} else {
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERREUR : creation du venv echouee" -ForegroundColor Red
        exit 1
    }
    Write-Host "  venv/ cree" -ForegroundColor Green
}

Write-Host ""
Write-Host "[3/4] Installation des dependances..." -ForegroundColor Yellow

& "venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& "venv\Scripts\pip.exe" install -r requirements.txt

if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERREUR : installation des dependances echouee" -ForegroundColor Red
    exit 1
}
Write-Host "  Toutes les dependances installees" -ForegroundColor Green

Write-Host ""
Write-Host "[4/4] Telechargement des ressources NLTK..." -ForegroundColor Yellow

& "venv\Scripts\python.exe" -c "
import nltk
for r in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4', 'vader_lexicon']:
    nltk.download(r, quiet=True)
print('Ressources NLTK OK')
"
Write-Host "  NLTK OK" -ForegroundColor Green

Write-Host ""
Write-Host "Setup termine !" -ForegroundColor Green
Write-Host ""
Write-Host "COMMANDES POUR LANCER LES SERVICES :" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Activer le venv :"
Write-Host "    venv\Scripts\activate" -ForegroundColor White
Write-Host ""
Write-Host "  Dashboard Streamlit (Terminal 1) :"
Write-Host "    venv\Scripts\streamlit run app_streamlit.py" -ForegroundColor White
Write-Host "    -> http://localhost:8501"
Write-Host ""
Write-Host "  API FastAPI (Terminal 2) :"
Write-Host "    venv\Scripts\uvicorn src.nlp_project.api:app --reload --port 8001" -ForegroundColor White
Write-Host "    -> http://localhost:8001/docs"
Write-Host ""
Write-Host "  MLflow UI (Terminal 3) :"
Write-Host "    venv\Scripts\mlflow ui --backend-store-uri ./mlruns" -ForegroundColor White
Write-Host "    -> http://localhost:5000"
Write-Host ""
Write-Host "NOTE : Vous pouvez aussi tout lancer avec Docker :" -ForegroundColor Gray
Write-Host "  docker-compose up --build" -ForegroundColor Gray
