# =============================================================
# Projet 11 - Script de deploiement complet
# Build Docker -> Push Docker Hub -> Init GitHub
#
# PREREQUIS :
#   1. Docker Desktop installe et en cours d'execution
#   2. Compte Docker Hub : https://hub.docker.com
#   3. Git installe : https://git-scm.com/download/win
#   4. Compte GitHub + depot vide cree : https://github.com
#
# UTILISATION :
#   cd C:\Users\Vina\Documents\NLP_Projet
#   powershell -ExecutionPolicy Bypass -File deploy.ps1
# =============================================================

# >>> MODIFIEZ CES 4 VALEURS AVANT DE LANCER <<<
$DOCKERHUB_USERNAME = "quantumspider777"
$GITHUB_USERNAME    = "Vina771"
$GITHUB_REPO        = "projet-nlp"
$GIT_EMAIL          = "raharitsifa.vina@gmail.com"

# =============================================================
$DOCKERHUB_STREAMLIT_IMAGE = "${DOCKERHUB_USERNAME}/projet11-streamlit:latest"
$GITHUB_REPO_URL           = "https://github.com/${GITHUB_USERNAME}/${GITHUB_REPO}.git"

Write-Host ""
Write-Host "Projet 11 - Deploiement complet" -ForegroundColor Cyan
Write-Host "Docker Hub : $DOCKERHUB_USERNAME" -ForegroundColor Gray
Write-Host "GitHub     : $GITHUB_REPO_URL"    -ForegroundColor Gray
Write-Host ""

# =============================================================
# ETAPE 1 - Verifications
# =============================================================
Write-Host "[1/5] Verification des outils..." -ForegroundColor Yellow

try { docker --version | Out-Null; Write-Host "  Docker OK" -ForegroundColor Green }
catch { Write-Host "  ERREUR : Docker non installe ou non demarre." -ForegroundColor Red; exit 1 }

try { git --version | Out-Null; Write-Host "  Git OK" -ForegroundColor Green }
catch { Write-Host "  ERREUR : Git non installe." -ForegroundColor Red; exit 1 }

if (-not (Test-Path "models\best_model.pkl")) {
    Write-Host "  ERREUR : models\best_model.pkl introuvable." -ForegroundColor Red
    Write-Host "  Telechargez-le depuis Google Drive d'abord." -ForegroundColor Red
    exit 1
}
Write-Host "  best_model.pkl OK" -ForegroundColor Green

if (-not (Test-Path "models\tfidf_vectorizer.pkl")) {
    Write-Host "  ERREUR : models\tfidf_vectorizer.pkl introuvable." -ForegroundColor Red
    exit 1
}
Write-Host "  tfidf_vectorizer.pkl OK" -ForegroundColor Green

# =============================================================
# ETAPE 2 - Connexion Docker Hub
# =============================================================
Write-Host ""
Write-Host "[2/5] Connexion a Docker Hub..." -ForegroundColor Yellow
Write-Host "  Entrez votre mot de passe Docker Hub quand demande."
docker login --username $DOCKERHUB_USERNAME
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERREUR : Connexion Docker Hub echouee." -ForegroundColor Red
    exit 1
}
Write-Host "  Connexion OK" -ForegroundColor Green

# =============================================================
# ETAPE 3 - Build et Push des images Docker
# =============================================================
Write-Host ""
Write-Host "[3/5] Build et push des images Docker..." -ForegroundColor Yellow

# Build Streamlit
Write-Host "  Build image Streamlit..."
docker build -f Dockerfile.streamlit -t $DOCKERHUB_STREAMLIT_IMAGE .
if ($LASTEXITCODE -ne 0) { Write-Host "  ERREUR build Streamlit" -ForegroundColor Red; exit 1 }
Write-Host "  Build Streamlit OK" -ForegroundColor Green

# Push Streamlit
Write-Host "  Push image Streamlit vers Docker Hub..."
docker push $DOCKERHUB_STREAMLIT_IMAGE
if ($LASTEXITCODE -ne 0) { Write-Host "  ERREUR push Streamlit" -ForegroundColor Red; exit 1 }
Write-Host "  Push Streamlit OK" -ForegroundColor Green

Write-Host ""
Write-Host "  Images disponibles sur Docker Hub :" -ForegroundColor Green
Write-Host "    https://hub.docker.com/r/$DOCKERHUB_USERNAME/projet11-api"
Write-Host "    https://hub.docker.com/r/$DOCKERHUB_USERNAME/projet11-streamlit"

# =============================================================
# ETAPE 4 - Mise a jour des fichiers avec le bon username
# =============================================================
Write-Host ""
Write-Host "[4/5] Mise a jour des fichiers avec votre username Docker Hub..." -ForegroundColor Yellow

(Get-Content docker-compose.yml)     -replace "votre_username", $DOCKERHUB_USERNAME |
    Set-Content docker-compose.yml
(Get-Content docker-compose.hub.yml) -replace "votre_username", $DOCKERHUB_USERNAME |
    Set-Content docker-compose.hub.yml

Write-Host "  docker-compose.yml mis a jour" -ForegroundColor Green
Write-Host "  docker-compose.hub.yml mis a jour" -ForegroundColor Green

# =============================================================
# ETAPE 5 - Initialisation et push GitHub
# =============================================================
Write-Host ""
Write-Host "[5/5] Initialisation du depot GitHub..." -ForegroundColor Yellow

git config --global user.name  $GITHUB_USERNAME
git config --global user.email $GIT_EMAIL

# Creer les dossiers vides trackables
$folders = @("models", "outputs", "reports", "mlruns")
foreach ($folder in $folders) {
    if (-not (Test-Path $folder)) { New-Item -ItemType Directory -Path $folder | Out-Null }
    $keep = "$folder\.gitkeep"
    if (-not (Test-Path $keep)) { New-Item -ItemType File -Path $keep | Out-Null }
}

# Init git si pas encore fait
if (-not (Test-Path ".git")) {
    git init
    Write-Host "  Depot Git initialise" -ForegroundColor Green
}

# Ajouter les fichiers de code uniquement
$filesToAdd = @(
    ".gitignore",
    "README.md",
    "requirements.txt",
    "api.py",
    "app_streamlit.py",
    "setup_models.py",
    "Dockerfile.api",
    "Dockerfile.streamlit",
    "docker-compose.yml",
    "docker-compose.hub.yml",
    "models\.gitkeep",
    "outputs\.gitkeep",
    "reports\.gitkeep"
)

foreach ($file in $filesToAdd) {
    if (Test-Path $file) { git add $file }
}

# Ajouter config streamlit
if (Test-Path ".streamlit") { git add ".streamlit/" }

# Ajouter les rapports (legers, utiles pour la demo)
if (Test-Path "reports\comparaison_modeles.png")  { git add "reports\comparaison_modeles.png" }
if (Test-Path "reports\rapport_modelisation.txt") { git add "reports\rapport_modelisation.txt" }
if (Test-Path "reports\comparaison_modeles.csv")  { git add "reports\comparaison_modeles.csv" }

# Ajouter les notebooks si presents
if (Test-Path "projet11_modelisation_v2.ipynb") { git add "projet11_modelisation_v2.ipynb" }
if (Test-Path "projet11_pipeline.py")           { git add "projet11_pipeline.py" }

git commit -m "Projet 11 - NLP Sentiment Analysis - Deploiement complet

Datasets     : DS1 (238k) + DS2 (767) + DS3 (1.45M) = 1.69M tweets
Modele       : LinearSVC + TF-IDF bigrammes 50k features
F1 Test      : 0.8902
MLflow       : 6 runs trackes + Model Registry Projet11_SentimentAnalysis
API          : FastAPI /predict /predict/batch /health
Dashboard    : Streamlit 6 pages + analyse temps reel
Docker Hub   : $DOCKERHUB_STREAMLIT_IMAGE"

# Connecter et pousser
$remoteExists = git remote get-url origin 2>$null
if ($remoteExists) {
    git remote set-url origin $GITHUB_REPO_URL
} else {
    git remote add origin $GITHUB_REPO_URL
}

git branch -M main
git push -u origin main

Write-Host ""
Write-Host "Deploiement termine avec succes !" -ForegroundColor Green
Write-Host ""
Write-Host "RESUME :" -ForegroundColor Cyan
Write-Host "  GitHub     : $GITHUB_REPO_URL"
Write-Host "  Docker Hub Streamlit : https://hub.docker.com/r/$DOCKERHUB_USERNAME/projet11-streamlit"
Write-Host ""
Write-Host "POUR LANCER DEPUIS N'IMPORTE QUELLE MACHINE :" -ForegroundColor Cyan
Write-Host "  docker-compose -f docker-compose.hub.yml up"
Write-Host ""
Write-Host "ACCES AUX SERVICES :" -ForegroundColor Cyan
Write-Host "  Streamlit  : http://localhost:8501"
Write-Host "  MLflow UI  : http://localhost:5000"
