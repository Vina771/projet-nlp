"""
Projet 11 - Téléchargement automatique du modèle
Ce fichier est exécuté au démarrage de l'app Streamlit
Il télécharge best_model.pkl et tfidf_vectorizer.pkl depuis Google Drive
"""

import os
from pathlib import Path

MODEL_FILE_ID = "https://drive.google.com/file/d/1Uz5OWnaq0YNuB-sPX54HKsYa-Cyx_A-h/view?usp=sharing"
TFIDF_FILE_ID = "https://drive.google.com/file/d/1enUkbrp1GeLTREt9i0LDBhlfl8wpszJY/view?usp=sharing"

def download_models():
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_path = models_dir / "best_model.pkl"
    tfidf_path = models_dir / "tfidf_vectorizer.pkl"

    if model_path.exists() and tfidf_path.exists():
        print("Modeles deja presents.")
        return

    try:
        import gdown
    except ImportError:
        os.system("pip install gdown --quiet")
        import gdown

    if not model_path.exists():
        print("Telechargement de best_model.pkl depuis Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={MODEL_FILE_ID}",
            str(model_path),
            quiet=False
        )
        print("best_model.pkl telecharge.")

    if not tfidf_path.exists():
        print("Telechargement de tfidf_vectorizer.pkl depuis Google Drive...")
        gdown.download(
            f"https://drive.google.com/uc?id={TFIDF_FILE_ID}",
            str(tfidf_path),
            quiet=False
        )
        print("tfidf_vectorizer.pkl telecharge.")

    print("Modeles prets.")

if __name__ == "__main__":
    download_models()
