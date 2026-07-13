"""
PROJET 11 - ANALYSE DES DISCOURS POLITIQUES (NLP)
Pipeline : Installation + Data Mining + Cleaning + Preprocessing
Master 1 IA & Data Science - INSI Madagascar

Structure attendue dans G:\Mon Drive\NLP_Project\ :
    NLP_Project\
        DS1\
            Political_tweets.csv
        DS2\
            Set-1\
                NATO.csv, SaveUkraineNow.csv, ... (13 fichiers)
            Set-2\
                ... (13 fichiers)
        DS3\
            hashtag_donaldtrump.csv
            hashtag_joebiden.csv
        projet11_pipeline.py

Lancement depuis le dossier NLP_Project :
    cd "G:\Mon Drive\NLP_Project"
    python projet11_pipeline.py
"""

# ===========================================================
# ETAPE 0 - INSTALLATION AUTOMATIQUE DES DEPENDANCES
# ===========================================================
import subprocess
import sys

PACKAGES = [
    "pandas",
    "numpy",
    "matplotlib",
    "seaborn",
    "nltk",
    "vaderSentiment",
]

print("ETAPE 0 - Verification et installation des dependances")
for package in PACKAGES:
    try:
        __import__(package)
        print(f"  OK  {package} deja installe")
    except ImportError:
        print(f"  Installation de {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "--quiet"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  OK  {package} installe avec succes")

print("Toutes les dependances sont disponibles.")
print()


# ===========================================================
# ETAPE 1 - IMPORTS
# ===========================================================
import os
import re
import ast
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.filterwarnings("ignore")

print("ETAPE 1 - Telechargement des ressources NLTK")
for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass
print("Ressources NLTK OK")
print()


# ===========================================================
# ETAPE 2 - CONFIGURATION DES CHEMINS
# ===========================================================
# Le script doit etre lance depuis G:\Mon Drive\NLP_Project\
# Exemple : cd "G:\Mon Drive\NLP_Project" puis python projet11_pipeline.py
# Les fichiers de sortie seront dans G:\Mon Drive\NLP_Project\outputs\

BASE_DIR  = Path(".")
DS1_PATH  = BASE_DIR / "DS1" / "Political_tweets.csv"
DS2_DIR   = BASE_DIR / "DS2"
DS3_TRUMP = BASE_DIR / "DS3" / "hashtag_donaldtrump.csv"
DS3_BIDEN = BASE_DIR / "DS3" / "hashtag_joebiden.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

print("ETAPE 2 - Configuration des chemins")
print(f"  DS1 : {DS1_PATH.resolve()}")
print(f"  DS2 : {DS2_DIR.resolve()}")
print(f"  DS3 Trump : {DS3_TRUMP.resolve()}")
print(f"  DS3 Biden : {DS3_BIDEN.resolve()}")
print(f"  Outputs   : {OUTPUT_DIR.resolve()}")
print()


# ===========================================================
# ETAPE 3 - CHARGEMENT DES DONNEES
# ===========================================================
print("ETAPE 3 - Chargement des donnees")

# DS1 - Global Political Tweets (122 MB)
print("  Chargement DS1 : Global Political Tweets...")
try:
    df1 = pd.read_csv(DS1_PATH, on_bad_lines="skip", low_memory=False)
    df1 = df1.rename(columns={"text": "tweet", "date": "created_at"})
    df1["source_dataset"] = "DS1_global_political"
    print(f"  DS1 charge : {df1.shape[0]:,} lignes x {df1.shape[1]} colonnes")
except FileNotFoundError:
    print(f"  DS1 introuvable : {DS1_PATH}")
    df1 = pd.DataFrame()

# DS2 - Political Sentiment Analysis Ukraine (26 fichiers CSV)
print("  Chargement DS2 : Political Sentiment Analysis (Set-1 et Set-2)...")
ds2_frames = []
if DS2_DIR.exists():
    for set_folder in ["Set-1", "Set-2"]:
        folder = DS2_DIR / set_folder
        if folder.exists():
            for csv_file in sorted(folder.glob("*.csv")):
                try:
                    tmp = pd.read_csv(csv_file, on_bad_lines="skip")
                    tmp["hashtag_file"] = csv_file.stem
                    tmp["set_folder"]   = set_folder
                    ds2_frames.append(tmp)
                except Exception as e:
                    print(f"    Erreur lecture {csv_file.name} : {e}")
    if ds2_frames:
        df2 = pd.concat(ds2_frames, ignore_index=True)
        df2 = df2.rename(columns={"text": "tweet"})
        df2["source_dataset"] = "DS2_ukraine_conflict"
        print(f"  DS2 charge : {df2.shape[0]:,} lignes ({len(ds2_frames)} fichiers)")
    else:
        print("  Aucun CSV trouve dans DS2")
        df2 = pd.DataFrame()
else:
    print(f"  DS2 introuvable : {DS2_DIR}")
    df2 = pd.DataFrame()

# DS3 - US Election 2020 Tweets (Trump + Biden, ~865 MB total)
# Chargement par chunks pour eviter la saturation de la RAM
print("  Chargement DS3 : US Election 2020 Tweets (Trump + Biden)...")
ds3_frames = []
for path, label in [(DS3_TRUMP, "trump"), (DS3_BIDEN, "biden")]:
    try:
        chunks = []
        for chunk in pd.read_csv(
            path, chunksize=100_000, on_bad_lines="skip", low_memory=False
        ):
            chunks.append(chunk)
        tmp = pd.concat(chunks, ignore_index=True)
        tmp["candidate"] = label
        ds3_frames.append(tmp)
        print(f"  DS3 {label} charge : {tmp.shape[0]:,} lignes")
    except FileNotFoundError:
        print(f"  DS3 {label} introuvable : {path}")

if ds3_frames:
    df3 = pd.concat(ds3_frames, ignore_index=True)
    df3["source_dataset"] = "DS3_us_election_2020"
    print(f"  DS3 total : {df3.shape[0]:,} lignes")
else:
    df3 = pd.DataFrame()

print()


# ===========================================================
# ETAPE 4 - EXPLORATION RAPIDE
# ===========================================================
print("ETAPE 4 - Exploration rapide des datasets")

def eda_summary(df, name):
    if df.empty:
        print(f"  [{name}] vide - ignore")
        return
    print(f"  [{name}]")
    print(f"    Lignes   : {df.shape[0]:,}")
    print(f"    Colonnes : {df.columns.tolist()}")
    missing = (df.isnull().sum() / len(df) * 100).round(1)
    missing = missing[missing > 0]
    if not missing.empty:
        print(f"    Valeurs manquantes (%) :")
        print(missing.to_string())
    print()

eda_summary(df1, "DS1 Global Political Tweets")
eda_summary(df2, "DS2 Political Sentiment Analysis")
eda_summary(df3, "DS3 US Election 2020")


# ===========================================================
# ETAPE 5 - FONCTIONS DE NETTOYAGE
# ===========================================================
print("ETAPE 5 - Preparation des fonctions de nettoyage")

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

# Mots politiques importants a conserver malgre les stopwords
KEEP_WORDS = {
    "not", "no", "never", "against", "war", "peace",
    "trump", "biden", "ukraine", "russia", "nato",
    "election", "vote", "democrat", "republican"
}
STOP_WORDS = STOP_WORDS - KEEP_WORDS


def clean_tweet(text):
    """
    Nettoyage d'un tweet :
    - Supprime les URLs, mentions @user, RT
    - Garde le mot derriere les hashtags (#ukraine -> ukraine)
    - Supprime les emojis et caracteres non-ASCII
    - Supprime la ponctuation et les chiffres
    - Met en minuscules
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"^RT\s+", "", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_tweet(text):
    """
    Preprocessing NLP complet :
    - Nettoyage (clean_tweet)
    - Tokenisation
    - Suppression des stopwords
    - Lemmatisation
    - Filtre les tokens de moins de 3 caracteres
    """
    text = clean_tweet(text)
    if not text:
        return ""
    tokens = word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]
    return " ".join(tokens)


def extract_hashtags(raw):
    """Extrait les hashtags depuis une chaine de type ['politics', 'nato']."""
    if isinstance(raw, list):
        return [h.lower() for h in raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            return [h.lower() for h in parsed]
        except Exception:
            return [raw.lower()]
    return []


print("  Fonctions de nettoyage pretes")
print()


# ===========================================================
# ETAPE 6 - NETTOYAGE DS1
# ===========================================================
print("ETAPE 6 - Nettoyage DS1")

if not df1.empty:
    df1 = df1.drop_duplicates(subset=["tweet"])
    if "is_retweet" in df1.columns:
        df1 = df1[df1["is_retweet"] == False].copy()
    df1 = df1[df1["tweet"].notna() & (df1["tweet"].str.strip() != "")]
    df1["tweet_clean"]  = df1["tweet"].apply(clean_tweet)
    df1["tweet_tokens"] = df1["tweet"].apply(preprocess_tweet)
    if "hashtags" in df1.columns:
        df1["hashtags_list"] = df1["hashtags"].apply(extract_hashtags)
    df1["created_at"]   = pd.to_datetime(df1["created_at"], errors="coerce")
    df1["tweet_length"] = df1["tweet_clean"].str.split().str.len()
    df1 = df1[df1["tweet_length"] >= 3]
    df1 = df1.drop(
        columns=["user_description", "user_created", "user_verified", "is_retweet"],
        errors="ignore"
    )
    print(f"  DS1 nettoye : {df1.shape[0]:,} lignes")
else:
    print("  DS1 vide - ignore")
print()


# ===========================================================
# ETAPE 7 - NETTOYAGE DS2
# ===========================================================
print("ETAPE 7 - Nettoyage DS2")

if not df2.empty:
    df2 = df2.drop_duplicates(subset=["tweet"])
    df2 = df2[df2["tweet"].notna() & (df2["tweet"].str.strip() != "")]
    df2 = df2[~df2["tweet"].str.startswith("RT ", na=False)]
    df2["tweet_clean"]  = df2["tweet"].apply(clean_tweet)
    df2["tweet_tokens"] = df2["tweet"].apply(preprocess_tweet)
    df2["tweet_length"] = df2["tweet_clean"].str.split().str.len()
    df2 = df2[df2["tweet_length"] >= 3]
    print(f"  DS2 nettoye : {df2.shape[0]:,} lignes")
else:
    print("  DS2 vide - ignore")
print()


# ===========================================================
# ETAPE 8 - NETTOYAGE DS3
# ===========================================================
print("ETAPE 8 - Nettoyage DS3")

if not df3.empty:
    if "tweet_id" in df3.columns:
        df3 = df3.drop_duplicates(subset=["tweet_id"], keep="first")
    else:
        df3 = df3.drop_duplicates(subset=["tweet"])
    df3 = df3[df3["tweet"].notna() & (df3["tweet"].str.strip() != "")]
    df3 = df3[~df3["tweet"].str.startswith("RT ", na=False)]
    df3["tweet_clean"]  = df3["tweet"].apply(clean_tweet)
    df3["tweet_tokens"] = df3["tweet"].apply(preprocess_tweet)
    df3["tweet_length"] = df3["tweet_clean"].str.split().str.len()
    df3 = df3[df3["tweet_length"] >= 3]
    if "created_at" in df3.columns:
        df3["created_at"] = pd.to_datetime(df3["created_at"], errors="coerce")
    print(f"  DS3 nettoye : {df3.shape[0]:,} lignes")
else:
    print("  DS3 vide - ignore")
print()


# ===========================================================
# ETAPE 9 - LABELLISATION DU SENTIMENT AVEC VADER
# ===========================================================
print("ETAPE 9 - Labellisation du sentiment avec VADER")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
    print("  VADER charge avec succes")
except ImportError:
    try:
        nltk.download("vader_lexicon", quiet=True)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        vader = SentimentIntensityAnalyzer()
        VADER_AVAILABLE = True
        print("  VADER NLTK charge avec succes")
    except Exception:
        VADER_AVAILABLE = False
        print("  VADER non disponible - labels non generes")


def label_sentiment(text):
    """
    Classe le sentiment d'un texte en 3 categories :
    - positive  : score VADER >= 0.05
    - negative  : score VADER <= -0.05
    - neutral   : score entre -0.05 et 0.05
    """
    if not VADER_AVAILABLE or not isinstance(text, str) or text.strip() == "":
        return "neutral"
    score = vader.polarity_scores(text)["compound"]
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    else:
        return "neutral"


def get_vader_score(text):
    """Retourne le score compound VADER brut entre -1 et 1."""
    if not VADER_AVAILABLE or not isinstance(text, str):
        return 0.0
    return vader.polarity_scores(text)["compound"]


if VADER_AVAILABLE:
    for df, name in [(df1, "DS1"), (df2, "DS2"), (df3, "DS3")]:
        if not df.empty and "tweet_clean" in df.columns:
            print(f"  Labellisation {name}...")
            df["sentiment_label"] = df["tweet_clean"].apply(label_sentiment)
            df["sentiment_score"] = df["tweet_clean"].apply(get_vader_score)
            dist = df["sentiment_label"].value_counts(normalize=True).mul(100).round(1)
            print(f"    Distribution : {dist.to_dict()}")

print()


# ===========================================================
# ETAPE 10 - HARMONISATION ET FUSION
# ===========================================================
print("ETAPE 10 - Harmonisation et fusion des datasets")

COMMON_COLS = [
    "tweet",
    "tweet_clean",
    "tweet_tokens",
    "tweet_length",
    "sentiment_label",
    "sentiment_score",
    "source_dataset",
    "created_at",
]


def harmonize(df, name):
    if df.empty:
        return df
    available = [c for c in COMMON_COLS if c in df.columns]
    print(f"  [{name}] colonnes conservees : {available}")
    return df[available].copy()


df1_h = harmonize(df1, "DS1")
df2_h = harmonize(df2, "DS2")
df3_h = harmonize(df3, "DS3")

frames = [f for f in [df1_h, df2_h, df3_h] if not f.empty]
if frames:
    df_all = pd.concat(frames, ignore_index=True)
    print(f"  Dataset fusionne : {df_all.shape[0]:,} lignes au total")
    if "sentiment_label" in df_all.columns:
        print("  Distribution globale du sentiment :")
        print(df_all["sentiment_label"].value_counts().to_string())
else:
    df_all = pd.DataFrame()
    print("  Aucun dataset disponible pour la fusion")

print()


# ===========================================================
# ETAPE 11 - EXPORT DES FICHIERS NETTOYES
# ===========================================================
print("ETAPE 11 - Export des fichiers nettoyes")

export_map = {
    "ds1_global_political_clean.csv" : df1_h,
    "ds2_ukraine_conflict_clean.csv" : df2_h,
    "ds3_us_election_2020_clean.csv" : df3_h,
    "dataset_fusionne_complet.csv"   : df_all,
}

for filename, df in export_map.items():
    if not df.empty:
        path = OUTPUT_DIR / filename
        df.to_csv(path, index=False)
        size_mb = path.stat().st_size / 1_048_576
        print(f"  Exporte : {filename} ({df.shape[0]:,} lignes, {size_mb:.1f} MB)")
    else:
        print(f"  Vide - non exporte : {filename}")

print()


# ===========================================================
# ETAPE 12 - VISUALISATIONS
# ===========================================================
print("ETAPE 12 - Generation des visualisations")

plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Projet 11 - Exploration des Donnees Politiques",
    fontsize=16, fontweight="bold"
)

colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}

# Plot 1 - Distribution du sentiment par dataset
ax1 = axes[0, 0]
sentiment_data = {}
for df_h, name in [(df1_h, "DS1"), (df2_h, "DS2"), (df3_h, "DS3")]:
    if not df_h.empty and "sentiment_label" in df_h.columns:
        sentiment_data[name] = (
            df_h["sentiment_label"].value_counts(normalize=True) * 100
        )

if sentiment_data:
    sent_df = pd.DataFrame(sentiment_data).T.fillna(0)
    for col in ["positive", "neutral", "negative"]:
        if col not in sent_df.columns:
            sent_df[col] = 0
    sent_df[["positive", "neutral", "negative"]].plot(
        kind="bar", ax=ax1,
        color=[colors["positive"], colors["neutral"], colors["negative"]],
        edgecolor="white", width=0.7
    )
    ax1.set_title("Distribution du sentiment par dataset (%)")
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("Pourcentage (%)")
    ax1.legend(["Positif", "Neutre", "Negatif"])
    ax1.tick_params(axis="x", rotation=0)

# Plot 2 - Distribution des scores VADER
ax2 = axes[0, 1]
for df_h, name, color in [
    (df1_h, "DS1", "#3498db"),
    (df2_h, "DS2", "#e67e22"),
    (df3_h, "DS3", "#9b59b6"),
]:
    if not df_h.empty and "sentiment_score" in df_h.columns:
        sample = df_h["sentiment_score"].dropna().sample(
            min(5000, len(df_h)), random_state=42
        )
        ax2.hist(sample, bins=50, alpha=0.5, label=name, color=color)
ax2.axvline(0.05,  color="green", linestyle="--", linewidth=1, label="Seuil positif")
ax2.axvline(-0.05, color="red",   linestyle="--", linewidth=1, label="Seuil negatif")
ax2.set_title("Distribution des scores VADER")
ax2.set_xlabel("Score compound")
ax2.set_ylabel("Frequence")
ax2.legend(fontsize=8)

# Plot 3 - Longueur des tweets en nombre de mots
ax3 = axes[1, 0]
for df_h, name, color in [
    (df1_h, "DS1", "#3498db"),
    (df2_h, "DS2", "#e67e22"),
    (df3_h, "DS3", "#9b59b6"),
]:
    if not df_h.empty and "tweet_length" in df_h.columns:
        sample = df_h["tweet_length"].dropna().sample(
            min(5000, len(df_h)), random_state=42
        )
        ax3.hist(sample.clip(0, 50), bins=25, alpha=0.5, label=name, color=color)
ax3.set_title("Longueur des tweets (nombre de mots apres nettoyage)")
ax3.set_xlabel("Nombre de mots")
ax3.set_ylabel("Frequence")
ax3.legend()

# Plot 4 - Taille comparative des datasets
ax4 = axes[1, 1]
ds_sizes = {}
for df_h, name in [
    (df1_h, "DS1\nGlobal"),
    (df2_h, "DS2\nUkraine"),
    (df3_h, "DS3\nElection"),
]:
    if not df_h.empty:
        ds_sizes[name] = len(df_h)

if ds_sizes:
    bar_colors = ["#3498db", "#e67e22", "#9b59b6"]
    bars = ax4.bar(
        list(ds_sizes.keys()),
        list(ds_sizes.values()),
        color=bar_colors[: len(ds_sizes)],
        edgecolor="white"
    )
    for bar in bars:
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(ds_sizes.values()) * 0.01,
            f"{int(bar.get_height()):,}",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    ax4.set_title("Taille des datasets apres nettoyage")
    ax4.set_ylabel("Nombre de tweets")
    ax4.set_ylim(0, max(ds_sizes.values()) * 1.15)

plt.tight_layout()
viz_path = OUTPUT_DIR / "exploration_donnees.png"
plt.savefig(viz_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Visualisation sauvegardee : {viz_path}")
print()


# ===========================================================
# ETAPE 13 - RAPPORT FINAL
# ===========================================================
print("ETAPE 13 - Generation du rapport final")

report_lines = [
    "PROJET 11 - RAPPORT DE PREPROCESSING",
    f"Genere le : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    "",
]

for df_h, name, desc in [
    (df1_h, "DS1", "Global Political Tweets"),
    (df2_h, "DS2", "Political Sentiment Analysis (Ukraine)"),
    (df3_h, "DS3", "US Election 2020 Tweets"),
]:
    report_lines.append(f"[{name}] {desc}")
    if df_h.empty:
        report_lines.append("  Dataset non charge (fichier manquant ou introuvable)")
    else:
        report_lines.append(f"  Lignes finales  : {len(df_h):,}")
        report_lines.append(f"  Colonnes        : {df_h.columns.tolist()}")
        if "sentiment_label" in df_h.columns:
            dist = df_h["sentiment_label"].value_counts()
            report_lines.append(f"  Sentiment :\n{dist.to_string()}")
    report_lines.append("")

if not df_all.empty:
    report_lines.append(f"[FUSION TOTALE] {len(df_all):,} tweets")
    report_lines.append("")

report_lines.append("[FICHIERS EXPORTES]")
for f in OUTPUT_DIR.glob("*.csv"):
    size_mb = f.stat().st_size / 1_048_576
    report_lines.append(f"  {f.name} ({size_mb:.1f} MB)")

report_text = "\n".join(report_lines)
print(report_text)

report_path = OUTPUT_DIR / "rapport_preprocessing.txt"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)

print()
print(f"Rapport sauvegarde : {report_path}")
print()
print("Pipeline termine avec succes.")
print(f"Tous les fichiers sont dans : {OUTPUT_DIR.resolve()}")