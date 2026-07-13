import argparse
import os
import tempfile
import time
from pathlib import Path

import gdown
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from nlp_project.ensemble import ConfidenceRouter, WeightedSentimentEnsemble
from nlp_project.text_processing import clean_and_tokenize


ROOT = Path(__file__).resolve().parents[1]
LABEL_MAP = {
    "negative": 0,
    "negatif": 0,
    "0": 0,
    0: 0,
    "neutral": 1,
    "neutre": 1,
    "1": 1,
    1: 1,
    "positive": 2,
    "positif": 2,
    "2": 2,
    2: 2,
}
DEFAULT_DATASETS = [
    ROOT / "outputs" / "modelisation_dataset.csv",
    ROOT / "outputs" / "tweets_modelisation.csv",
    ROOT / "outputs" / "tweets_labeled.csv",
    Path("/content/drive/MyDrive/NLP_Project/outputs/modelisation_dataset.csv"),
]


def resolve_data_source(value):
    source = value or os.getenv("DATASET_SOURCE")
    if not source:
        for candidate in DEFAULT_DATASETS:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Aucun dataset trouve. Fournir --data-source ou DATASET_SOURCE "
            "avec un CSV local ou un identifiant gdown."
        )

    source = str(source)
    if Path(source).exists():
        return Path(source)

    file_id = source.removeprefix("gdown://")
    output = Path(tempfile.gettempdir()) / "projet11_dataset.csv"
    gdown.download(id=file_id, output=str(output), quiet=False)
    return output


def load_dataset(source):
    data_path = resolve_data_source(source)
    df = pd.read_csv(data_path)

    text_column = next(
        (c for c in ["tweet_tokens", "tweet_clean", "clean_text", "tweet", "text", "content"] if c in df.columns),
        None,
    )
    label_column = next((c for c in ["sentiment", "label", "target", "class"] if c in df.columns), None)
    if text_column is None or label_column is None:
        raise ValueError(
            "Le CSV doit contenir une colonne texte "
            "(tweet_tokens/tweet_clean/clean_text/tweet/text/content) et une colonne label."
        )

    data = df[[text_column, label_column]].dropna().copy()
    data.columns = ["text", "label"]
    data["label"] = data["label"].map(lambda x: LABEL_MAP.get(str(x).strip().lower(), LABEL_MAP.get(x)))
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    if text_column != "tweet_tokens":
        data["text"] = data["text"].map(clean_and_tokenize)
    data = data[data["text"].str.len() > 0]
    return data


def stratified_sample(data, sample_per_class):
    if not sample_per_class:
        return data
    return (
        data.groupby("label", group_keys=False)
        .apply(lambda part: part.sample(min(len(part), sample_per_class), random_state=42))
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def split_data(data):
    train_df, temp_df = train_test_split(
        data, test_size=0.30, random_state=42, stratify=data["label"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df["label"]
    )
    return train_df, val_df, test_df


def evaluate(model, x_val, y_val, x_test, y_test):
    start = time.perf_counter()
    pred_val = model.predict(x_val)
    pred_test = model.predict(x_test)
    inference_ms = (time.perf_counter() - start) * 1000 / (len(y_val) + len(y_test))
    return {
        "f1_val": f1_score(y_val, pred_val, average="macro"),
        "f1_test": f1_score(y_test, pred_test, average="macro"),
        "inference_ms": inference_ms,
        "report": classification_report(y_test, pred_test, target_names=["negative", "neutral", "positive"]),
    }


def log_run(name, model, metrics, params, report_dir):
    report_path = report_dir / f"{name}_classification_report.txt"
    report_path.write_text(metrics["report"], encoding="utf-8")
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, float)})
        mlflow.log_artifact(str(report_path))
        mlflow.sklearn.log_model(model, artifact_path="model")


def build_pipeline(classifier, ngram_max):
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, ngram_max))),
            ("model", classifier),
        ]
    )


def export_pipeline_model(pipeline_or_search, models_dir):
    pipeline = getattr(pipeline_or_search, "best_estimator_", pipeline_or_search)
    joblib.dump(pipeline.named_steps["model"], models_dir / "best_model.pkl")
    joblib.dump(pipeline.named_steps["tfidf"], models_dir / "tfidf_vectorizer.pkl")


def main():
    parser = argparse.ArgumentParser(description="Projet 11 - entrainement modelisation NLP")
    parser.add_argument("--data-source", help="CSV local ou gdown://FILE_ID. Sinon DATASET_SOURCE.")
    parser.add_argument("--sample-per-class", type=int, default=200000)
    parser.add_argument("--mlflow-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="Projet11_Modelisation")
    parser.add_argument("--export-best", action="store_true")
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflow_uri)
    mlflow.set_experiment(args.experiment)
    reports_dir = ROOT / "reports" / "modelisation_variantes"
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = stratified_sample(load_dataset(args.data_source), args.sample_per_class)
    train_df, val_df, test_df = split_data(data)
    x_train, y_train = train_df["text"], train_df["label"]
    x_val, y_val = val_df["text"], val_df["label"]
    x_test, y_test = test_df["text"], test_df["label"]

    variants = {
        "linearsvc_baseline": build_pipeline(LinearSVC(C=1.0), 2),
        "ensemble_vote_pondere": build_pipeline(
            WeightedSentimentEnsemble(
                estimators=[
                    ("linearsvc", LinearSVC(C=1.0)),
                    ("logreg", LogisticRegression(max_iter=1000, n_jobs=-1)),
                ],
                weights=[0.65, 0.35],
            ),
            2,
        ),
        "routage_confiance": build_pipeline(
            ConfidenceRouter(
                primary=LinearSVC(C=1.0),
                fallback=LogisticRegression(max_iter=1000, n_jobs=-1),
                threshold=0.35,
            ),
            2,
        ),
    }

    if not args.skip_tuning:
        variants["linearsvc_tuned"] = GridSearchCV(
            build_pipeline(LinearSVC(), 2),
            param_grid={"model__C": [0.5, 1.0, 1.5, 2.0], "tfidf__ngram_range": [(1, 2), (1, 3)]},
            scoring="f1_macro",
            cv=3,
            n_jobs=-1,
        )

    results = []
    best_name, best_model, best_f1 = None, None, -1.0
    for name, model in variants.items():
        start = time.perf_counter()
        model.fit(x_train, y_train)
        train_seconds = time.perf_counter() - start
        metrics = evaluate(model, x_val, y_val, x_test, y_test)
        metrics["train_seconds"] = train_seconds
        if name == "routage_confiance":
            metrics["fallback_rate"] = float(getattr(model.named_steps["model"], "last_fallback_rate_", 0.0))
        params = {"sample_per_class": args.sample_per_class, "variant": name}
        if hasattr(model, "best_params_"):
            params.update({f"best_{k}": v for k, v in model.best_params_.items()})
        log_run(name, model, metrics, params, reports_dir)
        results.append({"variant": name, **{k: v for k, v in metrics.items() if k != "report"}})
        if metrics["f1_test"] > best_f1:
            best_name, best_model, best_f1 = name, model, metrics["f1_test"]

    pd.DataFrame(results).to_csv(reports_dir / "comparaison_variantes.csv", index=False)
    if args.export_best:
        models_dir = ROOT / "models"
        models_dir.mkdir(exist_ok=True)
        export_pipeline_model(best_model, models_dir)
        print(f"Meilleur modele exporte : {best_name} F1={best_f1:.4f}")


if __name__ == "__main__":
    main()
