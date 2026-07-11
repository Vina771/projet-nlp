from pathlib import Path

import joblib
import numpy as np

from .text_processing import clean_and_tokenize


LABEL_NAMES = ["negative", "neutral", "positive"]


class SentimentModel:
    def __init__(self, model_path: Path, vectorizer_path: Path):
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.model = None
        self.vectorizer = None

    @property
    def ready(self) -> bool:
        return self.model is not None and self.vectorizer is not None

    def load(self):
        if not self.ready:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        return self

    def predict(self, text: str) -> dict:
        self.load()
        tokens = clean_and_tokenize(text)
        if not tokens:
            return {"label": "neutral", "score": 0.0, "tokens": ""}

        vector = self.vectorizer.transform([tokens])
        pred = int(self.model.predict(vector)[0])
        return {
            "label": LABEL_NAMES[pred],
            "score": self._score(vector, pred),
            "tokens": tokens,
        }

    def _score(self, vector, pred: int) -> float:
        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(vector)[0][pred])
        if hasattr(self.model, "decision_function"):
            decision = np.atleast_1d(self.model.decision_function(vector)[0])
            if len(decision) == 1:
                raw = float(decision[0])
                return float(1 / (1 + np.exp(-abs(raw))))
            exp = np.exp(decision - np.max(decision))
            proba = exp / exp.sum()
            return float(proba[pred])
        return 0.0


def default_model() -> SentimentModel:
    root = Path(__file__).resolve().parents[2]
    return SentimentModel(
        root / "models" / "best_model.pkl",
        root / "models" / "tfidf_vectorizer.pkl",
    )
