import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone


def _scores(estimator, x):
    if hasattr(estimator, "decision_function"):
        values = estimator.decision_function(x)
    elif hasattr(estimator, "predict_proba"):
        values = estimator.predict_proba(x)
    else:
        pred = estimator.predict(x)
        values = np.zeros((len(pred), len(estimator.classes_)))
        for row, label in enumerate(pred):
            values[row, int(label)] = 1.0

    values = np.asarray(values)
    if values.ndim == 1:
        values = np.column_stack([-values, values])
    return values


class WeightedSentimentEnsemble(BaseEstimator, ClassifierMixin):
    """Weighted LinearSVC + LogisticRegression ensemble compatible with joblib."""

    def __init__(self, estimators=None, weights=None):
        self.estimators = estimators or []
        self.weights = weights or []

    def fit(self, x, y):
        self.models_ = []
        for _, estimator in self.estimators:
            model = clone(estimator)
            model.fit(x, y)
            self.models_.append(model)
        self.classes_ = np.unique(y)
        return self

    def decision_function(self, x):
        weights = np.asarray(self.weights or [1.0] * len(self.models_), dtype=float)
        weights = weights / weights.sum()
        combined = None
        for weight, model in zip(weights, self.models_):
            current = _scores(model, x)
            combined = current * weight if combined is None else combined + current * weight
        return combined

    def predict(self, x):
        scores = self.decision_function(x)
        return self.classes_[np.argmax(scores, axis=1)]


class ConfidenceRouter(BaseEstimator, ClassifierMixin):
    """Use LinearSVC first, then LogisticRegression only for low-margin cases."""

    def __init__(self, primary=None, fallback=None, threshold=0.35):
        self.primary = primary
        self.fallback = fallback
        self.threshold = threshold

    def fit(self, x, y):
        self.primary_ = clone(self.primary)
        self.fallback_ = clone(self.fallback)
        self.primary_.fit(x, y)
        self.fallback_.fit(x, y)
        self.classes_ = np.unique(y)
        return self

    def _fallback_mask(self, x):
        scores = _scores(self.primary_, x)
        sorted_scores = np.sort(scores, axis=1)
        margins = sorted_scores[:, -1] - sorted_scores[:, -2]
        return margins < self.threshold

    def predict(self, x):
        predictions = self.primary_.predict(x)
        mask = self._fallback_mask(x)
        if mask.any():
            predictions[mask] = self.fallback_.predict(x[mask])
        self.last_fallback_rate_ = float(mask.mean())
        return predictions

    def decision_function(self, x):
        scores = _scores(self.primary_, x)
        mask = self._fallback_mask(x)
        if mask.any():
            scores[mask] = _scores(self.fallback_, x[mask])
        self.last_fallback_rate_ = float(mask.mean())
        return scores
