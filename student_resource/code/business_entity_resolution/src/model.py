"""Classifier training and precision-oriented threshold tuning."""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
import numpy as np
from .evaluate import macro_f05

FEATURE_NAMES = ["name_ratio", "name_token_jaccard", "name_exact", "name_phonetic", "address_ratio", "address_token_jaccard", "postal_exact", "country_exact", "name_length_delta", "address_length_delta", "name_token_count_delta", "address_token_count_delta"]


def fit_classifier(rows: List[Tuple[Dict[str, float], int]]) -> Any:
    from xgboost import XGBClassifier

    x = np.asarray([[row[name] for name in FEATURE_NAMES] for row, _ in rows], dtype=float)
    y = np.asarray([label for _, label in rows], dtype=int)
    model = XGBClassifier(n_estimators=220, max_depth=7, learning_rate=0.08,
                          subsample=0.85, colsample_bytree=0.85,
                          objective="binary:logistic", eval_metric="logloss",
                          tree_method="hist", n_jobs=4, random_state=42)
    model.fit(x, y)
    return model


def predict_probability(model, features: Dict[str, float]) -> float:
    values = np.asarray([[features[name] for name in FEATURE_NAMES]], dtype=float)
    return float(model.predict_proba(values)[0, 1])


def tune_threshold(scored: Dict[str, List[Tuple[str, float]]], truth: Dict[str, Set[str]]) -> Tuple[float, float]:
    best_threshold, best_score = 0.85, -1.0
    for threshold in np.linspace(0.50, 0.995, 100):
        predictions = {s1: {candidate for candidate, score in pairs if score >= threshold} for s1, pairs in scored.items()}
        score = macro_f05(truth, predictions)
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score
