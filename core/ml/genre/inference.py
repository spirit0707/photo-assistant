import os
import joblib
import numpy as np
from core.features.extractors import extract_features

GENRE_MODEL_PATH = os.path.join('core', 'ml', 'genre', 'genre_classifier.pkl')
_genre_clf = None

def get_genre_clf():
    global _genre_clf
    if _genre_clf is None:
        if not os.path.exists(GENRE_MODEL_PATH):
            raise FileNotFoundError("Файл genre_classifier.pkl не найден. Сначала обучите модель.")
        _genre_clf = joblib.load(GENRE_MODEL_PATH)
    return _genre_clf

def predict_genre_ml(image, image_cv):
    features = extract_features(image_cv, image).reshape(1, -1)
    clf = get_genre_clf()
    genre = clf.predict(features)[0]
    return genre 