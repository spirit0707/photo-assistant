import os
import glob
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import cv2
from core.features.extractors import extract_features

DATASET_DIR = os.path.join(os.path.dirname(__file__), '../../data/dataset')
DATASET_DIR = os.path.abspath(DATASET_DIR)
if not os.path.isdir(DATASET_DIR):
    raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
GENRES = os.listdir(DATASET_DIR)

X = []
y = []

for genre in GENRES:
    genre_dir = os.path.join(DATASET_DIR, genre)
    for img_path in glob.glob(os.path.join(genre_dir, '*.jpg')):
        try:
            image = Image.open(img_path).convert('RGB')
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            features = extract_features(image_cv, image)
            X.append(features)
            y.append(genre)
        except Exception as e:
            print(f"Ошибка с {img_path}: {e}")

X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

print("Оценка на тесте:")
print(classification_report(y_test, clf.predict(X_test)))

joblib.dump(clf, 'core/ml/genre/genre_classifier.pkl')
print("Модель сохранена в core/ml/genre/genre_classifier.pkl") 