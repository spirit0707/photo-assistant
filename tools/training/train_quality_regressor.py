import os
import pandas as pd
import numpy as np
from PIL import Image
from core.features.extractors import extract_features
from core.ml.clustering.inference import normalize_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import matplotlib.pyplot as plt
import cv2

SAMPLES_DIR = 'data/samples/quality_samples'
LABELS_CSV = 'data/labels/quality_labels.csv'
MODEL_PATH = 'core/ml/quality/quality_regressor.pkl'
PLOT_PATH = 'core/ml/quality/quality_pred_vs_true.png'

labels_df = pd.read_csv(LABELS_CSV)

X = []
y = []

for idx, row in labels_df.iterrows():
    filename = row['filename']
    score = row['score']
    img_path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(img_path):
        print(f"Пропущено: {filename} (нет файла)")
        continue
    try:
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        features = extract_features(image_cv, image)
        X.append(features)
        y.append(score)
    except Exception as e:
        print(f"Ошибка для {filename}: {e}")

if len(X) == 0:
    raise ValueError("Нет валидных изображений для обучения!")

X = np.vstack(X)
y = np.array(y)

X_norm, scaler = normalize_features(X)

reg = RandomForestRegressor(n_estimators=100, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(reg, X_norm, y, cv=kf)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"Кросс-валидация: MAE={mae:.3f}, RMSE={rmse:.3f}")

plt.figure(figsize=(6,6))
plt.scatter(y, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Идеал')
plt.xlabel('Реальное качество')
plt.ylabel('Предсказанное качество')
plt.title('Качество: предсказание vs. реальность (CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_PATH)
print(f"График сохранён в {PLOT_PATH}")

reg.fit(X_norm, y)

feature_names = [
    'Яркость',            
    'Контраст',           
    'Насыщенность',       
    'R',                  
    'G',                  
    'B',                  
    'Colorfulness',       
    'std(R-G)',           
    'std(R-B)',           
    'std(G-B)',           
    'SharpGridMean',      
    'SharpGridStd',       
    'SharpGridMin',      
    'SharpGridMax',       
    'CentralDist',       
    'ThirdsDist',         
    'Людей',              
    'Животных',          
    'Техники',            
    'Жанр',              
    'Тени',              
    'Средние',           
    'Света',             
    'Есть объект',       
    'Композиция'          
]
importances = reg.feature_importances_
print("feature_names:", len(feature_names), "importances:", len(importances))
plt.figure(figsize=(10, 5))
plt.barh(feature_names, importances)
plt.xlabel('Важность')
plt.title('Важность признаков для качества фото (RandomForest)')
plt.tight_layout()
plt.savefig('core/ml/quality/feature_importance.png')
print('График важности признаков сохранён в core/ml/quality/feature_importance.png')

joblib.dump({'regressor': reg, 'scaler': scaler}, MODEL_PATH)
print(f"Модель сохранена в {MODEL_PATH}") 
