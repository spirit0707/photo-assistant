import joblib
import numpy as np
from core.features.extractors import extract_features
import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2

def predict_quality_ml(image_cv, image_pil, model_path='core/ml/quality/quality_regressor.pkl'):
    model_data = joblib.load(model_path)
    reg = model_data['regressor']
    scaler = model_data['scaler']
    features = extract_features(image_cv, image_pil)
    features_norm = scaler.transform([features])
    score = reg.predict(features_norm)[0]
    return score

def explain_quality(features):
    explanations = []
    idx = {
        'contrast': 0,
        'saturation': 1,
        'colorfulness': 6,
        'central_dist': 14,
        'thirds_dist': 15,
        'comp_score': -1,  # последний
    }
    if features[idx['contrast']] < 20:
        explanations.append('Низкий контраст')
    if features[idx['saturation']] < 30 and features[1] != 0.0:
        explanations.append('Низкая насыщенность (для цветных фото)')
    if features[idx['colorfulness']] < 10:
        explanations.append('Слабая цветовая выразительность')
    if features[idx['central_dist']] > 0.3:
        explanations.append('Главный объект далеко от центра кадра')
    if features[idx['thirds_dist']] > 0.3:
        explanations.append('Главный объект далеко от точек правила третей')
    if features[idx['comp_score']] == -1:
        explanations.append('На фото не найдено объектов — композиция не анализировалась')
    elif features[idx['comp_score']] < 40:
        explanations.append('Композиция не соответствует основным правилам')
    if not explanations:
        explanations.append('Нет явных проблем — фото сбалансировано по основным признакам')
    return explanations 

CNN_MODEL_PATH = os.path.join('core', 'ml', 'quality', 'quality_cnn.pt')
_cnn_model = None
_cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
        model.eval()
        _cnn_model = model.to(device)
    return _cnn_model

def predict_quality_cnn(image_pil, image_cv=None):
    if not isinstance(image_pil, Image.Image):
        if isinstance(image_pil, np.ndarray):
            image_pil = Image.fromarray(cv2.cvtColor(image_pil, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError('image_pil должен быть PIL.Image или np.ndarray')
    x = _cnn_transform(image_pil).unsqueeze(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    model = get_cnn_model()
    with torch.no_grad():
        pred = model(x).cpu().numpy().flatten()[0]
    return float(pred) 