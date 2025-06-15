import cv2
from core.analysis.brightness import analyze_brightness
import numpy as np
from core.ml.clustering.inference import fit_kmeans, normalize_features
from PIL import Image
from core.ml.quality.inference import predict_quality_ml
import pytest

def test_brightness_on_sample():
    img = cv2.imread("test_image.jpg")
    value = analyze_brightness(img)
    assert 100 < value < 255 

def test_quality_ml():
    img = cv2.imread("portrait.jpg")
    pil_img = Image.open("portrait.jpg").convert("RGB")
    score = predict_quality_ml(img, pil_img)
    assert 0 <= score <= 10

def test_kmeans_clusters():
    features = np.random.rand(10, 25)
    normed, _ = normalize_features(features)
    kmeans = fit_kmeans(normed, n_clusters=3)
    assert len(set(kmeans.labels_)) == 3

def test_invalid_image():
    with pytest.raises(Exception):
        img = Image.open("1.txt")
        analyze_brightness(img)