import numpy as np
import cv2
from core.analysis.brightness import analyze_brightness
from core.analysis.sharpness import analyze_sharpness
from core.analysis.contrast import analyze_contrast
from core.analysis.saturation import analyze_saturation
from core.detection.yolo_detector import detect_categories, detect_objects
from core.analysis.composition import analyze_composition

GENRE_LABELS = [
    'Портрет', 'Пейзаж', 'Животные', 'Техника', 'Улица', 'Другое'
]
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRE_LABELS)}
GENRE_MODEL_TO_RU = {
    'portrait': 'Портрет',
    'landscape': 'Пейзаж',
    'animal': 'Животные',
    'tech': 'Техника',
    'street': 'Улица',
    'other': 'Другое'
}

def extract_features(image_cv, image_pil=None):
    brightness = analyze_brightness(image_cv)
    sharpness = analyze_sharpness(image_cv)
    contrast = analyze_contrast(image_cv)
    saturation = analyze_saturation(image_cv)
    mean_color = cv2.mean(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))[:3]

    if np.std(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)) < 2:
        saturation = 0.0

    img_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    (R, G, B) = cv2.split(img_rgb)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    colorfulness = np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))

    std_rg = np.std(R - G)
    std_rb = np.std(R - B)
    std_gb = np.std(G - B)

    try:
        detected_boxes = detect_objects(image_cv)
        has_human, has_animal, has_tech, humans, animals, tech = detect_categories(detected_boxes)
    except Exception:
        detected_boxes = []
        has_human, has_animal, has_tech, humans, animals, tech = detect_categories([])
    n_humans = len(humans) if humans is not None else 0
    n_animals = len(animals) if animals is not None else 0
    n_tech = len(tech) if tech is not None else 0

    h, w = image_cv.shape[:2]
    center_point = (w // 2, h // 2)
    min_dist_center = 1.0
    if detected_boxes:
        dists_center = []
        for box in detected_boxes:
            _, _, (x1, y1, x2, y2) = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dists_center.append(np.hypot(cx - center_point[0], cy - center_point[1]))
        min_dist_center = min(dists_center) / np.hypot(w, h)  # нормируем
    else:
        min_dist_center = 1.0

    thirds_points = [
        (w // 3, h // 3), (w // 3, 2 * h // 3),
        (2 * w // 3, h // 3), (2 * w // 3, 2 * h // 3)
    ]
    min_dist_thirds = 1.0
    if detected_boxes:
        dists = []
        for box in detected_boxes:
            _, _, (x1, y1, x2, y2) = box
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            dists_obj = [np.hypot(cx - tx, cy - ty) for (tx, ty) in thirds_points]
            dists.append(min(dists_obj))
        min_dist_thirds = min(dists) / np.hypot(w, h)  # нормируем
    else:
        min_dist_thirds = 1.0


    grid_sharpness = []
    grid_size = 3
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    gh, gw = gray.shape[0] // grid_size, gray.shape[1] // grid_size
    for i in range(grid_size):
        for j in range(grid_size):
            patch = gray[i*gh:(i+1)*gh, j*gw:(j+1)*gw]
            if patch.size > 0:
                grid_sharpness.append(np.var(cv2.Laplacian(patch, cv2.CV_64F)))
    sharp_grid_mean = np.mean(grid_sharpness)
    sharp_grid_std = np.std(grid_sharpness)
    sharp_grid_min = np.min(grid_sharpness)
    sharp_grid_max = np.max(grid_sharpness)

    has_object = int(len(detected_boxes) > 0)

    try:
        if len(detected_boxes) > 0:
            comp_score, _, _ = analyze_composition(image_cv, detected_boxes, grid_type='rule_of_thirds')
        else:
            comp_score = -1  
    except Exception:
        comp_score = -1

    genre_idx = GENRE_TO_IDX['Другое']

    hist = cv2.calcHist([gray], [0], None, [3], [0, 256]).flatten()
    hist = hist / (gray.shape[0] * gray.shape[1])

    features = [
        brightness, contrast, saturation,
        *mean_color,
        colorfulness, std_rg, std_rb, std_gb,
        sharp_grid_mean, sharp_grid_std, sharp_grid_min, sharp_grid_max,
        min_dist_center, min_dist_thirds,
        n_humans, n_animals, n_tech,
        genre_idx,
        *hist,
        has_object,
        comp_score
    ]
    return np.array(features)

__all__ = ["extract_features", "GENRE_LABELS", "GENRE_TO_IDX"] 