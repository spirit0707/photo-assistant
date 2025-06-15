import cv2
import numpy as np

def draw_grid(image, grid_type="Rule of Thirds"):
    height, width = image.shape[:2]
    overlay = image.copy()

    color = (0, 255, 255)
    thickness = 1

    if grid_type == "Rule of Thirds":
        for i in [1, 2]:
            cv2.line(overlay, (int(width * i / 3), 0), (int(width * i / 3), height), color, thickness)
            cv2.line(overlay, (0, int(height * i / 3)), (width, int(height * i / 3)), color, thickness)

    elif grid_type == "Golden Ratio":
        phi = 0.618
        x = int(width * phi)
        y = int(height * phi)
        cv2.line(overlay, (x, 0), (x, height), color, thickness)
        cv2.line(overlay, (0, y), (width, y), color, thickness)

    elif grid_type == "Center":
        cv2.line(overlay, (width // 2, 0), (width // 2, height), color, thickness)
        cv2.line(overlay, (0, height // 2), (width, height // 2), color, thickness)

    elif grid_type == "None":
        return image

    return overlay

def apply_genre_filter(image, genre_name):
    genre = genre_name.lower()
    img = image.copy()
    if 'портрет' in genre:
        # размытие фона
        img = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(image, 1.5, img, -0.5, 0)
    elif 'пейзаж' in genre:
        # зелёный и синий в плюс
        img[:,:,1] = cv2.add(img[:,:,1], 20)  
        img[:,:,0] = cv2.add(img[:,:,0], 20) 
    elif 'животн' in genre:
        kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
        img = cv2.filter2D(img, -1, kernel)
    elif 'техника' in genre:
        cold = np.full(img.shape, (60, 30, 0), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.8, cold, 0.2, 0)
    elif 'архитектура' in genre or 'интерьер' in genre:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    elif 'улиц' in genre:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        cold = np.full(img.shape, (40, 20, 0), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.9, cold, 0.1, 0)
    return img

def apply_scene_filter(image, scene_name):
    scene = scene_name.lower()
    img = image.copy()

    nature_keywords = [
        'forest', 'creek','park', 'mountain', 'lake', 'field/wild', 'river', 'valley', 'field', 'meadow', 'garden', 'beach'
    ]
    if any(k in scene for k in nature_keywords):
        # мягкое повышение контраста
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.equalizeHist(l)
        lab = cv2.merge((l, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        # теплый тон
        warm = np.full(img.shape, (10, 30, 60), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.85, warm, 0.15, 0)
        # виньетка
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols/2)
        kernel_y = cv2.getGaussianKernel(rows, rows/2)
        kernel = kernel_y * kernel_x.T
        mask = 255 * kernel / np.max(kernel)
        for i in range(3):
            vignette = (img[:,:,i] * (mask/255)).astype(np.uint8)
            img[:,:,i] = cv2.addWeighted(img[:,:,i], 0.9, vignette, 0.1, 0)
        # приглушенные цвета
        img = cv2.addWeighted(img, 0.85, np.zeros_like(img), 0.15, 20)
        return img
    elif 'beach' in scene:
        # теплый фильтр
        warm = np.full(img.shape, (0, 30, 60), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.8, warm, 0.2, 0)
    elif 'forest' in scene or 'park' in scene:
        img[:,:,1] = cv2.add(img[:,:,1], 30)
    elif 'street' in scene or 'road' in scene or 'city' in scene:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl,a,b))
        img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    elif 'mountain' in scene:
        cold = np.full(img.shape, (60, 30, 0), dtype=np.uint8)
        img = cv2.addWeighted(img, 0.8, cold, 0.2, 0)
    elif 'kitchen' in scene or 'room' in scene or 'interior' in scene:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = cv2.add(v, 30)
        img = cv2.cvtColor(cv2.merge((h,s,v)), cv2.COLOR_HSV2BGR)
    return img
