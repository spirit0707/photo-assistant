import cv2
 
def detect_blur(image_cv, threshold=100.0):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    is_blur = laplacian_var < threshold
    return is_blur, laplacian_var 