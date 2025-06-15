import mediapipe as mp
import cv2
import numpy as np

mp_pose = mp.solutions.pose
mp_face = mp.solutions.face_mesh

def analyze_pose(image):
    h, w, _ = image.shape
    suggestions = []
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose, mp_face.FaceMesh(static_image_mode=True) as face:
        pose_results = pose.process(img_rgb)
        face_results = face.process(img_rgb)

        if pose_results.pose_landmarks:
            left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            dx = (right_shoulder.x - left_shoulder.x) * w
            dy = (right_shoulder.y - left_shoulder.y) * h
            angle = np.degrees(np.arctan2(dy, dx))
            if abs(angle) > 20:
                suggestions.append("Тело заметно наклонено. Постарайтесь держать плечи более горизонтально для сбалансированной композиции.")

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            face_center_x = (left_eye.x + right_eye.x) / 2
            if face_center_x < 0.4:
                suggestions.append("Человек смотрит вправо (возможно, за пределы кадра). Попробуйте направить взгляд внутрь кадра.")
            elif face_center_x > 0.6:
                suggestions.append("Человек смотрит влево (возможно, за пределы кадра). Попробуйте направить взгляд внутрь кадра.")
            else:
                suggestions.append("Взгляд направлен внутрь кадра — это хорошо для композиции.")

    return suggestions

def draw_pose(image):
    h, w, _ = image.shape
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output = image.copy()

    with mp_pose.Pose(static_image_mode=True) as pose, mp_face.FaceMesh(static_image_mode=True) as face:
        pose_results = pose.process(img_rgb)
        face_results = face.process(img_rgb)

        if pose_results.pose_landmarks:
            for connection in mp_pose.POSE_CONNECTIONS:
                start = pose_results.pose_landmarks.landmark[connection[0]]
                end = pose_results.pose_landmarks.landmark[connection[1]]
                x1, y1 = int(start.x * w), int(start.y * h)
                x2, y2 = int(end.x * w), int(end.y * h)
                cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), 3)
            for lm in pose_results.pose_landmarks.landmark:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(output, (cx, cy), 5, (0, 0, 255), -1)

        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            nose_tip = face_landmarks.landmark[1]
            eye_x = int(((left_eye.x + right_eye.x) / 2) * w)
            eye_y = int(((left_eye.y + right_eye.y) / 2) * h)
            nose_x = int(nose_tip.x * w)
            nose_y = int(nose_tip.y * h)
            cv2.arrowedLine(output, (eye_x, eye_y), (nose_x, nose_y), (255, 0, 0), 4, tipLength=0.3)
            cv2.circle(output, (eye_x, eye_y), 6, (255, 255, 0), -1)
            cv2.circle(output, (nose_x, nose_y), 6, (255, 0, 255), -1)

    return output