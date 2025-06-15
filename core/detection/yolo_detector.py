from ultralytics import YOLO

model = YOLO('yolov10n.pt')

HUMAN_CLASSES = {"person"}
ANIMAL_CLASSES = {"cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
TECH_CLASSES = {"car", "bicycle", "motorbike", "bus", "truck", "train", "boat", "airplane"}

def detect_objects(image):
    results = model(image)
    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            detections.append((label, conf, (x1, y1, x2, y2)))
    return detections

def detect_categories(yolo_boxes):
    humans = [obj for obj in yolo_boxes if obj[0] in HUMAN_CLASSES]
    animals = [obj for obj in yolo_boxes if obj[0] in ANIMAL_CLASSES]
    tech = [obj for obj in yolo_boxes if obj[0] in TECH_CLASSES]
    return bool(humans), bool(animals), bool(tech), humans, animals, tech
