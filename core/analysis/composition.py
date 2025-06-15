import cv2
from .overlay import draw_grid
from collections import defaultdict

def draw_object_boxes(image, objects, grid_points=None, main_object=None):
    overlay = image.copy()
    h, w, _ = image.shape
    for (label, conf, (x1, y1, x2, y2)) in objects:
        color = (0, 255, 0)
        thickness = 2
        draw_arrow = False
        if main_object and (label, conf, (x1, y1, x2, y2)) == main_object:
            cv2.rectangle(overlay, (x1-3, y1-3), (x2+3, y2+3), (0,0,0), 8)  # тень
            color = (0, 0, 255)
            thickness = 3
            draw_arrow = True  
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(overlay, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        if draw_arrow and grid_points and len(grid_points) > 0:
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            gx, gy = min(grid_points, key=lambda p: (cx - p[0])**2 + (cy - p[1])**2)
            cv2.arrowedLine(overlay, (cx, cy), (gx, gy), (0, 165, 255), 2, tipLength=0.15)
    return overlay


def select_main_object(objects, image_shape):
    h, w = image_shape[:2]
    center_x, center_y = w // 2, h // 2

    def score(obj):
        label, conf, (x1, y1, x2, y2) = obj
        area = (x2 - x1) * (y2 - y1)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dist = ((cx - center_x) ** 2 + (cy - center_y) ** 2) ** 0.5
        return area * conf - dist * 0.5

    if not objects:
        return None
    return max(objects, key=score)

def detect_rhythm(objects, w, h):
    rhythm_tips = []
    by_label = defaultdict(list)
    for (label, conf, (x1, y1, x2, y2)) in objects:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        by_label[label].append((cx, cy))
    for label, centers in by_label.items():
        if len(centers) < 3:
            continue
        centers_sorted_x = sorted(centers, key=lambda p: p[0])
        dx = [centers_sorted_x[i+1][0] - centers_sorted_x[i][0] for i in range(len(centers_sorted_x)-1)]
        if len(dx) > 1 and max(dx) > 0:
            mean_dx = sum(dx) / len(dx)
            if all(abs(d - mean_dx) < 0.2 * mean_dx for d in dx):
                rhythm_tips.append(f"На фотографии обнаружен горизонтальный ритм: повторяющиеся объекты класса '{label}' расположены на равных расстояниях.")
                continue
        centers_sorted_y = sorted(centers, key=lambda p: p[1])
        dy = [centers_sorted_y[i+1][1] - centers_sorted_y[i][1] for i in range(len(centers_sorted_y)-1)]
        if len(dy) > 1 and max(dy) > 0:
            mean_dy = sum(dy) / len(dy)
            if all(abs(d - mean_dy) < 0.2 * mean_dy for d in dy):
                rhythm_tips.append(f"На фотографии обнаружен вертикальный ритм: повторяющиеся объекты класса '{label}' расположены на равных расстояниях.")
    return rhythm_tips

def analyze_composition(image, objects, grid_type="rule_of_thirds"):
    h, w, _ = image.shape
    evaluation = []
    score = 0
    tips = set()

    if grid_type == "rule_of_thirds":
        grid_x = [w // 3, 2 * w // 3]
        grid_y = [h // 3, 2 * h // 3]
        grid_name = "правило третей"
        grid_points = [(gx, gy) for gx in grid_x for gy in grid_y]
    elif grid_type == "golden_ratio":
        phi = 0.618
        grid_x = [int(w * phi), int(w * (1 - phi))]
        grid_y = [int(h * phi), int(h * (1 - phi))]
        grid_name = "золотое сечение"
        grid_points = [(gx, gy) for gx in grid_x for gy in grid_y]
    elif grid_type == "center":
        grid_x = [w // 2]
        grid_y = [h // 2]
        grid_name = "центр"
        grid_points = [(gx, gy) for gx in grid_x for gy in grid_y]
    else:
        grid_x, grid_y = [], []
        grid_name = grid_type
        grid_points = []

    if not objects:
        evaluation.append("На изображении не обнаружено объектов для анализа композиции.")
        return 0, evaluation, image

    persons = [obj for obj in objects if obj[0] == 'person']
    if persons:
        main_object = select_main_object(persons, image.shape)
        objects_to_analyze = [main_object]
        other_objects = [obj for obj in objects if obj[0] != 'person']
    else:
        main_object = select_main_object(objects, image.shape)
        objects_to_analyze = [main_object]
        other_objects = [obj for obj in objects if obj != main_object]

    for (label, conf, (x1, y1, x2, y2)) in objects_to_analyze:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        margin = 0.07
        if cx < w * margin or cx > w * (1 - margin) or cy < h * margin or cy > h * (1 - margin):
            tips.add(f"Объект '{label}' слишком близко к краю кадра. Рекомендуется размещать важные объекты дальше от границ.")

        if grid_type == "center":
            center_x, center_y = w // 2, h // 2
            if abs(cx - center_x) < w * 0.15 and abs(cy - center_y) < h * 0.15:
                score += 1
            elif abs(cx - center_x) < w * 0.22 and abs(cy - center_y) < h * 0.22:
                tips.add(f"Объект '{label}' почти по центру, но можно расположить ещё точнее.")
            else:
                tips.add(f"Объект '{label}' не по центру. Попробуйте расположить его ближе к центру кадра.")
        else:
            near_strong = False
            for gx, gy in grid_points:
                if abs(cx - gx) < w * 0.12 and abs(cy - gy) < h * 0.12:
                    score += 1
                    near_strong = True
                    break
            if not near_strong and grid_points:
                for gx, gy in grid_points:
                    if abs(cx - gx) < w * 0.18 and abs(cy - gy) < h * 0.18:
                        tips.add(f"Объект '{label}' почти на сильной точке сетки ({grid_name}), но можно расположить ещё точнее.")
                        break
                else:
                    gx, gy = min(grid_points, key=lambda p: (cx - p[0])**2 + (cy - p[1])**2)
                    tips.add(f"Объект '{label}' не совпадает с сильными точками сетки ({grid_name}). Попробуйте расположить его ближе к точке ({gx}, {gy}).")

    rhythm_tips = detect_rhythm(objects, w, h)
    evaluation.extend(rhythm_tips)

    total_possible = len(objects_to_analyze)
    final_score = (score / total_possible * 100) if total_possible > 0 else 0

    if total_possible > 0 and score == total_possible:
        if grid_type == "center":
            evaluation.append("Отличная композиция! Главный объект хорошо расположен по центру.")
        else:
            evaluation.append(f"Отличная композиция! Главный объект хорошо размещён по сетке ({grid_name}).")
    else:
        if final_score > 60:
            evaluation.append("В целом композиция хорошая, но есть что улучшить:")
        elif final_score > 30:
            evaluation.append("Композиция требует доработки. Обратите внимание на следующие советы:")
        else:
            evaluation.append("Композиция неудачна. Рекомендуется пересмотреть расположение главного объекта:")
        evaluation.extend(sorted(tips))

    grid_image = draw_grid(image, grid_type)
    result_image = draw_object_boxes(grid_image, objects, grid_points=grid_points, main_object=main_object)

    return final_score, evaluation, result_image
