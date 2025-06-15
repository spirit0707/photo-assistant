import streamlit as st
import cv2
import numpy as np
from PIL import Image
from core.analysis.brightness import analyze_brightness
from core.analysis.sharpness import analyze_sharpness
from core.analysis.contrast import analyze_contrast
from core.analysis.saturation import analyze_saturation
from core.analysis.overlay import draw_grid, apply_genre_filter, apply_scene_filter
from core.analysis.composition import analyze_composition
from core.detection.yolo_detector import detect_objects, detect_categories
from core.analysis.pose import analyze_pose, draw_pose
from core.analysis.blur_detection import detect_blur
from core.ml.genre.inference import predict_genre_ml
from core.features.extractors import extract_features, GENRE_LABELS, GENRE_TO_IDX, GENRE_MODEL_TO_RU
from core.ml.quality.inference import predict_quality_ml, explain_quality, predict_quality_cnn
from core.visualization.cluster_viz import visualize_clusters
from core.ml.clustering.inference import fit_kmeans, normalize_features
from core.ml.scene.inference import classify_scene
import scipy.stats

st.set_page_config(page_title="Ассистент по композиции фото", layout="wide", page_icon="📷")

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown("""
    <script>
    document.body.classList.remove('theme-dark', 'theme-light');
    document.body.classList.add('theme-light');
    </script>
""", unsafe_allow_html=True)

analyze_tab, cluster_tab = st.tabs(["Анализ", "Кластеризация"])

with analyze_tab:
    st.sidebar.header("Опции анализа")
    analyze_brightness_opt = st.sidebar.checkbox("Яркость", True)
    analyze_sharpness_opt = st.sidebar.checkbox("Резкость", True)
    analyze_contrast_opt = st.sidebar.checkbox("Контраст", True)
    analyze_saturation_opt = st.sidebar.checkbox("Насыщенность", True)
    analyze_pose_opt = st.sidebar.checkbox("Анализ позы", False)
    analyze_composition_opt = st.sidebar.checkbox("Композиция", value=True)
    apply_scene_filter_opt = st.sidebar.checkbox("Применять фильтр по жанру", value=False)
    apply_scene_filter_scene_opt = st.sidebar.checkbox("Применять фильтр по сцене", value=False)

    ml_analysis_type = st.sidebar.radio(
        "ML-анализ:",
        ["YOLO (детекция объектов)"]
    )

    grid_type = st.sidebar.selectbox("Сетка наложения", ["Правило третей", "Золотое сечение", "Центр", "Нет"], index=3)

    grid_type_map = {
        "Правило третей": "Rule of Thirds",
        "Золотое сечение": "Golden Ratio",
        "Центр": "Center",
        "Нет": "None"
    }

    input_mode = st.sidebar.radio("Источник изображения:", ["Загрузить файл", "Веб-камера"])
    quality_model_type = st.sidebar.radio("Оценка качества:", ["ML (признаки)", "CNN (ResNet)"])

    image = None
    uploaded_file = None
    camera_image = None

    if input_mode == "Загрузить файл":
        uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    elif input_mode == "Веб-камера":
        camera_image = st.camera_input("Сделайте снимок для анализа")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")

    st.markdown("## 📷 Ассистент по фотографии")
    st.markdown("Загрузите фотографию для анализа.")

    def show_metrics(metrics):
        items = list(metrics.items())
        for i in range(0, len(items), 3):
            cols = st.columns(3)
            for j, (name, (display, _)) in enumerate(items[i:i+3]):
                with cols[j]:
                    st.markdown(f"<div class='metric'><h3>{name}</h3><p>{display}</p></div>", unsafe_allow_html=True)

    if image is not None:
        image_np = np.array(image)
        image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        detected_boxes = detect_objects(image_cv)

        metrics = {}

        if analyze_brightness_opt:
            val = analyze_brightness(image_cv)
            metrics["Яркость"] = (f"{val:.2f}", val)
        if analyze_sharpness_opt:
            val = analyze_sharpness(image_cv)
            metrics["Резкость"] = (f"{val:.2f}", val)
        if analyze_contrast_opt:
            val = analyze_contrast(image_cv)
            metrics["Контраст"] = (f"{val:.2f}", val)
        if analyze_saturation_opt:
            val = analyze_saturation(image_cv)
            metrics["Насыщенность"] = (f"{val:.2f}", val)

        # ML-метрики
        is_blur, sharpness_blur = detect_blur(image_cv)
        metrics["Размытие"] = (f"{'Да' if is_blur else 'Нет'} (резкость: {sharpness_blur:.1f})", sharpness_blur)
        has_human, has_animal, has_tech, humans, animals, tech = detect_categories(detected_boxes)

        # Классификация сцены и жанра
        scene_name, scene_prob = classify_scene(image)
        metrics["Сцена (Places365)"] = (f"{scene_name} ({scene_prob:.2%})", scene_prob)
        try:
            genre = predict_genre_ml(image, image_cv)
            genre_ru = GENRE_MODEL_TO_RU.get(genre, 'Другое')
            metrics["Жанр"] = (genre_ru, 1)
        except Exception as e:
            genre = f"Ошибка: {e}"
            genre_ru = 'Другое'
            metrics["Жанр"] = (genre, 0)

        try:
            if quality_model_type == "ML (признаки)":
                quality_score = predict_quality_ml(image_cv, image)
                metrics["Качество (ML)"] = (f"{quality_score:.2f} / 10", quality_score)
                features = extract_features(image_cv, image)
                has_object = features[23] 
                explanations = explain_quality(features)
                if not has_object:
                    explanations = [exp for exp in explanations if "композици" not in exp.lower() and "объект" not in exp.lower()]
            else:
                quality_score = predict_quality_cnn(image, image_cv)
                metrics["Качество (CNN)"] = (f"{quality_score:.2f} / 10", quality_score)
        except Exception as e:
            metrics["Качество (ML)"] = (f"Ошибка: {e}", 0)

        # Анализ композиции
        if analyze_composition_opt:
            grid_type_eng = grid_type_map.get(grid_type, "None")
            grid_type_for_composition = grid_type_eng.lower().replace(" ", "_")
            final_score, composition_tips, image_cv = analyze_composition(
                image_cv, detected_boxes, grid_type=grid_type_for_composition
            )
            metrics["Оценка композиции"] = (f"{final_score:.1f} / 100", final_score)
        
        pose_suggestions = None
        if analyze_pose_opt:
            image_cv = draw_pose(image_cv)
            pose_suggestions = analyze_pose(image_cv)

        col1, col2 = st.columns([1, 1])
        with col1:
            show_metrics(metrics)
            if quality_model_type == "ML (признаки)" and 'explanations' in locals():
                st.markdown("### Почему такая оценка?")
                for exp in explanations:
                    st.markdown(f"<div class='suggestion'>{exp}</div>", unsafe_allow_html=True)
            if analyze_composition_opt:
                if 'composition_tips' in locals() and composition_tips and "На изображении не обнаружено объектов для анализа композиции." not in composition_tips:
                    st.markdown("### Советы по композиции")
                    for tip in composition_tips:
                        st.markdown(f"<div class='suggestion'>{tip}</div>", unsafe_allow_html=True)
                else:
                    st.markdown("### Советы по композиции")
                    st.info("На изображении не обнаружено объектов для анализа композиции.")
            if analyze_pose_opt:
                if 'pose_suggestions' in locals() and pose_suggestions:
                    st.markdown("### Советы по позе")
                    for tip in pose_suggestions:
                        st.markdown(f"<div class='suggestion'>{tip}</div>", unsafe_allow_html=True)
        with col2:
            image_cv_disp = image_cv.copy()
            if apply_scene_filter_opt and genre:
                image_cv_disp = apply_genre_filter(image_cv_disp, genre_ru)
            if apply_scene_filter_scene_opt and scene_name:
                image_cv_disp = apply_scene_filter(image_cv_disp, scene_name)
            image_cv_disp = draw_grid(image_cv_disp, grid_type_map.get(grid_type, "None"))
            final_image = cv2.cvtColor(image_cv_disp, cv2.COLOR_BGR2RGB)
            st.image(final_image, caption="Анализированное изображение", use_container_width=True)

    else:
        st.info("📤 Загрузите изображение или сделайте снимок для начала анализа.")

with cluster_tab:
    st.markdown("## Кластеризация изображений по признакам")
    st.markdown("""
    Загрузите несколько изображений (jpg, png). Будет выполнена кластеризация по признакам (яркость, резкость, контраст, насыщенность, средний цвет, объекты, жанр, гистограмма).
    Выберите число кластеров, затем нажмите 'Кластеризовать'.
    """)
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = None
    if 'cluster_results' not in st.session_state:
        st.session_state.cluster_results = None

    uploaded_files = st.file_uploader("Загрузите изображения для кластеризации", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="cluster_upload")
    n_clusters = st.number_input("Число кластеров", min_value=2, max_value=10, value=3, step=1)
    cluster_btn = st.button("Кластеризовать")
    clear_btn = st.button("Очистить")

    if clear_btn:
        st.session_state.uploaded_files = None
        st.session_state.cluster_results = None
        st.rerun()

    if uploaded_files:
        st.session_state.uploaded_files = uploaded_files
    else:
        uploaded_files = st.session_state.uploaded_files

    if uploaded_files and cluster_btn:
        images = []
        features_list = []
        filenames = []
        for file in uploaded_files:
            try:
                image = Image.open(file).convert("RGB")
                image_np = np.array(image)
                image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                features = extract_features(image_cv, image)
                try:
                    genre = predict_genre_ml(image, image_cv)
                    genre_ru = GENRE_MODEL_TO_RU.get(genre, 'Другое')
                    features[19] = GENRE_TO_IDX.get(genre_ru, GENRE_TO_IDX['Другое'])
                except Exception:
                    features[19] = GENRE_TO_IDX['Другое']
                images.append(image)
                features_list.append(features)
                filenames.append(file.name)
            except Exception as e:
                st.warning(f"Ошибка при обработке {file.name}: {e}")
        features_arr = np.vstack(features_list)
        normed_features, scaler = normalize_features(features_arr)
        kmeans = fit_kmeans(normed_features, n_clusters=n_clusters, save_path="core/ml/clustering/kmeans_model.pkl")
        labels = kmeans.labels_
        st.success(f"Кластеризация завершена. Модель сохранена в core/ml/clustering/kmeans_model.pkl")

        plot_path = "core/ml/clustering/cluster_plot.png"
        visualize_clusters(normed_features, labels, save_path=plot_path)
        st.image(plot_path, caption="Визуализация кластеров (PCA)", use_container_width=True)

        st.markdown("### Результаты кластеризации:")
        for cluster_id in range(n_clusters):
            idxs = [i for i, lbl in enumerate(labels) if lbl == cluster_id]
            if not idxs:
                continue
            mean_feat = features_arr[idxs].mean(axis=0)
            brightness = mean_feat[0]
            contrast = mean_feat[1]
            saturation = mean_feat[2]
            mean_r, mean_g, mean_b = mean_feat[3:6]
            colorfulness, std_rg, std_rb, std_gb = mean_feat[6:10]
            sharp_grid_mean, sharp_grid_std, sharp_grid_min, sharp_grid_max = mean_feat[10:14]
            min_dist_center, min_dist_thirds = mean_feat[14:16]
            n_humans = int(np.sum(features_arr[idxs, 16]))
            n_animals = int(np.sum(features_arr[idxs, 17]))
            n_tech = int(np.sum(features_arr[idxs, 18]))
            # жанр по моде
            genre_idxs = [int(features_arr[idx, 19]) for idx in idxs]
            genre_idx = int(scipy.stats.mode(genre_idxs, keepdims=False).mode)
            hist_bins = mean_feat[20:23]
            has_object = int(np.sum(features_arr[idxs, 23]))
            comp_score = mean_feat[24]

            color_desc = f"R={mean_r:.0f}, G={mean_g:.0f}, B={mean_b:.0f}"
            genre_desc = GENRE_LABELS[genre_idx] if genre_idx < len(GENRE_LABELS) else "Другое"
            hist_desc = f"Тени: {hist_bins[0]:.2f}, Средние: {hist_bins[1]:.2f}, Света: {hist_bins[2]:.2f}"
            desc = (
                f"Яркость: {brightness:.1f}, Резкость: {sharp_grid_mean:.1f}, Контраст: {contrast:.1f}, "
                f"Насыщенность: {saturation:.1f}, Цвет: {color_desc}, "
                f"Людей: {n_humans}, Животных: {n_animals}, Техники: {n_tech}, "
                f"Жанр: {genre_desc}, {hist_desc}"
            )
            st.markdown(f"#### Кластер {cluster_id+1}: \n {desc}")
            cols = st.columns(5)
            for i, idx in enumerate(idxs):
                with cols[i % 5]:
                    st.image(images[idx], caption=filenames[idx], use_container_width=True)
    elif cluster_btn:
        st.warning("Пожалуйста, загрузите хотя бы одно изображение.")

