import streamlit as st
import os
import pandas as pd

SAMPLES_DIR = 'data/quality_samples'
LABELS_CSV = 'data/quality_labels.csv'

image_files = [f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
image_files.sort()

if os.path.exists(LABELS_CSV):
    labels_df = pd.read_csv(LABELS_CSV)
    labeled = set(labels_df['filename'])
else:
    labels_df = pd.DataFrame(columns=['filename', 'score'])
    labeled = set()

to_label = [f for f in image_files if f not in labeled]

st.title('Разметка качества фото')
st.markdown('Поставьте оценку (1-10) для каждого фото. Оценка — субъективное качество (композиция, резкость, свет, общее впечатление).')

if not to_label:
    st.success('Все фото размечены!')
    st.dataframe(labels_df)
else:
    filename = to_label[0]
    st.image(os.path.join(SAMPLES_DIR, filename), caption=filename, use_column_width=True)
    score = st.slider('Оценка качества', 1, 10, 5)
    if st.button('Сохранить оценку'):
        new_row = pd.DataFrame({'filename': [filename], 'score': [score]})
        labels_df = pd.concat([labels_df, new_row], ignore_index=True)
        labels_df.to_csv(LABELS_CSV, index=False)
        st.rerun() 