import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from tensorflow.keras.models import load_model
from PIL import Image
from huggingface_hub import hf_hub_download

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'project_dl_model')
REPORT_DIR = os.path.join(BASE_DIR, 'project_dl_report')

def download_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    files = [
        'project_dl_model_baseline.h5',
        'project_dl_model_optimized.h5',
        'project_dl_model_optimized_2.h5',
        'project_dl_model_mobilenetv2.h5'
    ]
    for fname in files:
        path = os.path.join(MODEL_DIR, fname)
        if not os.path.exists(path):
            hf_hub_download(
                repo_id='Ahyeonn/skinscan-models',
                filename=fname,
                local_dir=MODEL_DIR
            )

download_models()

st.set_page_config(
    page_title='스킨스캔',
    layout="wide"
)


st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

IMG_SIZE = 96
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
CLASS_NAMES_KR = ['멜라닌세포 모반', '흑색종', '검버섯', '기저세포 암종', '광선각화증', '혈관 병변', '피부섬유종']
CLASS_FULLNAMES = {
    'nv':    '멜라닌세포 모반 (Melanocytic Nevi)',
    'mel':   '흑색종 (Melanoma)',
    'bkl':   '검버섯 (Benign Keratosis)',
    'bcc':   '기저세포 암종 (Basal Cell Carcinoma)',
    'akiec': '광선각화증 (Actinic Keratoses)',
    'vasc':  '혈관 병변 (Vascular Lesions)',
    'df':    '피부섬유종 (Dermatofibroma)'
}

CLASS_INFO = {
    'nv':    '멜라닌세포 모반은 점세포가 피부 내에 증식한 양성 종양(점)입니다. 대부분 양성이지만 크기가 갑자기 커지거나 모양이 불규칙해지면 전문의 진단을 권장합니다.',
    'mel':   '흑색종은 멜라닌 세포에서 발생하는 피부암으로 악성도가 높습니다. 조기 발견 시 치료 예후가 좋으나 전이가 빠르므로 즉시 전문의 진단이 필요합니다.',
    'bkl':   '검버섯은 피부 노화와 관련된 양성 병변으로 악성으로 변할 가능성은 낮습니다. 미용 목적으로 제거하는 경우가 많습니다.',
    'bcc':   '기저세포 암종은 가장 흔한 피부암으로 천천히 자라고 전이가 드문 편입니다. 조기에 발견하면 치료가 용이하므로 전문의 진단을 권장합니다.',
    'akiec': '광선각화증은 자외선에 의한 전암성 병변으로 방치 시 편평세포암으로 진행될 수 있습니다. 조기 치료가 중요합니다.',
    'vasc':  '혈관 병변은 혈관의 비정상적인 증식으로 생기는 양성 병변입니다. 대부분 악성 변화 가능성이 낮으나 크기가 커지면 전문의 진단을 권장합니다.',
    'df':    '피부섬유종은 피부의 섬유조직이 증식한 양성 종양입니다. 악성 변화 가능성이 매우 낮으며 특별한 치료 없이 경과 관찰하는 경우가 많습니다.',
}

CLASS_RISK = {
    'nv':    ('낮음'),
    'mel':   ('높음'),
    'bkl':   ('낮음'),
    'bcc':   ('중간'),
    'akiec': ('중간'),
    'vasc':  ('낮음'),
    'df':    ('낮음')
}



@st.cache_resource
def load_models():
    models = {}
    for name, fname in [
        ('CNN Baseline',  'project_dl_model_baseline.h5'),
        ('CNN Optimized', 'project_dl_model_optimized.h5'),
        ('MobileNetV2',   'project_dl_model_mobilenetv2.h5')
    ]:
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            models[name] = load_model(path, compile=False)
    return models

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert('RGB')
    img_array = np.array(img)
    img_resized = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized.astype('float32') / 255.0
    return img_array, img_normalized

def compute_gradcam(model, img_array, class_idx):
    last_conv_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_name = layer.name
    if last_conv_name is None:
        return None

    inp = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = inp
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv_name:
            conv_out = x

    grad_model = tf.keras.Model(inputs=inp, outputs=[conv_out, x])
    img_tensor = tf.cast(img_array[np.newaxis, ...], tf.float32)
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap).numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap

def overlay_gradcam(original_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)

st.title('🔬 스킨스캔')
st.markdown('피부 사진 한 장으로 병변 종류를 AI가 분석해드립니다')
st.divider()

models = load_models()

if not models:
    st.error('모델 파일이 없습니다. project_dl_app.ipynb를 먼저 실행해 주세요.')
    st.stop()

tab1, tab2 = st.tabs(['피부 분석', '모델 성능 비교'])

with tab1:
    col_left, col_right = st.columns([1, 2])

    with col_left:
        st.subheader('이미지 업로드')
        uploaded_file = st.file_uploader('피부 이미지를 업로드하세요', type=['jpg', 'jpeg', 'png'])
        selected_model_name = st.selectbox('모델 선택', list(models.keys()))
        show_gradcam = st.checkbox('Grad-CAM 시각화', value=True)

    with col_right:
        if uploaded_file:
            original_img, preprocessed_img = preprocess_image(uploaded_file)
            model = models[selected_model_name]

            pred = model.predict(preprocessed_img[np.newaxis, ...], verbose=0)[0]
            pred_idx   = np.argmax(pred)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = pred[pred_idx]
            risk = CLASS_RISK[pred_class]

            st.subheader('결과')
            c1, c2, c3 = st.columns(3)
            c1.metric('예측 : ', CLASS_FULLNAMES[pred_class])
            c2.metric('신뢰도', f'{confidence*100:.1f}%')
            c3.metric('위험도', f'{risk}')
            st.progress(float(confidence))
            st.info(CLASS_INFO[pred_class])
            st.divider()

            img_c1, img_c2 = st.columns(2)
            with img_c1:
                st.image(original_img, caption='원본 이미지', use_container_width=True)
            if show_gradcam:
                with img_c2:
                    try:
                        heatmap = compute_gradcam(model, preprocessed_img, pred_idx)
                        if heatmap is not None:
                            overlay = overlay_gradcam(original_img, heatmap)
                            st.image(overlay, caption='Grad-CAM', use_container_width=True)
                        else:
                            st.info('이 모델은 Grad-CAM을 지원하지 않습니다.')
                    except Exception as e:
                        st.warning(f'Grad-CAM 생성 실패: {e}')

            st.subheader('병변 종류별 가능성')
            fig, ax = plt.subplots(figsize=(8, 3))
            bar_colors = ['red' if i == pred_idx else 'blue' for i in range(len(CLASS_NAMES))]
            ax.barh(CLASS_NAMES_KR, pred * 100, color=bar_colors)
            ax.set_xlabel('확률 (%)')
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

with tab2:
    st.subheader('모델 학습 결과')

    report_imgs = [
        ('학습 곡선', 'project_dl_report_history.png',
        '학습이 진행될수록 정확도는 올라가고 손실은 내려가야 정상입니다. 실선은 훈련 데이터, 점선은 검증 데이터이며 두 선의 차이가 크면 과적합을 의심할 수 있습니다.'),
        ('모델 성능 비교', 'project_dl_report_comparison.png',
        '테스트 데이터로 평가한 최종 정확도입니다. CNN Optimized v2가 63.5%로 가장 높았으며, 과도한 Dropout을 적용한 CNN Optimized는 학습이 제대로 이루어지지 않아 낮은 성능을 보였습니다.'),
        ('Confusion Matrix - Baseline', 'project_dl_report_cm_cnn_baseline.png',
        '행은 실제 병변, 열은 모델이 예측한 병변입니다. 대각선 숫자가 클수록 정확하게 맞춘 것이며, 대각선 외 숫자는 잘못 분류된 경우입니다.'),
        ('Confusion Matrix - Optimized', 'project_dl_report_cm_cnn_optimized.png',
        'Dropout을 모든 레이어에 과도하게 적용한 결과 학습이 제대로 되지 않아 대부분의 병변을 nv(멜라닌세포 모반)로만 예측하는 경향을 보입니다.'),
        ('Confusion Matrix - Optimized v2', 'project_dl_report_cm_cnn_optimized_v2.png',
        'Dropout을 마지막 레이어에만 0.3으로 줄인 결과 전체적으로 균형있게 분류하며 가장 높은 성능을 달성했습니다.'),
        ('Confusion Matrix - MobileNetV2', 'project_dl_report_cm_mobilenetv2_ft.png',
        'ImageNet 기반 사전학습 모델로, 일반 사물 이미지에 최적화되어 있어 피부 병변의 미세한 특징을 충분히 학습하지 못한 것으로 분석됩니다.'),
        ('Grad-CAM - Optimized v2', 'project_dl_report_gradcam_optimized_v2.png',
        '모델이 병변을 판단할 때 실제로 어느 부위를 봤는지 시각화한 결과입니다. 빨간색에 가까울수록 모델이 집중한 부위이며, CNN Optimized v2는 병변 중심부에 집중하는 경향을 보입니다.'),
        ('Grad-CAM - MobileNetV2', 'project_dl_report_gradcam_mobilenetv2.png',
        'MobileNetV2는 히트맵이 전체적으로 퍼져있어 병변 부위를 명확하게 인식하지 못한 것을 확인할 수 있습니다.'),
        ('클래스 분포', 'project_dl_report_class_dist.png',
        '전체 이미지의 67%가 nv(멜라닌세포 모반)에 편중되어 있습니다. 이러한 클래스 불균형을 보정하기 위해 class weight를 적용하여 학습했습니다.'),
        ('샘플 이미지', 'project_dl_report_samples.png',
        '각 병변 종류의 실제 이미지 샘플입니다. 병변마다 색상과 형태가 다르며, 일부 병변은 육안으로도 구분이 어려울 만큼 유사한 특징을 가집니다.'),
    ]

    for title, fname, desc in report_imgs:
        path = os.path.join(REPORT_DIR, fname)
        if os.path.exists(path):
            st.markdown(f'**{title}**')
            st.caption(desc)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(path, use_container_width=True)
            st.divider()

with st.sidebar:
    st.markdown("## 사용법")
    st.markdown("1. 피부 이미지 업로드")
    st.markdown("2. 결과 확인")
    st.markdown("모델 선택으로 결과가 어떻게 달라지는지 비교해보세요")
    st.markdown("Grad-CAM 시각화는 AI가 이미지의 어느 부분을 보고 판단했는지 표시해줍니다")
    st.markdown("📊 **모델 비교**")
    st.markdown("CNN Baseline, CNN Optimized v2, and MobileNetV2.")
    st.divider()
    st.warning("학습용으로 제작된 AI 모델입니다. 의료적 판단이나 진단을 위한 용도로 사용될 수 없습니다.")
    st.divider()
    st.markdown("[![GitHub](https://img.shields.io/badge/GitHub-choi--ahyeon-181717?style=flat&logo=github)](https://github.com/choi-ahyeon)")
