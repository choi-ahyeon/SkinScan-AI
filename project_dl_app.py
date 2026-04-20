import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["KERAS_BACKEND"] = "tensorflow"

import streamlit as st

st.set_page_config(
    page_title='스킨스캔',
    layout="wide"
)

import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tf_keras
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False
from PIL import Image
from huggingface_hub import hf_hub_download

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, 'project_dl_model')
REPORT_DIR = os.path.join(BASE_DIR, 'project_dl_report')

@st.cache_resource
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

with st.spinner('모델 다운로드 중... 잠시만 기다려주세요 ☕'):
    download_models()

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
