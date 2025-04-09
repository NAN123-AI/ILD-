import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# 加载模型和标准化器
gbdt_model = load('gbdt_model.joblib')
scaler = load('scaler.joblib')

# 页面设置，确保页面在移动设备上友好显示
st.set_page_config(page_title="AIP-MDA5-SILD", page_icon="🧑‍⚕️", layout="centered")

# 页面标题和描述
st.title("AIP-MDA5-SILD")
st.markdown("""
    **AI-assisted prediction of severe interstitial lung disease associated with MDA5 positive dermatomyositis**
    
    请输入相关数据，系统将自动计算并预测 ILD 分级。  
    输入后，系统会实时更新并显示预测结果。
""")

# 使用 container 来确保布局居中
with st.container():
    # 使用单列排列所有输入框
    st.header("请输入数据")
    
    # 创建竖排的输入框
    alt = st.number_input('ALT（单位：U/L）', min_value=0.0, help="输入血清ALT水平，单位：U/L")
    erythrocyte_sedimentation = st.number_input('血沉（单位：mm/h）', min_value=0.0, help="输入红细胞沉降率，单位：mm/h")
    albumin = st.number_input('白蛋白（单位：g/L）', min_value=0.0, help="输入血清白蛋白水平，单位：g/L")
    antibody = st.selectbox('抗合成酶抗体阳性', [0, 1], help="选择是否抗合成酶抗体阳性（0=否，1=是）")
    hemoglobin = st.number_input('血红蛋白（单位：g/L）', min_value=0.0, help="输入血红蛋白水平，单位：g/L")
    triglyceride = st.number_input('甘油三酯（单位：mmol/L）', min_value=0.0, help="输入甘油三酯水平，单位：mmol/L")

# 计算 ALT/血沉比值
alt_erythrocyte_sedimentation = alt / erythrocyte_sedimentation if erythrocy_
