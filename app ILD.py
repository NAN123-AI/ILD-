import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# 加载模型和标准化器
gbdt_model = load('gbdt_model.joblib')
scaler = load('scaler.joblib')

# 页面设置，确保页面在移动设备上友好显示
st.set_page_config(page_title="ILD 分级预测", page_icon="🧑‍⚕️", layout="centered")

# 页面标题和描述
st.title("ILD 分级预测")
st.markdown("""
    请根据以下输入框输入相关数据，系统将自动计算并预测 ILD 分级。  
    输入后，系统将实时更新并显示预测结果。
""")

# 使用侧边栏来接受输入
with st.sidebar:
    st.header("输入数据")
    alt = st.number_input('ALT（单位：U/L）', min_value=0.0, help="输入血清ALT水平，单位：U/L")
    erythrocyte_sedimentation = st.number_input('血沉（单位：mm/h）', min_value=0.0, help="输入红细胞沉降率，单位：mm/h")
    albumin = st.number_input('白蛋白（单位：g/L）', min_value=0.0, help="输入血清白蛋白水平，单位：g/L")
    antibody = st.selectbox('抗合成酶抗体阳性', [0, 1], help="选择是否抗合成酶抗体阳性（0=否，1=是）")
    hemoglobin = st.number_input('血红蛋白（单位：g/L）', min_value=0.0, help="输入血红蛋白水平，单位：g/L")
    triglyceride = st.number_input('甘油三酯（单位：mmol/L）', min_value=0.0, help="输入甘油三酯水平，单位：mmol/L")

# 计算 ALT/血沉比值
alt_erythrocyte_sedimentation = alt / erythrocyte_sedimentation if erythrocyte_sedimentation != 0 else 0

# 输入数据整理
input_data = {
    'ALT': alt,
    '血沉': erythrocyte_sedimentation,
    'ALT_÷_血沉': alt_erythrocyte_sedimentation,
    '白蛋白': albumin,
    '抗合成酶抗体阳性': antibody,
    '血红蛋白': hemoglobin,
    '甘油三酯': triglyceride
}

# 数据转换为 DataFrame
X_new = pd.DataFrame([input_data])[['ALT_÷_血沉', '白蛋白', '抗合成酶抗体阳性', '血红蛋白', '甘油三酯']]

# 标准化数据
X_new_scaled = scaler.transform(X_new)

# 预测结果
prob = gbdt_model.predict_proba(X_new_scaled)[0][1]
result = "ILD分级为1级" if prob >= 0.5 else "ILD分级为0级"

# 显示计算结果
st.markdown("### 预测结果")
st.write(f"预测结果：{result}")
st.write(f"预测概率：{prob:.2f}")

# 美化布局，使用两列显示数据
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 输入数据")
    st.write(input_data)

with col2:
    st.markdown("### 计算过程")
    st.write(f"ALT / 血沉：{alt_erythrocyte_sedimentation:.2f}")
    st.write(f"标准化后的数据：{X_new_scaled}")

# 提供清晰的界面提示
st.markdown("### 使用提示")
st.markdown("""
    - 系统会自动根据你输入的数据进行计算，预测结果会即时更新。
    - 若有任何问题，可以随时调整输入，查看不同情况的预测结果。
""")
