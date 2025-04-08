import streamlit as st
import pandas as pd
from joblib import load
import numpy as np

# 加载模型和标准化器
gbdt_model = load('gbdt_model.joblib')
scaler = load('scaler.joblib')

# 定义 Streamlit 页面布局
st.title("ILD 分级预测")
st.markdown("请根据以下选项输入数据，预测 ILD 分级。")

# 用户输入
alt = st.number_input('ALT（单位：U/L）', min_value=0.0)
erythrocyte_sedimentation = st.number_input('血沉（单位：mm/h）', min_value=0.0)
albumin = st.number_input('白蛋白（单位：g/L）', min_value=0.0)
antibody = st.selectbox('抗合成酶抗体阳性', [0, 1])
hemoglobin = st.number_input('血红蛋白（单位：g/L）', min_value=0.0)
triglyceride = st.number_input('甘油三酯（单位：mmol/L）', min_value=0.0)

# 计算 ALT/血沉
alt_erythrocyte_sedimentation = alt / erythrocyte_sedimentation if erythrocyte_sedimentation != 0 else 0

# 输入数据
input_data = {
    'ALT': alt,
    '血沉': erythrocyte_sedimentation,
    'ALT_÷_血沉': alt_erythrocyte_sedimentation,
    '白蛋白': albumin,
    '抗合成酶抗体阳性': antibody,
    '血红蛋白': hemoglobin,
    '甘油三酯': triglyceride
}

# 转换为 DataFrame 并进行标准化
X_new = pd.DataFrame([input_data])[['ALT_÷_血沉', '白蛋白', '抗合成酶抗体阳性', '血红蛋白', '甘油三酯']]
X_new_scaled = scaler.transform(X_new)

# 预测结果
prob = gbdt_model.predict_proba(X_new_scaled)[0][1]
result = "ILD分级为1级" if prob >= 0.5 else "ILD分级为0级"

# 显示结果
if st.button('进行预测'):
    st.write(f"预测结果：{result}")
    st.write(f"预测概率：{prob:.2f}")

