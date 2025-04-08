import streamlit as st
import pandas as pd
from joblib import load
import numpy as np
from flask import Flask
from io import BytesIO

# 初始化 Flask 应用
flask_app = Flask(__name__)

# 加载模型和标准化器
gbdt_model = load('gbdt_model.joblib')
scaler = load('scaler.joblib')

# Streamlit 页面配置
st.set_page_config(page_title="ILD 分级预测", page_icon="🧑‍⚕️", layout="wide")

# 页面标题和描述
st.title("ILD 分级预测")
st.markdown("""
    请根据以下输入框输入相关数据，系统将自动计算并预测 ILD 分级。  
    你可以通过输入**ALT**、**血沉**、**白蛋白**、**抗合成酶抗体阳性**等指标，  
    我们将自动计算结果并给出预测分级。
""")

# 使用侧边栏来接受输入
with st.sidebar:
    st.header("请输入数据")
    
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

# 在主页面展示数据的输入框和结果
st.markdown("### 输入数据")
st.write(input_data)

# 美化布局
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 计算过程")
    st.write(f"ALT / 血沉：{alt_erythrocyte_sedimentation:.2f}")
    st.write(f"标准化后的数据：{X_new_scaled}")

with col2:
    st.markdown("### 提示")
    st.markdown("""
        - 输入越准确，预测结果越可靠。
        - 在输入过程中，结果将会自动更新。
        - 你可以根据需要修改输入的参数来查看不同情况的预测结果。
    """)

# Flask API: 返回预测结果
@flask_app.route('/predict', methods=['POST'])
def predict():
    # 通过 POST 请求获取用户输入
    input_data = request.get_json()  # 获取请求数据
    df = pd.DataFrame([input_data])
    
    # 标准化并进行预测
    X_new_scaled = scaler.transform(df)
    prob = gbdt_model.predict_proba(X_new_scaled)[0][1]
    result = "ILD分级为1级" if prob >= 0.5 else "ILD分级为0级"
    
    # 返回预测结果
    return jsonify({"result": result, "probability": prob})

# 运行 Flask API
if __name__ == "__main__":
    flask_app.run(debug=True)
