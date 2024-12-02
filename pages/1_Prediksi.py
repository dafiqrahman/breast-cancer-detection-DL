import streamlit as st
import pandas as pd
import numpy as np
import pickle
from functions.predict import predict
import joblib

ROOT_PATH = "./"
# set to wide mode
st.set_page_config(layout="wide")
st.markdown("""
<style>
.center {
    text-align: center;

}
</style>
""", unsafe_allow_html=True)
st.markdown('<h2 class="center">Aplikasi Prediksi Kanker Payudara</h2>',
            unsafe_allow_html=True)

st.write('---')


@st.cache_resource
def load_model():
    model = joblib.load(ROOT_PATH + 'model/model_mlp.pkl')
    return model


model = load_model()
# inputs
col1, col2, col3, col4 = st.columns(4)
with col1:
    radius_mean = st.number_input(
        'Radius Mean', min_value=6., max_value=29., step=3.5, value=14.127292)
    texture_mean = st.number_input(
        'Texture Mean', min_value=9., max_value=40., step=4.301036, value=19.289649)
    smoothness_mean = st.number_input(
        'Smoothness Mean', min_value=0.052630, max_value=0.17, step=0.014064, value=0.096360)
    compactness_mean = st.number_input(
        'Compactness Mean', min_value=0.019380, max_value=0.345400, step=0.052813, value=0.104341)
    concavity_mean = st.number_input(
        'Concavity Mean', min_value=0., max_value=0.426800, step=0.079720, value=0.088799)

with col2:
    radius_se = st.number_input(
        'Radius SE', min_value=0.1, max_value=20.0, step=0.2, value=16.269190)
    texture_se = st.number_input(
        'Texture SE', min_value=0.0, max_value=35.0, step=0.5, value=25.677223)
    smoothness_se = st.number_input(
        'Smoothness SE', min_value=0.0, max_value=0.5, step=0.02, value=0.132369)
    compactness_se = st.number_input(
        'Compactness SE', min_value=0.0, max_value=0.5, step=0.02, value=0.254265)
    concavity_se = st.number_input(
        'Concavity SE', min_value=0.0, max_value=0.5, step=0.02, value=0.272188)


with col3:
    symmetry_mean = st.number_input(
        'Symmetry Mean', min_value=0.106000, max_value=0.304000, step=0.027414, value=0.181162)
    fractal_dimension_mean = st.number_input(
        'Fractal Dimension Mean', min_value=0.049960, max_value=0.097440, step=0.007060, value=0.083946)
    concavity_worst = st.number_input(
        'Concavity Worst', min_value=0.0, max_value=0.5, step=0.05, value=0.272188)
    symmetry_worst = st.number_input(
        'Symmetry Worst', min_value=0.1, max_value=0.4, step=0.05, value=0.290076)
    fractal_dimension_worst = st.number_input(
        'Fractal Dimension Worst', min_value=0.05, max_value=0.15, step=0.01, value=0.083946)
with col4:
    concave_points_se = st.number_input(
        'Concave Points SE', min_value=0.0, max_value=0.2, step=0.01, value=0.114606)
    symmetry_se = st.number_input(
        'Symmetry SE', min_value=0.0, max_value=0.3, step=0.05, value=0.290076)
    fractal_dimension_se = st.number_input(
        'Fractal Dimension SE', min_value=0.0, max_value=0.1, step=0.002, value=0.083946)
    smoothness_worst = st.number_input(
        'Smoothness Worst', min_value=0.1, max_value=0.3, step=0.02, value=0.132369)
    compactness_worst = st.number_input(
        'Compactness Worst', min_value=0.1, max_value=0.4, step=0.05, value=0.254265)


predict_button = st.button('Prediksi')

if predict_button:
    # create a dataframe
    df = pd.DataFrame({
        'radius_mean': radius_mean,
        'texture_mean': texture_mean,
        'smoothness_mean': smoothness_mean,
        'compactness_mean': compactness_mean,
        'concavity_mean': concavity_mean,
        'symmetry_mean': symmetry_mean,
        'fractal_dimension_mean': fractal_dimension_mean,
        'radius_se': radius_se,
        'texture_se': texture_se,
        'smoothness_se': smoothness_se,
        'compactness_se': compactness_se,
        'concavity_se': concavity_se,
        'concave points_se': concave_points_se,
        'symmetry_se': symmetry_se,
        'fractal_dimension_se': fractal_dimension_se,
        'smoothness_worst': smoothness_worst,
        'compactness_worst': compactness_worst,
        'concavity_worst': concavity_worst,
        'symmetry_worst': symmetry_worst,
        'fractal_dimension_worst': fractal_dimension_worst
    }, index=[0])

    st.write(df)

    pred = predict(model, df)
    sub_col1, sub_col2, sub_col3 = st.columns(3)
    with sub_col1:
        st.metric('Probabilitas Kanker :', pred[0][0])
    with sub_col2:
        st.metric('Probabilitas Sehat :', pred[0][1])
    with sub_col3:
        if np.argmax(pred) == 0:
            st.metric('Prediksi :', 'Kanker')
        else:
            st.metric('Prediksi :', 'Sehat')
