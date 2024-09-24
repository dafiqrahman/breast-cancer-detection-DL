import streamlit as st
import pandas as pd
import numpy as np
import pickle
from functions.predict import predict
import joblib

ROOT_PATH = "./"

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
col1, col2 = st.columns(2)
with col1:
    radius_mean = st.number_input(
        'Radius ', min_value=6., max_value=29., step=3.5, value=14.127292)
    texture_mean = st.number_input(
        'Texture', min_value=9., max_value=40., step=4.301036, value=19.289649)
    smootheness_mean = st.number_input(
        'Smootheness', min_value=0.052630, max_value=0.17, step=0.014064, value=0.096360)
    compactness_mean = st.number_input(
        'Compactness', min_value=0.019380, max_value=0.345400, step=0.052813, value=0.104341)
    # select box for model type
with col2:
    concativity_mean = st.number_input(
        'Concavity', min_value=0., max_value=0.426800, step=0.079720, value=0.088799)
    concave_points_mean = st.number_input(
        'Concave Points', min_value=0., max_value=0.201200, step=0.038803, value=0.048919)
    symmetry_mean = st.number_input(
        'Symmetry Mean', min_value=0.106000, max_value=0.304000, step=0.027414, value=0.181162)
    fractal_dimension_mean = st.number_input(
        'Fractal Dimension', min_value=0.049960, max_value=0.097440, step=0.007060, value=0.062798)

predict_button = st.button('Prediksi')

if predict_button:
    # create a dataframe
    df = pd.DataFrame({'radius_mean': radius_mean,
                       'texture_mean': texture_mean,
                       'smoothness_mean': smootheness_mean,
                       'compactness_mean': compactness_mean,
                       'concavity_mean': concativity_mean,
                       'concave points_mean': concave_points_mean,
                       'symmetry_mean': symmetry_mean,
                       'fractal_dimension_mean': fractal_dimension_mean}, index=[0])
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
