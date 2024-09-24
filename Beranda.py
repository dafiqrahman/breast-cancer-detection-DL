import streamlit as st
import pandas as pd
import numpy as np
import pickle
from functions.predict import predict

# make center
st.markdown("""
<style>
.center {
    text-align: center;

}
</style>
""", unsafe_allow_html=True)
st.markdown('<h1 class="center">Aplikasi Prediksi Kanker Payudara Menggunakan Jaringan Saraf Tiruan</h1>',
            unsafe_allow_html=True)

# write html

st.markdown("""
<br><h4 class="center">Nama Pembuat:</h4>
<p class="center">1. Rizwan Arisandi</p>
<p class="center">2. Adhe Lingga Dewi</p>
<p class="center">3. Fathy Radhia</p>
<p class="center">4. Julian Daffa Dzaky</p>
""", unsafe_allow_html=True)
