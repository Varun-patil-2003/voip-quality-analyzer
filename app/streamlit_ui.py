import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict_quality import predict_mos


import streamlit as st
# from src.predict_quality import predict_mos
import tempfile

st.title("VoIP Call Quality Analyzer")

audio_file = st.file_uploader("Upload a .wav file", type=["wav"])

if audio_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_file.read())
        tmp_path = tmp.name

    score = predict_mos(tmp_path)
    st.success(f"Predicted MOS Score: {score}")
