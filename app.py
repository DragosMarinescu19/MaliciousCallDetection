import os
import streamlit as st
from audio_utils import MODEL_INFO
from ui_parts import render_compare_tab, render_record_tab, render_upload_tab



st.set_page_config(
    page_title="Malicious Call Detector",
    page_icon="🛡️",
    layout="wide",
)

css_path = "style.css"
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.info("Custom stylesheet (style.css) not found — using default Streamlit theme.")



st.markdown("# 🛡️ Malicious Call Detector")
st.markdown(
    "Upload an audio file or record from your microphone to detect "
    "fraudulent phone calls using deep learning."
)
st.divider()



with st.sidebar:
    st.header("⚙️ Settings")

    selected_model: str = st.selectbox(
        "Select Model",
        list(MODEL_INFO.keys()),
        index=0,
        help="Choose which trained model to use for prediction.",
    )
    st.caption(MODEL_INFO[selected_model]["desc"])

    st.divider()
    st.markdown("**Audio Limits**")
    max_duration: int = st.slider("Max audio duration (seconds)", 10, 600, 120, step=10)

    st.divider()
  



tab_upload, tab_record, tab_compare = st.tabs(
    ["📁 Upload Audio", "🎙️ Record Audio", "⚖️ Compare Models"]
)

with tab_upload:
    render_upload_tab(selected_model, max_duration)

with tab_record:
    render_record_tab(selected_model, max_duration)

with tab_compare:
    render_compare_tab(max_duration)