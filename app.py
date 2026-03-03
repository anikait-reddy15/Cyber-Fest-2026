import streamlit as st
import os
from extractor import extract_metadata
from model_inference import get_safety_percentage

st.set_page_config(page_title="Fake App Detector", page_icon="X")
st.title("Fake Application Detector")

uploaded_file = st.file_uploader("Upload APK", type=["apk"])

if uploaded_file is not None:
    temp_path = os.path.join("temp", uploaded_file.name)
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    metadata = extract_metadata(temp_path)
    
    if metadata:
        st.json(metadata)
        score = get_safety_percentage(metadata)
        
        if score > 75:
            st.success(f"Safety: {score}%")
        elif score > 40:
            st.warning(f"Safety: {score}%")
        else:
            st.error(f"Safety: {score}%")
            
    if os.path.exists(temp_path):
        os.remove(temp_path)