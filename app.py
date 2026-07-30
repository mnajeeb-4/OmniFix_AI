"""
OmniFix AI - Main Streamlit Application
Diagnoses mechanical issues using audio spectrogram analysis and computer vision AR overlays.
"""

import streamlit as st
import numpy as np
import pandas as pd
import tempfile
import os
from modules.audio_processor import AudioProcessor
from modules.vision_processor import VisionProcessor
from modules.tools_db import TOOLS_DB
import cv2
from PIL import Image
import base64
import time

# ----------------------------
# Page Configuration & Styling
# ----------------------------
st.set_page_config(
    page_title="OmniFix AI",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism dark theme
CUSTOM_CSS = """
<style>
    /* Global dark theme */
    .stApp {
        background: #0b0b0e;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    /* Glassmorphism card */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    /* Badge tags */
    .badge-critical {
        background: #ff0000;
        color: white;
        padding: 2px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-warning {
        background: #ffcc00;
        color: #111;
        padding: 2px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-moderate {
        background: #ff8c00;
        color: white;
        padding: 2px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    .badge-healthy {
        background: #00cc66;
        color: white;
        padding: 2px 12px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.8rem;
    }
    /* Tool grid cards */
    .tool-card {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 15px;
        text-align: center;
        transition: transform 0.2s;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .tool-card:hover {
        transform: translateY(-5px);
        border-color: rgba(255,255,255,0.2);
    }
    .tool-icon {
        font-size: 3rem;
        margin-bottom: 10px;
    }
    /* Sidebar */
    .css-1d391kg {
        background-color: #121216;
    }
    /* Streamlit elements tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px;
        background: rgba(255,255,255,0.03);
        padding: 8px 16px;
        color: #ccc;
    }
    .stTabs [aria-selected="true"] {
        background: rgba(255,255,255,0.1);
        color: white;
    }
    /* Input fields */
    .stFileUploader {
        background: rgba(255,255,255,0.02);
        border-radius: 12px;
        padding: 10px;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------------------
# Initialize Processors
# ----------------------------
@st.cache_resource
def get_audio_processor():
    return AudioProcessor()

@st.cache_resource
def get_vision_processor():
    return VisionProcessor()

audio_proc = get_audio_processor()
vision_proc = get_vision_processor()

# ----------------------------
# Helper Functions
# ----------------------------
def render_badge(severity):
    """Return HTML badge based on severity."""
    if severity == "CRITICAL":
        return '<span class="badge-critical">CRITICAL</span>'
    elif severity == "WARNING":
        return '<span class="badge-warning">WARNING</span>'
    elif severity == "MODERATE":
        return '<span class="badge-moderate">MODERATE</span>'
    elif severity == "HEALTHY":
        return '<span class="badge-healthy">HEALTHY</span>'
    return ''

def display_audio_analysis(audio_file):
    """Run audio analysis and display results."""
    with st.spinner("Analyzing audio..."):
        try:
            results = audio_proc.full_analysis(audio_file)
        except Exception as e:
            st.error(f"Audio analysis failed: {e}")
            return
    # Display spectrogram
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Spectrogram")
    st.image(results["spectrogram"], use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Show metrics
    cols = st.columns(3)
    for i, (fault_type, result) in enumerate([("Bearing", results["bearing"]),
                                              ("Misfire", results["misfire"]),
                                              ("Belt", results["belt"])]):
        with cols[i]:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.caption(fault_type)
            severity = result["severity"]
            st.markdown(f"**Fault:** {result['fault']}")
            st.markdown(f"**Confidence:** {result['confidence']:.1%}")
            st.markdown(render_badge(severity), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # Pitch deviation
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.metric("Pitch Deviation", f"{results['pitch_deviation_pct']:.2f}%",
              help="Higher deviation indicates unstable frequency, potential mechanical issue.")
    st.markdown('</div>', unsafe_allow_html=True)

def display_vision_analysis(image_input, engine_type='car'):
    """Run vision analysis and display annotated image."""
    with st.spinner("Processing image..."):
        try:
            annotated_img, detections = vision_proc.run_detection(image_input, engine_type)
        except Exception as e:
            st.error(f"Vision analysis failed: {e}")
            return
    # Convert BGR to RGB for display
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    st.image(annotated_img_rgb, caption="AR Overlay", use_container_width=True)
    # Show detection summary
    if detections:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Detected Issues")
        df = pd.DataFrame(detections)
        df['severity'] = df['severity'].apply(lambda x: f"🔴 {x}" if x=='CRITICAL' else f"🟡 {x}" if x=='WARNING' else f"🟠 {x}" if x=='MODERATE' else f"🟢 {x}")
        st.dataframe(df[['label', 'severity', 'confidence'] if 'confidence' in df.columns else ['label', 'severity']],
                     use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.success("No significant issues detected.")

# ----------------------------
# Sidebar
# ----------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
    st.title("OmniFix AI")
    st.markdown("**Smart Diagnostics for Vehicles & Home Appliances**")
    st.markdown("---")
    st.write("Upload audio or image files to diagnose faults.")
    st.write("Powered by YOLO & Librosa.")

# ----------------------------
# Main Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🚗 Automobiles", "🏍️ Motorcycles", "🏠 Household", "🔧 Tool Dictionary"])

# ---------- Tab 1: Automobiles ----------
with tab1:
    st.markdown('<h2 style="margin-bottom:0;">🚗 Automobile Diagnostics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">Engine noise spectrogram & visual leak detection</p>', unsafe_allow_html=True)

    col_audio, col_vision = st.columns(2)
    with col_audio:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎧 Engine Audio")
        audio_file = st.file_uploader("Upload engine audio (WAV/MP3)", type=["wav", "mp3"], key="car_audio")
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            display_audio_analysis(tmp_path)
            os.unlink(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vision:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📷 Engine Visual")
        vision_input = st.file_uploader("Upload engine image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="car_image")
        if vision_input:
            display_vision_analysis(vision_input, engine_type='car')
        st.markdown('</div>', unsafe_allow_html=True)

    # Torque tool recommendation placeholder
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("🔧 Torque Recommendation")
    st.write("Based on the analysis, we recommend using a **digital torque wrench** set to 45 Nm for cylinder head bolts.")
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 2: Motorcycles ----------
with tab2:
    st.markdown('<h2 style="margin-bottom:0;">🏍️ Motorcycle Diagnostics</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">Chain slack visual analysis & exhaust sound frequency</p>', unsafe_allow_html=True)

    col_audio, col_vision = st.columns(2)
    with col_audio:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎧 Exhaust Sound")
        audio_file = st.file_uploader("Upload exhaust audio (WAV/MP3)", type=["wav", "mp3"], key="bike_audio")
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            display_audio_analysis(tmp_path)
            os.unlink(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vision:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📷 Chain & Drive")
        vision_input = st.file_uploader("Upload chain image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="bike_image")
        if vision_input:
            display_vision_analysis(vision_input, engine_type='motorcycle')
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 3: Household Appliances ----------
with tab3:
    st.markdown('<h2 style="margin-bottom:0;">🏠 Household Appliances</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">Fan humming/wobble & LED flickering detection</p>', unsafe_allow_html=True)

    col_audio, col_vision = st.columns(2)
    with col_audio:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🎧 Fan / Motor Noise")
        audio_file = st.file_uploader("Upload fan/motor audio (WAV/MP3)", type=["wav", "mp3"], key="house_audio")
        if audio_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp.write(audio_file.read())
                tmp_path = tmp.name
            display_audio_analysis(tmp_path)
            os.unlink(tmp_path)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_vision:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📷 Bulb / Circuit")
        vision_input = st.file_uploader("Upload bulb/circuit image (JPG/PNG)", type=["jpg", "jpeg", "png"], key="house_image")
        if vision_input:
            display_vision_analysis(vision_input, engine_type='household')
        st.markdown('</div>', unsafe_allow_html=True)

# ---------- Tab 4: Tool Dictionary ----------
with tab4:
    st.markdown('<h2 style="margin-bottom:0;">🔧 Interactive Tool Dictionary</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color:#aaa;">Essential mechanical tools with usage guides and safety tips</p>', unsafe_allow_html=True)

    # Grid of tools
    cols = st.columns(2)
    for idx, (key, tool) in enumerate(TOOLS_DB.items()):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div class="tool-card glass-card">
                <div class="tool-icon"><img src="{tool['icon_url']}" width="80"></div>
                <h4>{tool['name']}</h4>
                <p style="font-size:0.9rem; color:#bbb;">{tool['description']}</p>
                <p><strong>Usage:</strong> {tool['usage']}</p>
                <p><strong>⚠️ Safety:</strong> {tool['safety_tips']}</p>
            </div>
            """, unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<center style='color:#666;'>OmniFix AI v1.0 • Built with Streamlit, YOLO, and Librosa</center>", unsafe_allow_html=True)
