import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_custom_css():
    """Load enhanced CSS for a professional, dashboard-style AI product"""
    st.markdown("""
    <style>
        /* Global Reset & Base Styles */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }

        .main {
            background-color: #0f172a;
            padding: 0;
        }

        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Layout Container */
        .app-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 2rem;
        }

        /* Sidebar Styling */
        [data-testid="stSidebar"] {
            background-color: #1e293b;
            border-right: 1px solid rgba(255, 255, 255, 0.1);
        }

        .sidebar-header {
            padding: 1rem 0;
            margin-bottom: 2rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        .app-brand {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: -0.025em;
        }

        .sidebar-section-title {
            font-size: 0.75rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }

        /* Tech Card Styling */
        .tech-card {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.75rem;
            transition: all 0.2s ease;
        }

        .tech-card:hover {
            background: rgba(255, 255, 255, 0.06);
            border-color: rgba(99, 102, 241, 0.3);
            transform: translateX(4px);
        }

        .tech-card-name {
            font-size: 0.875rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }

        .tech-card-desc {
            font-size: 0.75rem;
            color: #94a3b8;
            line-height: 1.25;
        }

        /* Main Area Elements */
        .main-header {
            margin-bottom: 2.5rem;
            text-align: center;
        }

        .main-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.5rem;
            letter-spacing: -0.05em;
        }

        .main-subtitle {
            font-size: 1.125rem;
            color: #94a3b8;
            font-weight: 400;
        }

        /* Upload Section */
        .upload-container {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 2.5rem;
            text-align: center;
        }

        .upload-header {
            margin-bottom: 1.5rem;
        }

        .upload-title {
            font-size: 1.25rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }

        .upload-hint {
            font-size: 0.875rem;
            color: #94a3b8;
        }

        /* Uploader Override */
        [data-testid="stFileUploader"] {
            margin-top: 1rem;
        }

        /* Analysis Layout */
        .analysis-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 2rem;
            margin-top: 2rem;
        }

        .card-panel {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 1.5rem;
        }

        .panel-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 1.5rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }

        /* Result Cards */
        .result-card {
            background: rgba(255, 255, 255, 0.04);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            border-left: 3px solid #6366f1;
        }

        .result-label {
            font-size: 0.95rem;
            font-weight: 600;
            color: #ffffff;
            text-transform: capitalize;
            margin-bottom: 0.5rem;
        }

        .confidence-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.5rem;
        }

        .confidence-val {
            font-size: 0.8rem;
            color: #94a3b8;
            font-weight: 500;
        }

        .progress-bg {
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        }

        .progress-fill {
            height: 100%;
            background: #6366f1;
            border-radius: 10px;
        }

        /* Control Button */
        .stButton > button {
            background-color: #6366f1 !important;
            color: white !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            padding: 0.6rem 2rem !important;
            border: none !important;
            width: 100%;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_ai_model():
    return MobileNetV2(weights="imagenet")

def preprocess_image(image):
    img = np.array(image.convert('RGB'))
    img = cv2.resize(img, (224, 224))
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def main():
    st.set_page_config(
        page_title="VisionaryAI - Professional Image Recognition",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    load_custom_css()
    
    # --- SIDEBAR ---
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-header">
            <div class="app-brand">VisionaryAI</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-section-title">Technology Stack</div>', unsafe_allow_html=True)
        
        techs = [
            ("Streamlit", "Web interface framework"),
            ("TensorFlow & Keras", "Deep learning framework"),
            ("MobileNetV2", "Pre-trained image model"),
            ("OpenCV & Pillow", "Image processing libraries"),
            ("Python", "Programming language")
        ]
        
        for name, desc in techs:
            st.markdown(f"""
            <div class="tech-card">
                <div class="tech-card-name">{name}</div>
                <div class="tech-card-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- MAIN CONTENT ---
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">OBJECT RECOGNIZER</h1>
        <p class="main-subtitle">Smart Image Understanding with Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)



    # Center the uploader in the card visually (Streamlit logic)
    col1, col2, col3 = st.columns([1, 4, 1])
    with col2:
        uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file:
        image = Image.open(uploaded_file)
        
        # Action Button
        analyze_col1, analyze_col2, analyze_col3 = st.columns([1, 2, 1])
        with analyze_col2:
            analyze_btn = st.button("Analyze Image", type="primary")

        if analyze_btn:
            model = load_ai_model()
            with st.spinner("AI is analyzing the image..."):
                processed_img = preprocess_image(image)
                preds = model.predict(processed_img, verbose=0)
                decoded_preds = decode_predictions(preds, top=3)[0]

            # Results Display (Two Column Layout)
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.markdown('<div class="card-panel"><div class="panel-title">Uploaded Image</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            with res_col2:
                st.markdown('<div class="card-panel"><div class="panel-title">Prediction Results</div>', unsafe_allow_html=True)
                for _, label, score in decoded_preds:
                    score_pct = score * 100
                    label_clean = label.replace('_', ' ').title()
                    st.markdown(f"""
                    <div class="result-card">
                        <div class="confidence-row">
                            <div class="result-label">{label_clean}</div>
                            <div class="confidence-val">{score_pct:.1f}%</div>
                        </div>
                        <div class="progress-bg">
                            <div class="progress-fill" style="width: {score_pct}%;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("👆 Please upload an image to start recognition.")

if __name__ == "__main__":
    main()
