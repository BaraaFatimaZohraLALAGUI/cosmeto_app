        

import streamlit as st
import base64
import os  

# Convert the background image to Base64
def get_base64_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Background image not found at: {image_path}")
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()
    

def apply_styling():
    background_image_path = "cosmeto_app/images/bg.jpg" 
    background_image = get_base64_image(background_image_path)
    st.markdown(
        f"""
        <style>
        .main {{
            background-image: url("data:image/jpeg;base64,{background_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            background-position: center;
        }}
        h1, h2, h3 {{
            color: #2e765e; 
        }}
        p {{
            color: #638c80; /* Light Violet */
        }}
        .stTextInput input {{
            color: #f1ccd7;
            border: 1px solid #e2808a;
            border-radius: 5px;
            padding: 10px;
        }}
        stHeader header{{
        color: #e2808a;
        padding : 0px;
        }}
        .stButton button {{
            background-color: #e2808a;
            color: #CFF7EF;
            border: none;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
            transition: 0.3s;
        }}
        .stButton button:hover {{
            background-color: #638c80;
            color: #CFF7EF;
        }}
        

        .sidebar .sidebar-content{{
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }}


        </style>
        """,
        unsafe_allow_html=True
    )




if __name__ == "__main__":

    st.set_page_config(
        page_title = 'CosmetoCare',
        page_icon = '🪷',
    )

    apply_styling()

    st.write('# 🪷CosmetoCare ')
    st.divider()
    st.subheader("🔍 Cosmetovigilance Checker")
    st.markdown(
        """<div style="text-align: center;">
        <p>Welcome to <strong>CosmetoCare</strong>, your trusted tool for assessing the safety of cosmetic ingredients. This app uses advanced AI models to analyze the carcinogenicity and irritation risks of chemical compounds commonly found in cosmetic products. Whether you're a consumer or a manufacturer, <strong>CosmetoCare</strong> helps you make informed decisions about the products you use or create.</p>
        </div>""",
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <style>
        .sidebar .sidebar-content {
            background-image: linear-gradient(#2e7bcf,#2e7bcf);
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )  
    st.sidebar.success('Select prediction method, Multi for multi compound, uni for unicompound')

    # ui = UI()
