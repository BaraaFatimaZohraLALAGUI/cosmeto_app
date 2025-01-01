from app import apply_styling
import streamlit as st
import pickle
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch
import re

apply_styling()

class UI:
    def __init__(self):
        apply_styling()
        if "data" not in st.session_state:
            st.session_state.data = []

        if 'irrit_data' not in st.session_state:
            st.session_state.irrit_data = pd.read_csv('data/cosmetovigilance_ewg_compounds_final_19_07_2024.csv')

        self.tokenizer_pubmedbert = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")
        self.model_pubmedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract")

        with open('models/multi/svc.pkl', 'rb') as file:
            self.svc_model = pickle.load(file)

        # Project Description
        st.title("🪷CosmetoCare ")
        st.divider()
        st.subheader("🔍 CosmetoCare Multicompound")
        st.markdown(
            """<div style="text-align: center;">
            <p>In <strong>CosmetoCare Multicompound</strong>, take the list of compounds in your cosmetic product enter it, and check if the product you're product might be harmful, whether it can cause eye irritation, skin irritation or if it is carcinogenic </p>
            </div>""",
            unsafe_allow_html=True
        )

        with st.container():
            self.compound = st.text_input(
                label='Enter Compound List of the product',
                value='',
                placeholder='e.g., titanium dioxide',
                help="Enter chemical compounds separated by commas."
            )

            if self.compound and self.compound not in st.session_state.data:
                st.session_state.data.append(self.compound)

        if st.session_state.data:
            compounds_sentence = ', '.join(st.session_state.data)
            st.write(f"Compounds list: {compounds_sentence}")
            if st.button('Check Carcinogenicity'):
                self.predict_carc(st.session_state.data)

    def get_combined_embedding(self, compound_list):
        combined_text = ', '.join(compound_list)
        inputs = self.tokenizer_pubmedbert(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model_pubmedbert(**inputs)
        cls_embedding = np.array(outputs.last_hidden_state[0, 0])
        return cls_embedding

    def split_compound_input(self, compound_input):
        return [compound.strip() for compound in re.split(r'[,]', compound_input)]

    def predict_carc(self, compound_list):
        embeddings = self.get_combined_embedding(compound_list).reshape(1, -1)
        pred = self.svc_model.predict(embeddings)
        prediction = ['HIGH', 'LOW', 'MODERATE'][pred[0]]

        if prediction == 'HIGH':
            result_color = '#910c00'
        elif prediction == 'LOW':
            result_color = '#3bab5a'
        else:
            result_color = '#cc5216'
        
        # Display results at the bottom of the page
        with st.container():
            st.markdown(
                f'<div style="background-color: #f1ccd7; padding: 10px; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: #e2808a;">Carcinogenicity of the product: <span style="color: {result_color};"><strong>{prediction}</strong></span></h3>'
                f'</div>',
                unsafe_allow_html=True
            )

            compound_split_list = self.split_compound_input(compound_list[0])
            si, ei = self.get_irrit(compound_split_list)

            if si and ei:
                st.warning('🚨 This product can cause **Skin and Eye Irritation**')
            elif si:
                st.warning('🚨 This product can cause **Skin Irritation**')
            elif ei:
                st.warning('🚨 This product can cause **Eye Irritation**')
    def get_irrit(self, compound_list):
        irrit_data = st.session_state.irrit_data
        skin_irrit = False
        eye_irrit = False

        for compound in compound_list:
            filtered_data = irrit_data[irrit_data['chemical'] == compound]

            if not filtered_data.empty:
                if pd.notna(filtered_data['Skin Irrit percentage'].values[0]) and filtered_data['Skin Irrit percentage'].values[0] != 'N/A':
                    skin_irrit = True
                
                if pd.notna(filtered_data['Eye Irrit percentage'].values[0]) and filtered_data['Eye Irrit percentage'].values[0] != 'N/A':
                    eye_irrit = True

        return skin_irrit, eye_irrit

UI()
