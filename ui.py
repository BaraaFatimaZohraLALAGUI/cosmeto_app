import streamlit as st
import pickle 
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import torch
import sklearn
import re

class UI:
    def __init__(self):
        if "data" not in st.session_state:
            st.session_state.data = []

        if 'irrit_data' not in st.session_state:
            st.session_state.irrit_data = pd.read_csv('data/cosmetovigilance_ewg_compounds_final_19_07_2024.csv')

        # Load tokenizer, model, and SVM classifier
        self.tokenizer_pubmedbert = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.model_pubmedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        with open('models/svc.pkl', 'rb') as file:
            self.svc_model = pickle.load(file)

        # UI Components
        st.write("# Cosmetovigilance Checker")
        self.compound = st.text_input(label='', value='', placeholder='titanium dioxide')

        if self.compound and self.compound not in st.session_state.data:
            st.session_state.data.append(self.compound)

        if st.session_state.data:
            compounds_sentence = ', '.join(st.session_state.data)
            st.write(f"Compounds list: {compounds_sentence}")
            st.button('Check Carcinogenicity', on_click=self.predict_carc, args=(st.session_state.data,))

    def get_combined_embedding(self, compound_list):
        combined_text = ', '.join(compound_list)
        inputs = self.tokenizer_pubmedbert(combined_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model_pubmedbert(**inputs)
        cls_embedding = np.array(outputs.last_hidden_state[0, 0])
        return cls_embedding

    def split_compound_input(self, compound_input):
        return  [compound.strip() for compound in re.split(r'[,]', compound_input)]
    
    def predict_carc(self, compound_list):
        embeddings = self.get_combined_embedding(compound_list).reshape(1, -1)
        pred = self.svc_model.predict(embeddings)
        prediction = ['HIGH', 'LOW', 'MODERATE'][pred[0]]
        st.write(f'-- Carcinogenicity of the product: ***{prediction}***')

        # Split the input string for irritation checks individualy
        compound_split_list = self.split_compound_input(compound_list[0])
        si, ei = self.get_irrit(compound_split_list)

        # Display relevant warnings
        if si and ei:
            st.write('-- This product can cause ***Skin and Eye Irritation***')
        elif si:
            st.write('-- This product can cause ***Skin Irritation***')
        elif ei:
            st.write('-- This product can cause ***Eye Irritation***')

    
    def get_irrit(self, compound_list):
        irrit_data = st.session_state.irrit_data
        skin_irrit = False
        eye_irrit = False

        for compound in compound_list:
            filtered_data = irrit_data[irrit_data['chemical'] == compound]

            if not filtered_data.empty:
                # Check Skin Irritation
                if pd.notna(filtered_data['Skin Irrit percentage'].values[0]) and filtered_data['Skin Irrit percentage'].values[0] != 'N/A':
                    skin_irrit = True
                
                # Check Eye Irritation
                if pd.notna(filtered_data['Eye Irrit percentage'].values[0]) and filtered_data['Eye Irrit percentage'].values[0] != 'N/A':
                    eye_irrit = True

        return skin_irrit, eye_irrit
