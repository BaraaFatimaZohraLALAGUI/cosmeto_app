import streamlit as st
import pickle 
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import sklearn

class UI:
    def __init__(self):
        # Initialize session state for the data list
        if "data" not in st.session_state:
            st.session_state.data = []

        # Load the tokenizer and model 
        self.tokenizer_pubmedbert = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.model_pubmedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

        # Load SVM classifier
        with open('models/svc.pkl', 'rb') as file:
            self.svc_model = pickle.load(file)

        # Title and instructions
        st.write("""
        # Cosmetovigilance Checker
        ***Enter at least one component to check the carcinogenicity of the product***
        """)

        # Text input for compound names
        self.compound = st.text_input(
            label='',
            value='', 
            placeholder='titanium dioxide'
        )

        # Add the entered compound to the list if it's not empty and not a duplicate
        if self.compound and self.compound not in st.session_state.data:
            st.session_state.data.append(self.compound)

        if st.session_state.data:
            compounds_sentence = ', '.join(st.session_state.data)
            st.write(f"Compounds list: {compounds_sentence}")
            button = st.button('Check Carcinogenicity', on_click=self.predict_carc, args=(st.session_state.data,))
    
        # Function to get combined embedding
    def get_combined_embedding(self, compound_list):
        combined_text = ', '.join(compound_list)
        inputs = self.tokenizer_pubmedbert(combined_text, return_tensors="pt", padding =True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model_pubmedbert(**inputs)
        cls_embedding = np.array(outputs.last_hidden_state[0, 0])
        return cls_embedding


    def predict_carc(self, compound_list):
        embeddings = self.get_combined_embedding(compound_list).reshape(1, -1)
        pred = self.svc_model.predict(embeddings)

        prediction = ['HIGH', 'LOW', 'MODERATE'][pred[0]]

        st.write(f'The list of compounds is: {", ".join(compound_list)}')
        st.write(f'Carcinogenicity of the product: ***{prediction}***')
