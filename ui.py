import streamlit as st
import pickle 
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import re

class UI:
    def __init__(self):
        # Initialize session state for the data list
        if "data" not in st.session_state:
            st.session_state.data = []

        # Load the tokenizer and model 
        self.tokenizer_pubmedbert = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
        self.model_pubmedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

        # Load SVM classifier
        with open('svc.pkl', 'rb') as file:
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
    def get_combined_embedding(self, text):
        
        inputs = self.tokenizer_pubmedbert(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model_pubmedbert(**inputs)
        cls_embedding = outputs.last_hidden_state[0, 0].numpy()
        return cls_embedding


    def predict_carc(self, compound_list):
        embeddings = self.get_combined_embedding(compound_list)
        pred = self.svc_model.predict(embeddings)
        if pred == 0:
            prediction = 'HIGH'
        elif pred == 1:
            prediction = 'LOW'
        elif pred == 2:
            prediction = 'MODERATE'

        st.write(f'The list of compounds is {compound_list}')
        st.write(f'carcinogenicity of the product {prediction}')
