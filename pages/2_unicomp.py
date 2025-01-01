from app import apply_styling
import streamlit as st
import numpy as np
import pandas as pd
import torch
from pytorch_lightning import LightningModule
from chemprop import data, models, nn
from rdkit import Chem
import requests

class MPNNLightningModule(LightningModule):
    def __init__(self, mp, agg, ffn, batch_norm, metric_list):
        super().__init__()
        self.model = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

    def forward(self, x):
        return self.model(x)

apply_styling()
class UI:
    def __init__(self):
        apply_styling()
        if "data1" not in st.session_state:
            st.session_state.data1 = []
        
        mp = nn.BondMessagePassing()
        agg = nn.MeanAggregation()
        ffn_carc = nn.MulticlassClassificationFFN(n_classes=3, n_tasks=1)
        ffn_skin = nn.MulticlassClassificationFFN(n_classes=4, n_tasks=1)
        ffn_eye = nn.MulticlassClassificationFFN(n_classes=4, n_tasks=1)
        batch_norm = False
        metric_list = None
        self.carc_model = MPNNLightningModule.load_from_checkpoint('models/uni/model_chemprop_carc.ckpt',
                                                    mp=mp, agg=agg, ffn=ffn_carc, batch_norm=batch_norm, metric_list=metric_list)
        self.carc_featurizer = torch.load('models/uni/featurizer_carc.pth')

        self.skin_model = MPNNLightningModule.load_from_checkpoint('models/uni/model_chemprop_skin.ckpt',
                                                    mp=mp, agg=agg, ffn=ffn_skin, batch_norm=batch_norm, metric_list=metric_list)
        self.skin_featurizer = torch.load('models/uni/featurizer_skin.pth')

        self.eye_model = MPNNLightningModule.load_from_checkpoint('models/uni/model_chemprop_eye.ckpt',
                                                    mp=mp, agg=agg, ffn=ffn_eye, batch_norm=batch_norm, metric_list=metric_list)
        self.eye_featurizer = torch.load('models/uni/featurizer_eye.pth')

        self.smis = ['']
        # Project Description
        st.title("🪷CosmetoCare ")
        st.divider()
        st.subheader("🔍 CosmetoCare UniCompound")
        st.markdown(
            """<div style="text-align: center;">
            <p>In <strong>CosmetoCare Unicompound</strong>, input the name of a molecule and let cosmetocare check if it may cause skin irritation, eye irritation or if it may be carcinogenic </p>
            </div>""",
            unsafe_allow_html=True
        )

        with st.container():
            self.compound = st.text_input(
                label='Enter Compound name',
                value='',
                placeholder='e.g., Morin dihydrate',
                help="Enter chemical name (IUPAC)."
            )

            if self.compound and self.compound not in st.session_state.data1:
                st.session_state.data1.append(self.compound)

        if st.session_state.data1:
            st.write(f"Compounds list: {st.session_state.data1[0]}")
            if st.button('Check Carcinogenicity'):
                self.predict_carc(st.session_state.data1[0])

        if st.session_state.data1:
            st.write(f"Compounds list: {st.session_state.data1[0]}")
            if st.button('Check Skin Irritation'):
                self.predict_skin_irritation(st.session_state.data1[0])

        
        if st.session_state.data1:
            st.write(f"Compounds list: {st.session_state.data1[0]}")
            if st.button('Check Eye Irritation'):
                self.predict_eye_irritation(st.session_state.data1[0])

    def is_valid_smiles(self, smi):
        """Checks if a SMILES string is valid."""
        mol = Chem.MolFromSmiles(smi)
        return mol is not None
    
    def get_smiles(self, name):
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/JSON"

        try:
            response = requests.get(smiles_url)
            response.raise_for_status()
            smiles = response.json()['PropertyTable']['Properties'][0]['CanonicalSMILES']
            print(f'    SMILES: {smiles}') 
            return smiles
        except Exception as e:
            st.error(f"Error fetching SMILES for {name}: {e}")
            return None
        
    def preprocess(self, name):
        new_smiles = self.get_smiles(name)
        self.smis.append(new_smiles)
        valid_new_data = [smi for smi in self.smis if self.is_valid_smiles(smi)]
        new_all_data = [data.MoleculeDatapoint.from_smi(smi, None) for smi in valid_new_data]

        return new_all_data

    def predict_carc(self, name):
        new_all_data = self.preprocess(name)
        new_dset = data.MoleculeDataset(new_all_data, self.carc_featurizer)
        new_data_loader = data.build_dataloader(new_dset, num_workers = 0, shuffle = False)
        predictions = []
        with torch.no_grad():
            for batch in new_data_loader:
                inputs = batch[0]
                outputs = self.carc_model(inputs)
                predictions.extend(outputs.cpu().numpy())
        predictions = [pre.argmax() for pre in predictions]

        pred = predictions[1]
        prediction = ['not carcinogenic', 'May cause cancer', 'suspected to cause cancer'][pred]

        if prediction == 'suspected to cause cancer':
            result_color = '#910c00'
        elif prediction == 'not carcinogenic':
            result_color = '#3bab5a'
        else:
            result_color = '#cc5216'
              
        with st.container():
            st.markdown(
                f'<div style="background-color: #f1ccd7; padding: 10px; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: #e2808a;"> The molecule {name} is : <span style="color: {result_color};"><strong>{prediction}</strong></span></h3>'
                f'</div>',
                unsafe_allow_html=True
            )

    def predict_skin_irritation(self, name):
        new_all_data = self.preprocess(name)
        new_dset = data.MoleculeDataset(new_all_data, self.skin_featurizer)
        new_data_loader = data.build_dataloader(new_dset, num_workers = 0, shuffle = False)
        predictions = []
        with torch.no_grad():
            for batch in new_data_loader:
                inputs = batch[0]
                outputs = self.skin_model(inputs)
                predictions.extend(outputs.cpu().numpy())
        predictions = [pre.argmax() for pre in predictions]

        pred = predictions[1]
        prediction = ['no skin irritation', 'skin corrision', 'skin irritation', 'mild skin irritation'][pred]

        if prediction == 'skin corrision':
            result_color = '#910c00'
        elif prediction == 'no skin irritation':
            result_color = '#3bab5a'
        else:
            result_color = '#cc5216'
              
        with st.container():
            st.markdown(
                f'<div style="background-color: #f1ccd7; padding: 10px; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: #e2808a;"> The molecule {name} may cause : <span style="color: {result_color};"><strong>{prediction}</strong></span></h3>'
                f'</div>',
                unsafe_allow_html=True
            )


    def predict_eye_irritation(self, name):
        new_all_data = self.preprocess(name)
        new_dset = data.MoleculeDataset(new_all_data, self.eye_featurizer)
        new_data_loader = data.build_dataloader(new_dset, num_workers = 0, shuffle = False)
        predictions = []
        with torch.no_grad():
            for batch in new_data_loader:
                inputs = batch[0]
                outputs = self.eye_model(inputs)
                predictions.extend(outputs.cpu().numpy())
        predictions = [pre.argmax() for pre in predictions]

        pred = predictions[1]
        prediction = ['no eye irritantation', 'serious eye damage', 'reversible eye irritantation', 'mildly reversible eye irritantation'][pred]

        if prediction == 'serious eye damage':
            result_color = '#910c00'
        elif prediction == 'no eye irritantation':
            result_color = '#3bab5a'
        else:
            result_color = '#cc5216'
              
        with st.container():
            st.markdown(
                f'<div style="background-color: #f1ccd7; padding: 10px; border-radius: 5px; text-align: center;">'
                f'<h3 style="color: #e2808a;"> The molecule {name} may cause : <span style="color: {result_color};"><strong>{prediction}</strong></span></h3>'
                f'</div>',
                unsafe_allow_html=True
            )


UI()