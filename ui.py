import streamlit as st

class UI:
    def __init__(self):
        # Initialize session state for the data list
        if "data" not in st.session_state:
            st.session_state.data = []

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

    def predict_carc(self, compound_list):
        st.write(f'The list of compounds is {compound_list}')
