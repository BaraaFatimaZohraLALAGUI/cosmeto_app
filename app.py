import streamlit as st
import pandas as pd
import numpy as np
import time



# # Add a placeholder
# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0.1)

# '...and now we\'re done!'

st.write(
    
    """
# Cosmetovigilance Checker
    """
)

st.text_input(label='Enter the cosmetic product components', value= '')
