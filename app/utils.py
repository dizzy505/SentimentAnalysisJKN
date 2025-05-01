import base64
import pandas as pd
import logging
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_csv_download_link(df):
    """Generate a download link for the DataFrame"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="sentiment_data.csv">Download CSV Template</a>'
    return href

def init_session_state():
    """Initialize session state variables"""
    if 'logged_in' not in st.session_state:
        st.session_state.update({
            'logged_in': False,
            'role': None,
            'notifications': [],
            'model': None,
            'vectorizer': None,
            'data': None,
            'data_loaded': False,
            'sample_data_used': False,
            'db_connection': None
        }) 