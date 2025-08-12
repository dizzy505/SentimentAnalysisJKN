import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from models import SentimentAnalyzer
from database import create_db_connection, fetch_data_from_db, insert_data_to_db, batch_insert_to_db, register_user, authenticate_user
from utils import get_csv_download_link
import hashlib
from menus import login as login_menu
from menus import data_input as data_input_menu
from menus import data_overview as data_overview_menu
from menus import model_performance as model_performance_menu
from menus import sentiment_prediction as sentiment_prediction_menu
from menus import wordcloud as wordcloud_menu

logger = logging.getLogger(__name__)

st.markdown("""
<link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap\" rel=\"stylesheet\">
<style>
body {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
    background-color: #19223a;
}
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 1.1em;
    font-weight: 600;
    background: #22305a;
    color: #fff;
    border: none;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
    transition: all 0.2s;
}
.stButton>button:hover {
    background: #2a3962;
    color: #fff;
    transform: translateY(-2px) scale(1.03);
}
.stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
    border-radius: 12px;
    border: 1.5px solid #e0e6ed;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    background: #fff;
    margin-bottom: 0.5rem;
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stTextArea>div>div>textarea:focus {
    border: 1.5px solid #28407a;
    outline: none;
}
.stFileUploader>div>div>button {
    border-radius: 12px;
    background: #22305a;
    color: #fff;
    font-weight: 600;
    border: none;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
}
.stFileUploader>div>div>button:hover {
    background: #2a3962;
}
.css-1d391kg, .stAlert, .metric-card, .stDataFrame, .stExpander, .stTabs {
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
}
.metric-card {
    background-color: #22305a;
    padding: 1.2rem;
    border-radius: 16px;
    margin: 0.5rem;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
    color: #fff;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: #fff;
    border-radius: 10px 10px 0 0;
    background: #22305a;
    margin-right: 4px;
    padding: 0.7rem 1.2rem;
    transition: background 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #28407a;
    color: #fff;
}
.stExpander {
    background: #22305a;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer
        
        if st.session_state.db_connection is None:
            st.session_state.db_connection = create_db_connection()
        
        if not st.session_state.data_loaded and st.session_state.db_connection and st.session_state.db_connection.is_connected():
            self._load_database_data()
        
    def _load_database_data(self):
        """Otomatis memuat data dari database"""
        try:
            db_data = fetch_data_from_db(st.session_state.db_connection)
            if not db_data.empty:
                st.session_state.original_data = db_data.copy()
                
                positif_samples = db_data[db_data['Label'] == 'Positif']
                negatif_samples = db_data[db_data['Label'] == 'Negatif']
                
                if len(positif_samples) < 7000:
                    n_samples = 7000 - len(positif_samples)
                    synthetic_samples = positif_samples.sample(n=n_samples, replace=True, random_state=42)
                    db_data = pd.concat([db_data, synthetic_samples], ignore_index=True)
                
                st.session_state.data = db_data
                st.session_state.data_loaded = True
                st.session_state.sample_data_used = False
                st.session_state.data_source = "database"
                logger.info(f"Otomatis memuat {len(db_data)} records dari database")
            else:
                st.session_state.data_source = "none"
                logger.info("Tidak ada data ditemukan di database")
        except Exception as e:
            logger.error(f"Error otomatis memuat data dari database: {str(e)}")
            st.session_state.data_source = "none"

    def render_login(self):
        """Delegasi render login ke modul menu."""
        login_menu.render_login(self)

    def render_data_input(self):
        """Delegasi render data input ke modul menu."""
        data_input_menu.render_data_input(self)

    def render_data_overview(self):
        """Delegasi render data overview ke modul menu."""
        data_overview_menu.render_data_overview(self)

    def render_model_performance(self):
        """Delegasi render model performance ke modul menu."""
        model_performance_menu.render_model_performance(self)

    def render_sentiment_prediction(self):
        """Delegasi render sentiment prediction ke modul menu."""
        sentiment_prediction_menu.render_sentiment_prediction(self)

    def render_wordcloud(self):
        """Delegasi render wordcloud ke modul menu."""
        wordcloud_menu.render_wordcloud(self)
 