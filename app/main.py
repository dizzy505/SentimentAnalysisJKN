import streamlit as st

st.set_page_config(
    page_title="Analisis Sentimen Mobile JKN",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

from models import SentimentAnalyzer
from dashboard import Dashboard
from utils import init_session_state
from database import create_db_connection

init_session_state()
if "current_page" not in st.session_state:
    st.session_state.current_page = "Data Input"

st.markdown("""
<link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap\" rel=\"stylesheet\">
<style>
body {
    background-color: #19223a;
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
}
.main-header {
    background: #22305a;
    padding: 2rem 1rem 1.5rem 1rem;
    border-radius: 18px;
    margin-bottom: 1.5rem;
    color: #fff;
    text-align: center;
    box-shadow: 0 4px 16px rgba(44,62,80,0.10);
}
.main-header h1, .main-header h2, .main-header h3, .main-header h4, .main-header h5, .main-header h6 {
    color: #fff !important;
    text-shadow: 0 2px 8px rgba(25,34,58,0.10);
}
.nav-container, .compact-nav {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 24px;
    padding: 0.7rem 1rem;
    background: #22305a;
    border-radius: 12px;
    justify-content: flex-start;
    flex-wrap: wrap;
    box-shadow: 0 2px 8px rgba(44,62,80,0.06);
}
.nav-button, .compact-nav .stButton>button {
    padding: 0.7rem 1.3rem;
    color: white;
    font-weight: 600;
    border-radius: 10px;
    border: 2px solid transparent;
    transition: 0.2s ease-in-out;
    font-size: 1rem;
    background: #22305a;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
}
.nav-button:hover, .compact-nav .stButton>button:hover {
    border: 2px solid #60a5fa;
    background: #2a3962;
    cursor: pointer;
    color: #fff;
}
.nav-button.active, .compact-nav .stButton>button[data-testid="baseButton-primary"] {
    background: #28407a;
    border: 2px solid #60a5fa;
    color: #fff;
}
.status-connected {
    color: #28a745;
    font-weight: bold;
}
.status-disconnected {
    color: #dc3545;
    font-weight: bold;
}
.stButton > button {
    width: 100%;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
    background: #22305a;
    color: #fff;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
    border: none;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}
.stButton > button:hover {
    background: #2a3962;
    color: #fff;
    transform: translateY(-2px) scale(1.03);
}
.element-container {
    margin: 0 !important;
}
/* Card & container */
.metric-card, .stAlert, .css-1d391kg, .stDataFrame, .stExpander, .stTabs, .stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea, .stFileUploader>div>div>button {
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
}
.stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
    border: 1.5px solid #e0e6ed;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    background: #fff;
    margin-bottom: 0.5rem;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stTextArea>div>div>textarea:focus {
    border: 1.5px solid #28407a;
    outline: none;
}
.stFileUploader>div>div>button {
    background: #22305a;
    color: #fff;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
}
.stFileUploader>div>div>button:hover {
    background: #2a3962;
}
</style>
""", unsafe_allow_html=True)

def render_sidebar():
    with st.sidebar:
        st.markdown("### Pengaturan")

        if st.session_state.role == 'admin':
            menu_items = [
                ('', 'Data Input'),
                ('', 'Data Overview'),
                ('', 'Model Performance'),
                ('', 'Sentiment Prediction'),
                ('', 'Word Cloud')
            ]
        else:
            menu_items = [
                ('', 'Data Input'),
                ('', 'Data Overview'),
                ('', 'Sentiment Prediction')
            ]

        st.markdown("#### Database Status")
        with st.expander("Database Status", expanded=False):
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                st.success("Database terhubung")
            else:
                st.error("Database tidak terhubung")
                if st.button("Reconnect", use_container_width=True):
                    st.session_state.db_connection = create_db_connection()
                    st.rerun()

        st.markdown("#### Logout")
        if st.button('Logout', use_container_width=True):
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                st.session_state.db_connection.close()
            st.session_state.clear()
            st.rerun()

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>Analisis Sentimen Mobile JKN</h1>
    </div>
    """, unsafe_allow_html=True)

def render_navbar_compact():
    if st.session_state.role == 'admin':
        pages = {
            "Data Input": "",
            "Data Overview": "", 
            "Model Performance": "",
            "Sentiment Prediction": "",
            "Word Cloud": ""
        }
    else:
        pages = {
            "Data Input": "",
            "Data Overview": "",
            "Sentiment Prediction": ""
        }

    st.markdown("""
    <style>
    .compact-nav {
        display: flex;
        gap: 8px;
        margin-bottom: 20px;
        padding: 8px;
        background-color: #1e1e2f;
        border-radius: 10px;
        justify-content: flex-start;
        align-items: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="compact-nav">', unsafe_allow_html=True)
    
    button_container = st.container()
    with button_container:
        button_cols = st.columns(len(pages), gap="small")
        
        for i, (label, _) in enumerate(pages.items()):
            with button_cols[i]:
                is_current = st.session_state.current_page == label
                if st.button(
                    f"{label}", 
                    key=f"compact_nav_{label}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_page = label
                    st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    analyzer = SentimentAnalyzer()
    dashboard = Dashboard(analyzer)

    if not st.session_state.logged_in:
        dashboard.render_login()
        return

    render_sidebar()
    render_header()
    render_navbar_compact()

    page = st.session_state.current_page
    
    if not page:
        if st.session_state.role == 'admin':
            st.session_state.current_page = 'Data Input'
        else:
            st.session_state.current_page = 'Data Input'
        page = st.session_state.current_page
    
    with st.container():
        if page == 'Data Input':
            dashboard.render_data_input()
        elif page == 'Data Overview':
            dashboard.render_data_overview()
        elif page == 'Model Performance':
            if st.session_state.role == 'admin':
                dashboard.render_model_performance()
            else:
                st.error("Akses ditolak. Admin privileges required.")
        elif page == 'Sentiment Prediction':
            dashboard.render_sentiment_prediction()
        elif page == 'Word Cloud':
            if st.session_state.role == 'admin':
                dashboard.render_wordcloud()
            else:
                st.error("Akses ditolak. Admin privileges required.")
        else:
            st.error("Halaman tidak ditemukan")

if __name__ == "__main__":
    main()