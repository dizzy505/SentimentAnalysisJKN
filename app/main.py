import streamlit as st

# ==== PAGE CONFIG ====
st.set_page_config(
    page_title="Mobile JKN Sentiment Analysis",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

from models import SentimentAnalyzer
from dashboard import Dashboard
from utils import init_session_state
from database import create_db_connection


# ==== SESSION INIT ====
init_session_state()
if "current_page" not in st.session_state:
    st.session_state.current_page = "Data Input"

# ==== CUSTOM CSS ====
st.markdown("""
<style>
body {
    background-color: #f5f6fa;
}
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 15px;
    margin-bottom: 1rem;
    color: white;
    text-align: center;
}
.nav-container {
    display: flex;
    gap: 0.5rem;
    margin-bottom: 20px;
    padding: 0.5rem;
    background-color: #1e1e2f;
    border-radius: 10px;
    justify-content: flex-start;
    flex-wrap: wrap;
}
.nav-button {
    padding: 0.6rem 1.2rem;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    border: 2px solid transparent;
    transition: 0.2s ease-in-out;
    font-size: 0.9rem;
    flex-shrink: 0;
    white-space: nowrap;
}
.nav-button:hover {
    border: 2px solid #60a5fa;
    background-color: #374151;
    cursor: pointer;
}
.nav-button.active {
    background-color: #2563eb;
    border: 2px solid #60a5fa;
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
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.element-container {
    margin: 0 !important;
}
</style>
""", unsafe_allow_html=True)

# ==== SIDEBAR ====
def render_sidebar():
    with st.sidebar:
        st.markdown("### Settings")

        if st.session_state.role == 'admin':
            menu_items = [
                ('', 'Data Input'),
                ('', 'Data Overview'),
                ('', 'Model Performance'),
                ('', 'Sentiment Prediction'),
                ('', 'Word Cloud')
            ]
        else:
            menu_items = [('', 'Sentiment Prediction')]
            st.session_state.current_page = 'Sentiment Prediction'

        st.markdown("#### Database Status")
        with st.expander("Database Status", expanded=False):
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                st.success("Connected to MySQL")
            else:
                st.error("Not connected to MySQL")
                if st.button("Reconnect", use_container_width=True):
                    st.session_state.db_connection = create_db_connection()
                    st.rerun()

        st.markdown("#### Logout")
        if st.button('Logout', use_container_width=True):
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                st.session_state.db_connection.close()
            st.session_state.clear()
            st.rerun()

# ==== HEADER ====
def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>Mobile JKN Sentiment Analysis</h1>
    </div>
    """, unsafe_allow_html=True)

# ==== NAVBAR ====
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

# ==== MAIN ====
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
    with st.container():
        if page == 'Data Input':
            dashboard.render_data_input()
        elif page == 'Data Overview':
            dashboard.render_data_overview()
        elif page == 'Model Performance':
            dashboard.render_model_performance()
        elif page == 'Sentiment Prediction':
            dashboard.render_sentiment_prediction()
        elif page == 'Word Cloud':
            dashboard.render_wordcloud()

if __name__ == "__main__":
    main()