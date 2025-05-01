import streamlit as st

# Set page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="Mobile JKN Sentiment Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding: 1.5rem;
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 1.1em;
        background-color: #4CAF50;
        color: white;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Menu button styling */
    .menu-button {
        width: 100%;
        border-radius: 10px;
        padding: 0.8rem;
        margin-bottom: 0.5rem;
        background-color: #f8f9fa;
        border: 1px solid #ddd;
        text-align: left;
        transition: all 0.3s ease;
    }
    
    .menu-button:hover {
        background-color: #e9ecef;
        border-color: #4CAF50;
    }
    
    .menu-button.active {
        background-color: #4CAF50;
        color: white;
        border-color: #4CAF50;
    }
    
    /* Input field styling */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
        padding: 0.5rem;
    }
    
    /* Selectbox styling */
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    
    /* File uploader styling */
    .stFileUploader>div>div>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Alert styling */
    .stAlert {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Title styling */
    h1 {
        color: #2c3e50;
        margin-bottom: 1.5rem;
    }
    
    h2 {
        color: #34495e;
        margin-bottom: 1rem;
    }
    
    h3 {
        color: #7f8c8d;
        margin-bottom: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)

from models import SentimentAnalyzer
from dashboard import Dashboard
from utils import init_session_state
from database import create_db_connection

def main():
    """Main application entry point"""
    # Initialize session state
    init_session_state()
    
    # Initialize analyzer and dashboard
    analyzer = SentimentAnalyzer()
    dashboard = Dashboard(analyzer)
    
    # Handle authentication
    if not st.session_state.logged_in:
        dashboard.render_login()
        return
    
    # Sidebar navigation with enhanced styling
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2 style='color: #2c3e50;'>Menu</h2>
            </div>
        """, unsafe_allow_html=True)
        
        # Menu items based on role
        if st.session_state.role == 'admin':
            menu_items = [
                ('📤', 'Data Input'),
                ('📊', 'Data Overview'),
                ('📈', 'Model Performance'),
                ('🔮', 'Sentiment Prediction'),
                ('☁️', 'Word Cloud')
            ]
        else:
            # Regular users can only access Sentiment Prediction
            menu_items = [('🔮 Sentiment Prediction', 'Sentiment Prediction')]
            # Set current page to Sentiment Prediction for regular users
            st.session_state.current_page = 'Sentiment Prediction'
        
        # Create menu buttons
        for icon, item in menu_items:
            if st.button(
                f"{icon} {item}",
                key=f"menu_{item}",
                use_container_width=True,
                type="primary" if st.session_state.get('current_page') == item else "secondary"
            ):
                st.session_state.current_page = item
                st.rerun()
        
        # Database status with enhanced styling
        with st.expander("🔌 Database Status", expanded=False):
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                st.success("✅ Connected to MySQL")
            else:
                st.error("❌ Not connected to MySQL")
                if st.button("🔄 Reconnect", use_container_width=True):
                    st.session_state.db_connection = create_db_connection()
                    st.rerun()
        
        # Logout button with enhanced styling
        if st.button('🚪 Logout', use_container_width=True):
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                st.session_state.db_connection.close()
            st.session_state.clear()
            st.rerun()
    
    # Main content area with consistent padding
    with st.container():
        # Get current page from session state
        current_page = st.session_state.get('current_page', 'Data Input')
        
        # Render selected page
        if current_page == 'Data Input':
            dashboard.render_data_input()
        elif current_page == 'Data Overview':
            dashboard.render_data_overview()
        elif current_page == 'Model Performance':
            dashboard.render_model_performance()
        elif current_page == 'Sentiment Prediction':
            dashboard.render_sentiment_prediction()
        elif current_page == 'Word Cloud':
            dashboard.render_wordcloud()

if __name__ == "__main__":
    main() 