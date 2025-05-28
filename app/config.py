import hashlib
import streamlit as st
# Database Configuration
DB_CONFIG = {
    'host': st.secrets["mysql"]["host"],
    'user': st.secrets["mysql"]["user"],
    'password': st.secrets["mysql"]["password"],
    'database': st.secrets["mysql"]["database"]
}

# User Configuration
users = {
    'admin': {
        'password': hashlib.sha256('adminpass'.encode()).hexdigest(),
        'role': 'admin'
    },
    'user': {
        'password': hashlib.sha256('userpass'.encode()).hexdigest(),
        'role': 'user'
    }
} 
