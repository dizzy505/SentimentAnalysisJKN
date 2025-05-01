import hashlib
import os

# Database Configuration
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', 'localicad'),
    'database': os.getenv('DB_NAME', 'sentiment_analysis')
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
