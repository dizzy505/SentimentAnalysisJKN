import hashlib

# Database Configuration
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': 'localicad',
    'database': 'sentiment_analysis'
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