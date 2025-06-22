import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
import hashlib
from config import DB_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_db_connection():
    """Create a database connection to MySQL"""
    try:
        connection = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        
        if connection.is_connected():
            # Create database if it doesn't exist
            cursor = connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
            cursor.close()
            
            # Connect to the database
            connection.close()
            connection = mysql.connector.connect(
                host=DB_CONFIG['host'],
                user=DB_CONFIG['user'],
                password=DB_CONFIG['password'],
                database=DB_CONFIG['database']
            )
            
            if connection.is_connected():
                logger.info("MySQL Database connection successful")
                create_tables(connection)
                return connection
    except Error as e:
        logger.error(f"Error while connecting to MySQL: {e}")
        return None

def create_tables(connection):
    """Create necessary tables if they don't exist"""
    try:
        cursor = connection.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            email VARCHAR(100) UNIQUE,
            role ENUM('admin', 'user') DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
        )
        """)
        
        # Create sentiment_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            content TEXT NOT NULL,
            score INT,
            Label VARCHAR(10) NOT NULL,
            text_clean TEXT,
            text_StopWord TEXT,
            text_tokens TEXT,
            text_steamindo TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Insert default admin user if not exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = 'admin'")
        admin_exists = cursor.fetchone()[0]
        
        if admin_exists == 0:
            admin_password_hash = hashlib.sha256('adminpass'.encode()).hexdigest()
            cursor.execute("""
            INSERT INTO users (username, password_hash, email, role) 
            VALUES ('admin', %s, 'admin@jkn.com', 'admin')
            """, (admin_password_hash,))
            logger.info("Default admin user created")
        
        connection.commit()
        cursor.close()
        logger.info("Database tables created successfully")
    except Error as e:
        logger.error(f"Error creating tables: {e}")

def register_user(connection, username, password, email=None, role='user'):
    """Register a new user"""
    try:
        cursor = connection.cursor()
        
        # Check if username already exists
        cursor.execute("SELECT COUNT(*) FROM users WHERE username = %s", (username,))
        if cursor.fetchone()[0] > 0:
            cursor.close()
            return False, "Username already exists"
        
        # Check if email already exists (if provided)
        if email:
            cursor.execute("SELECT COUNT(*) FROM users WHERE email = %s", (email,))
            if cursor.fetchone()[0] > 0:
                cursor.close()
                return False, "Email already exists"
        
        # Hash password
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        # Insert new user
        cursor.execute("""
        INSERT INTO users (username, password_hash, email, role) 
        VALUES (%s, %s, %s, %s)
        """, (username, password_hash, email, role))
        
        connection.commit()
        cursor.close()
        logger.info(f"User {username} registered successfully")
        return True, "User registered successfully"
        
    except Error as e:
        logger.error(f"Error registering user: {e}")
        return False, f"Database error: {str(e)}"

def authenticate_user(connection, username, password):
    """Authenticate a user"""
    try:
        cursor = connection.cursor(dictionary=True)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        cursor.execute("""
        SELECT id, username, password_hash, email, role 
        FROM users 
        WHERE username = %s AND password_hash = %s
        """, (username, password_hash))
        
        user = cursor.fetchone()
        cursor.close()
        
        if user:
            logger.info(f"User {username} authenticated successfully")
            return True, user
        else:
            return False, None
            
    except Error as e:
        logger.error(f"Error authenticating user: {e}")
        return False, None

def get_user_by_username(connection, username):
    """Get user by username"""
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT id, username, email, role FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
        return user
    except Error as e:
        logger.error(f"Error getting user: {e}")
        return None

def fetch_data_from_db(connection):
    """Fetch all sentiment data from database"""
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute("SELECT * FROM sentiment_data")
        rows = cursor.fetchall()
        cursor.close()
        
        if rows:
            df = pd.DataFrame(rows)
            return df
        else:
            return pd.DataFrame(columns=['content', 'score', 'Label', 'text_clean', 'text_StopWord', 'text_tokens', 'text_steamindo'])
    except Error as e:
        logger.error(f"Error fetching data from database: {e}")
        return pd.DataFrame(columns=['content', 'score', 'Label', 'text_clean', 'text_StopWord', 'text_tokens', 'text_steamindo'])

def insert_data_to_db(connection, content, label, text_clean, text_StopWord, text_tokens, text_steamindo):
    """Insert a single record into the database"""
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO sentiment_data (content, Label, text_clean, text_StopWord, text_tokens, text_steamindo)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(query, (content, label, text_clean, text_StopWord, text_tokens, text_steamindo))
        connection.commit()
        cursor.close()
        return True
    except Error as e:
        logger.error(f"Error inserting data: {e}")
        return False

def batch_insert_to_db(connection, data_df):
    """Insert multiple records into the database"""
    try:
        cursor = connection.cursor()
        query = """
        INSERT INTO sentiment_data (content, score, Label, text_clean, text_StopWord, text_tokens, text_steamindo)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        data_tuples = list(zip(
            data_df['content'].tolist(),
            data_df['score'].tolist(),
            data_df['Label'].tolist(),
            data_df['text_clean'].tolist(),
            data_df['text_StopWord'].tolist(),
            data_df['text_tokens'].tolist(),
            data_df['text_steamindo'].tolist()
        ))
        
        cursor.executemany(query, data_tuples)
        connection.commit()
        cursor.close()
        return True
    except Error as e:
        logger.error(f"Error batch inserting data: {e}")
        return False 