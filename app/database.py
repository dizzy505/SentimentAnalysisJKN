import mysql.connector
from mysql.connector import Error
import pandas as pd
import logging
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
        
        connection.commit()
        cursor.close()
        logger.info("Database tables created successfully")
    except Error as e:
        logger.error(f"Error creating tables: {e}")

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