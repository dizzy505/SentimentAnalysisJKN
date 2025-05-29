# app/database.py
from firebase_config import init_firestore
import pandas as pd
import logging

logger = logging.getLogger(__name__)
db = init_firestore()

def fetch_data_from_db():
    """Ambil semua data dari Firestore"""
    try:
        docs = db.collection("sentiment_data").stream()
        data = [doc.to_dict() for doc in docs]
        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Error fetch data: {e}")
        return pd.DataFrame()

def insert_data_to_db(content, label, text_clean, text_StopWord, text_tokens, text_steamindo):
    """Insert satu data ke Firestore"""
    try:
        db.collection("sentiment_data").add({
            "content": content,
            "Label": label,
            "text_clean": text_clean,
            "text_StopWord": text_StopWord,
            "text_tokens": text_tokens,
            "text_steamindo": text_steamindo
        })
        return True
    except Exception as e:
        logger.error(f"Error insert data: {e}")
        return False

def batch_insert_to_db(df):
    """Insert banyak data ke Firestore"""
    try:
        batch = db.batch()
        for _, row in df.iterrows():
            doc_ref = db.collection("sentiment_data").document()
            batch.set(doc_ref, row.to_dict())
        batch.commit()
        return True
    except Exception as e:
        logger.error(f"Error batch insert: {e}")
        return False

def create_db_connection():
    return db