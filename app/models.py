import re
import logging
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from typing import Tuple
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        self.factory = StemmerFactory()
        self.stemmer = self.factory.create_stemmer()
        
    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data"""
        try:
            # Remove HTML tags
            text = re.sub('<[^>]*>', '', text)
            # Extract emoticons
            emoticons = re.findall('(?::|;|=)()(?:-)?(?:\\)|\\(|D|P)', text)
            # Convert to lowercase and join emoticons
            text = (re.sub('[\\W]+', ' ', text.lower()) + 
                   ' '.join(emoticons).replace('-', ''))
            # Apply stemming
            text = self.stemmer.stem(text)
            return text.strip()
        except Exception as e:
            logger.error(f"Error in text preprocessing: {str(e)}")
            raise

    def train_model(self, X_train: pd.Series, y_train: pd.Series) -> Tuple:
        """Train the sentiment analysis model"""
        try:
            tfidf_vectorizer = TfidfVectorizer(max_features=5000)
            tfidf_train = tfidf_vectorizer.fit_transform(X_train)
            nb = MultinomialNB()
            nb.fit(tfidf_train, y_train)
            return nb, tfidf_vectorizer
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

def create_sample_data() -> pd.DataFrame:
    """Create sample data for demonstration"""
    sample_data = {
        'content': [
            'Pelayanan JKN sangat memuaskan dan membantu saya',
            'Aplikasi JKN Mobile sangat mudah digunakan',
            'Proses klaim asuransi kesehatan cepat dan efisien',
            'Antrean di rumah sakit terlalu panjang dan melelahkan',
            'Pelayanan lambat dan petugas tidak ramah',
            'Prosedur pendaftaran rumit dan membingungkan',
            'Fasilitas kesehatan sangat bersih dan nyaman',
            'Dokter sangat profesional dan informatif',
            'Biaya obat-obatan masih terlalu mahal',
            'Sistem rujukan tidak efektif dan membuang waktu'
        ],
        'Label': [
            'Positif', 'Positif', 'Positif', 'Negatif', 'Negatif', 
            'Negatif', 'Positif', 'Positif', 'Negatif', 'Negatif'
        ]
    }
    
    # Create dataframe
    df = pd.DataFrame(sample_data)
    
    # Add preprocessed text column
    analyzer = SentimentAnalyzer()
    df['text_steamindo'] = df['content'].apply(analyzer.preprocess_text)
    
    return df 