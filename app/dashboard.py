import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from models import SentimentAnalyzer
from database import create_db_connection, fetch_data_from_db, insert_data_to_db, batch_insert_to_db
from utils import get_csv_download_link
from config import users
import hashlib

# Configure logging
logger = logging.getLogger(__name__)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        font-size: 1.1em;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stSelectbox>div>div>select {
        border-radius: 10px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
    }
    .stFileUploader>div>div>button {
        border-radius: 10px;
    }
    .css-1d391kg {
        padding: 1rem;
        border-radius: 10px;
        background-color: #f0f2f6;
    }
    .stAlert {
        border-radius: 10px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer
        
        # Ensure database connection
        if st.session_state.db_connection is None:
            st.session_state.db_connection = create_db_connection()
        
    def render_login(self):
        """Render login form"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <h1 style='color: #2c3e50;'>Mobile JKN Sentiment Analysis</h1>
                </div>
            """, unsafe_allow_html=True)
            
            with st.container():
                st.markdown("### Login Details")
                username = st.text_input('Username', placeholder='Enter your username')
                password = st.text_input('Password', type='password', placeholder='Enter your password')
                
                if st.button('Login', use_container_width=True):
                    hashed_password = hashlib.sha256(password.encode()).hexdigest()
                    if (username in users and 
                        users[username]['password'] == hashed_password):
                        st.session_state.logged_in = True
                        st.session_state.role = users[username]['role']
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error('Invalid credentials')

    def render_data_input(self):
        """Render data input section"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Input</h1>
            </div>
        """, unsafe_allow_html=True)
        
        tab1, tab2, tab3 = st.tabs(["Upload CSV", "Database Data", "Scrape Review"])
        
        with tab1:
            st.markdown("### Upload CSV File")
            st.markdown("Upload a CSV file with 'content' and 'Label' columns. The 'Label' column should contain 'Positif' or 'Negatif' values.")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv", help="Select a CSV file to upload")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    # Validate columns
                    required_cols = ['content', 'Label']
                    if not all(col in df.columns for col in required_cols):
                        st.error("CSV must contain 'content' and 'Label' columns")
                    else:
                        # Preprocess text
                        df['text_clean'] = df['content'].apply(self.analyzer.preprocess_text)
                        df['text_StopWord'] = df['text_clean']
                        df['text_tokens'] = df['text_StopWord']
                        df['text_steamindo'] = df['text_tokens']
                        
                        st.session_state.original_data = df.copy()

                        # Perform oversampling for positive labels
                        positif_samples = df[df['Label'] == 'Positif']
                        negatif_samples = df[df['Label'] == 'Negatif']
                        
                        if len(positif_samples) < 7000:
                            n_samples = 7000 - len(positif_samples)
                            synthetic_samples = positif_samples.sample(n=n_samples, replace=True, random_state=42)
                            df = pd.concat([df, synthetic_samples], ignore_index=True)
                            st.info(f"Oversampled positive labels to {len(df[df['Label'] == 'Positif'])} samples")
                        
                        # Save to database option
                        if st.checkbox("Save to database"):
                            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                                if batch_insert_to_db(st.session_state.db_connection, df):
                                    st.success(f"Successfully saved {len(df)} records to database")
                                else:
                                    st.error("Failed to save to database")
                            else:
                                st.error("Database connection not available")
                        
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.session_state.sample_data_used = False
                        st.success("Data loaded successfully!")
                        
                        # Display sample
                        st.markdown("### Data Preview")
                        st.dataframe(df.head().style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'border-radius': '10px',
                            'padding': '10px'
                        }))
                        
                except Exception as e:
                    st.error(f"Error loading CSV: {str(e)}")
        
        with tab2:
            st.markdown("### Load Data from Database")
            
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                if st.button("Load All Database Data", use_container_width=True):
                    try:
                        db_data = fetch_data_from_db(st.session_state.db_connection)
                        if not db_data.empty:
                            st.session_state.original_data = db_data.copy()
                            
                            # Perform oversampling for positive labels
                            positif_samples = db_data[db_data['Label'] == 'Positif']
                            negatif_samples = db_data[db_data['Label'] == 'Negatif']
                            
                            if len(positif_samples) < 7000:
                                n_samples = 7000 - len(positif_samples)
                                synthetic_samples = positif_samples.sample(n=n_samples, replace=True, random_state=42)
                                db_data = pd.concat([db_data, synthetic_samples], ignore_index=True)
                                st.info(f"Oversampled positive labels to {len(db_data[db_data['Label'] == 'Positif'])} samples")
                            
                            st.session_state.data = db_data
                            st.session_state.data_loaded = True
                            st.session_state.sample_data_used = False
                            st.success(f"Successfully loaded {len(db_data)} records from database")
                            st.dataframe(db_data.head(10).style.set_properties(**{
                                'background-color': '#f8f9fa',
                                'border-radius': '10px',
                                'padding': '10px'
                            }))
                        else:
                            st.info("ℹ️ No data found in database")
                    except Exception as e:
                        st.error(f"Error loading database data: {str(e)}")
            else:
                st.error("Database connection not available")

        with tab3:
            st.markdown("### Scrape Google Playstore Reviews")
            app_id = st.text_input("Masukkan App ID", value="app.bpjs.mobile")
            num_reviews = st.slider("Jumlah Review", 1000, 10000, 5000)

            if st.button("Ambil Review"):
                try:
                    from google_play_scraper import Sort, reviews

                    result, _ = reviews(
                        app_id,
                        lang='id',
                        country='id',
                        sort=Sort.NEWEST,
                        count=num_reviews
                    )

                    df = pd.DataFrame(result)[['content', 'score']]
                    df['Label'] = df['score'].apply(lambda x: 'Positif' if x >= 4 else 'Negatif')
                    df['text_clean'] = df['content'].apply(self.analyzer.preprocess_text)
                    df['text_StopWord'] = df['text_clean']
                    df['text_tokens'] = df['text_StopWord']
                    df['text_steamindo'] = df['text_tokens']

                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    st.session_state.data_loaded = True

                    st.success("Berhasil ambil dan proses data")
                    st.dataframe(df.head())

                    if st.checkbox("Simpan ke database"):
                        if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                            from database import batch_insert_to_db
                            df['score'] = df['score'].astype(int)
                            if batch_insert_to_db(st.session_state.db_connection, df):
                                st.success("Data berhasil disimpan ke database!")
                            else:
                                st.error("Gagal simpan ke database")
                except Exception as e:
                    st.error(f"Gagal scrape data: {e}")

    def render_data_overview(self):
        """Render data overview section"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Overview</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load or input data first")
            return
        
        # Display sample data
        st.markdown("### Sample Data")
        st.dataframe(st.session_state.data.head().style.set_properties(**{
            'background-color': '#f8f9fa',
            'border-radius': '10px',
            'padding': '10px'
        }))
        
        # Display sentiment distribution
        st.markdown("### Sentiment Distribution")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Ambil data asli sebelum oversampling kalo ada
            data_to_use = st.session_state.original_data if 'original_data' in st.session_state else st.session_state.data
            
            # Tentukan urutan label dan warna secara eksplisit
            labels = ['Positif', 'Negatif']
            colors = ['#4CAF50', '#FF5252']  # Hijau, Merah
            
            # Hitung jumlah per label sesuai urutan
            sentiment_counts = data_to_use['Label'].value_counts()
            values = [sentiment_counts.get(label, 0) for label in labels]

            # Pie chart
            ax.pie(values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

            
        with col2:
            st.markdown("#### Count Statistics")
            st.dataframe(sentiment_counts.to_frame().style.set_properties(**{
                'background-color': '#f8f9fa',
                'border-radius': '10px',
                'padding': '10px'
            }))

    def render_model_performance(self):
        """Render model performance metrics"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Model Performance Analysis</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load or input data first")
            return
        
        if len(st.session_state.data) < 10:
            st.warning("Insufficient data for model training. Please add more data (at least 10 entries).")
            return
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.data['content'], st.session_state.data['Label'], 
            test_size=0.2, random_state=42
        )
        
        # Train model
        try:
            model, vectorizer = self.analyzer.train_model(X_train, y_train)
            
            # Make predictions
            tfidf_test = vectorizer.transform(X_test)
            y_pred = model.predict(tfidf_test)
            
            # Display metrics
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Classification Report")
                st.code(classification_report(y_test, y_pred), language='text')
                
            with col2:
                st.markdown("### Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', ax=ax, 
                           cmap='Blues', cbar=False)
                plt.xlabel('Predicted')
                plt.ylabel('Actual')
                st.pyplot(fig)
                
            # Save model to session state
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            
        except Exception as e:
            logger.error(f"Error in model performance: {str(e)}")
            st.error("Error analyzing model performance. Check your data.")

    def render_sentiment_prediction(self):
        """Render sentiment prediction interface"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Sentiment Prediction</h1>
            </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area(
            "Enter text for analysis:",
            placeholder="Type your text here...",
            help="Enter the text you want to analyze for sentiment"
        )
        
        save_to_db = st.checkbox("Save result to database", value=True)
        
        if st.button('Analyze Sentiment', use_container_width=True):
            if not user_input:
                st.warning('Please enter some text to analyze')
                return
                
            if not st.session_state.data_loaded:
                st.warning("Please load or input data first to train the model")
                return
                
            try:
                # Preprocess input
                text_clean = self.analyzer.preprocess_text(user_input)
                text_StopWord = text_clean
                text_tokens = text_StopWord
                text_steamindo = text_tokens
                
                # Prepare model if not already cached
                if not st.session_state.model:
                    X_train, _, y_train, _ = train_test_split(
                        st.session_state.data['content'], st.session_state.data['Label'], 
                        test_size=0.2, random_state=42
                    )
                    st.session_state.model, st.session_state.vectorizer = (
                        self.analyzer.train_model(X_train, y_train)
                    )
                
                # Make prediction
                tfidf_input = st.session_state.vectorizer.transform([user_input])
                prediction = st.session_state.model.predict(tfidf_input)[0]
                
                # Display result with appropriate styling
                if prediction == 'Positif':
                    st.success(f"Sentiment: {prediction}")
                else:
                    st.error(f"Sentiment: {prediction}")
                
                # Show prediction probabilities
                probs = st.session_state.model.predict_proba(tfidf_input)[0]
                st.markdown("### Confidence Scores")
                prob_df = pd.DataFrame({
                    'Sentiment': st.session_state.model.classes_,
                    'Confidence': probs
                })
                st.dataframe(prob_df.style.set_properties(**{
                    'background-color': '#f8f9fa',
                    'border-radius': '10px',
                    'padding': '10px'
                }))
                
                # Save to database if selected
                if save_to_db and st.session_state.db_connection and st.session_state.db_connection.is_connected():
                    if insert_data_to_db(
                        st.session_state.db_connection, 
                        user_input, 
                        prediction, 
                        text_clean,
                        text_StopWord,
                        text_tokens,
                        text_steamindo
                    ):
                        st.success("Result saved to database")
                    else:
                        st.error("Failed to save result to database")
                
            except Exception as e:
                logger.error(f"Error in sentiment prediction: {str(e)}")
                st.error("An error occurred during analysis. Please try again.")

    def render_wordcloud(self):
        """Render word cloud visualization"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Word Cloud Visualization</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load or input data first")
            return
        
        sentiment = st.radio(
            'Choose sentiment to visualize:',
            ['Positive', 'Negative'],
            horizontal=True
        )
        
        try:
            # Filter data based on sentiment
            text_data = ' '.join(
                st.session_state.data[st.session_state.data['Label'] == 
                         ('Positif' if sentiment == 'Positive' else 'Negatif')]
                ['text_steamindo']
            )
            
            if not text_data:
                st.warning(f"No {sentiment.lower()} sentiment data available")
                return
            
            # Generate word cloud
            wordcloud = WordCloud(
                width=800, height=400,
                background_color="white",
                colormap="Greens" if sentiment == 'Positive' else "Reds",
                max_words=100,
                contour_width=3,
                contour_color='steelblue'
            ).generate(text_data)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)
            
        except Exception as e:
            logger.error(f"Error generating word cloud: {str(e)}")
            st.error("Failed to generate word cloud visualization") 