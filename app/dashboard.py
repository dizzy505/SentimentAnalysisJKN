import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from models import SentimentAnalyzer
from database import create_db_connection, fetch_data_from_db, insert_data_to_db, batch_insert_to_db, register_user, authenticate_user
from utils import get_csv_download_link
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
        
        # Auto-load data from database if not already loaded
        if not st.session_state.data_loaded and st.session_state.db_connection and st.session_state.db_connection.is_connected():
            self._load_database_data()
        
    def _load_database_data(self):
        """Load data from database automatically"""
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
                
                st.session_state.data = db_data
                st.session_state.data_loaded = True
                st.session_state.sample_data_used = False
                st.session_state.data_source = "database"
                logger.info(f"Auto-loaded {len(db_data)} records from database")
            else:
                st.session_state.data_source = "none"
                logger.info("No data found in database")
        except Exception as e:
            logger.error(f"Error auto-loading database data: {str(e)}")
            st.session_state.data_source = "none"

    def render_login(self):
        """Render login form with registration option"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <h1 style='color: #2c3e50;'>Mobile JKN Sentiment Analysis</h1>
                </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for login and registration
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.container():
                    st.markdown("### Login")
                    username = st.text_input('Username', placeholder='Enter your username', key='login_username')
                    password = st.text_input('Password', type='password', placeholder='Enter your password', key='login_password')
                    
                    if st.button('Login', use_container_width=True, key='login_button'):
                        if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                            success, user_data = authenticate_user(st.session_state.db_connection, username, password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.role = user_data['role']
                                st.session_state.username = user_data['username']
                                st.session_state.user_id = user_data['id']
                                st.success("Login successful!")
                                st.rerun()
                            else:
                                st.error('Invalid credentials')
                        else:
                            st.error('Database connection not available')
            
            with tab2:
                with st.container():
                    st.markdown("### Register New Account")
                    
                    reg_username = st.text_input('Username', placeholder='Choose a username', key='reg_username')
                    reg_email = st.text_input('Email (optional)', placeholder='Enter your email', key='reg_email')
                    reg_password = st.text_input('Password', type='password', placeholder='Choose a password', key='reg_password')
                    reg_confirm_password = st.text_input('Confirm Password', type='password', placeholder='Confirm your password', key='reg_confirm_password')
                    
                    # Password validation
                    password_requirements = """
                    **Password Requirements:**
                    - At least 6 characters long
                    - Contains at least one letter and one number
                    """
                    st.markdown(password_requirements)
                    
                    if st.button('Register', use_container_width=True, key='register_button'):
                        # Validation
                        if not reg_username or not reg_password:
                            st.error('Username and password are required')
                        elif reg_password != reg_confirm_password:
                            st.error('Passwords do not match')
                        elif len(reg_password) < 6:
                            st.error('Password must be at least 6 characters long')
                        elif not any(c.isalpha() for c in reg_password) or not any(c.isdigit() for c in reg_password):
                            st.error('Password must contain at least one letter and one number')
                        elif reg_email and '@' not in reg_email:
                            st.error('Please enter a valid email address')
                        else:
                            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                                success, message = register_user(
                                    st.session_state.db_connection, 
                                    reg_username, 
                                    reg_password, 
                                    reg_email if reg_email else None
                                )
                                if success:
                                    st.success(message)
                                    st.info("You can now login with your new account")
                                else:
                                    st.error(message)
                            else:
                                st.error('Database connection not available')

    def render_data_input(self):
        """Render data input section"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Input</h1>
            </div>
        """, unsafe_allow_html=True)
        
        # Display current data source status
        if st.session_state.data_loaded:
            data_source = st.session_state.get('data_source', 'unknown')
            if data_source == "database":
                st.info("**Current Active Data:** Database (auto-loaded)")
            elif data_source == "csv":
                st.info("**Current Active Data:** Uploaded CSV File")
            elif data_source == "scraped":
                st.info("**Current Active Data:** Scraped Data")
            else:
                st.info("**Current Active Data:** Database (auto-loaded)")
        else:
            st.warning("No data currently loaded")
        
        tab1, tab2, tab3 = st.tabs(["Upload CSV", "Database Data", "Scrape Review"])
        
        with tab1:
            st.markdown("### Upload CSV File")
            st.info("Upload a CSV file with 'Label' column. This will switch active data to the uploaded file.")
            
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
                        
                        # Switch active data to CSV
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.session_state.sample_data_used = False
                        st.session_state.data_source = "csv"
                        st.success("Data loaded successfully! Active data switched to uploaded CSV file.")
                        
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
            st.markdown("### Database Data")
            st.info("Data is automatically loaded from database when the application starts.")
            
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                if st.session_state.data_loaded and st.session_state.get('data_source') == "database":
                    st.success("Database data is currently active")
                    
                    # Show current data info
                    if 'original_data' in st.session_state and st.session_state.original_data is not None:
                        st.markdown("#### Original Database Data Info")
                        original_data = st.session_state.original_data
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records (Original)", len(original_data))
                        with col2:
                            positif_count = len(original_data[original_data['Label'] == 'Positif'])
                            st.metric("Positive (Original)", positif_count)
                        with col3:
                            negatif_count = len(original_data[original_data['Label'] == 'Negatif'])
                            st.metric("Negative (Original)", negatif_count)
                        
                        if 'data' in st.session_state and st.session_state.data is not None:
                            st.info(f"ℹFor model training, the positive class is oversampled to {len(st.session_state.data[st.session_state.data['Label'] == 'Positif'])} samples to balance the dataset.")

                        # Option to refresh database data
                        if st.button("Refresh Database Data", use_container_width=True):
                            self._load_database_data()
                            st.success("Database data refreshed!")
                            st.rerun()
                else:
                    st.info("ℹDatabase data is not currently active. Upload CSV or scrape data to switch active data.")
                    
                    # Option to switch back to database data
                    if st.button("Switch to Database Data", use_container_width=True):
                        self._load_database_data()
                        if st.session_state.data_loaded:
                            st.success("Switched to database data!")
                            st.rerun()
            else:
                st.error("Database connection not available")

        with tab3:
            st.markdown("### Scrape Google Playstore Reviews")
            st.info("Scrape reviews from Google Play Store. This will switch active data to the scraped data.")
            
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

                    # Switch active data to scraped data
                    st.session_state.data = df
                    st.session_state.original_data = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.data_source = "scraped"

                    st.success("Berhasil ambil dan proses data! Active data switched to scraped data.")
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
        
        # Display current data source
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "database":
            st.info("**Analyzing:** Database Data (auto-loaded)")
        elif data_source == "csv":
            st.info("**Analyzing:** Uploaded CSV File")
        elif data_source == "scraped":
            st.info("**Analyzing:** Scraped Data")
        else:
            st.info("**Analyzing:** Database Data (auto-loaded)")
        
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
            
        st.markdown("---")
        st.markdown("### Browse Data")
        
        # Use original_data for viewing to show data before oversampling
        data_to_view = st.session_state.original_data if 'original_data' in st.session_state else st.session_state.data
        
        search_query = st.text_input("Search in review content:", placeholder="Type here to search...")
        
        if search_query:
            # Filter data based on search query (case-insensitive)
            filtered_data = data_to_view[data_to_view['content'].str.contains(search_query, case=False, na=False)]
            st.dataframe(filtered_data.style.set_properties(**{
                'background-color': '#f8f9fa',
                'border-radius': '10px',
                'padding': '10px'
            }))
        else:
            # Show all data if search query is empty
            st.dataframe(data_to_view.style.set_properties(**{
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
        
        # Display current data source
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "database":
            st.info("**Training Model on:** Database Data (auto-loaded)")
        elif data_source == "csv":
            st.info("**Training Model on:** Uploaded CSV File")
        elif data_source == "scraped":
            st.info("**Training Model on:** Scraped Data")
        else:
            st.info("**Training Model on:** Database Data (auto-loaded)")
        
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
        
        if not st.session_state.data_loaded:
            st.warning("Please load or input data first to train the model")
            return
        
        # Display current data source
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "database":
            st.info("**Model trained on:** Database Data (auto-loaded)")
        elif data_source == "csv":
            st.info("**Model trained on:** Uploaded CSV File")
        elif data_source == "scraped":
            st.info("**Model trained on:** Scraped Data")
        else:
            st.info("**Model trained on:** Database Data (auto-loaded)")
        
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
        """Render wordcloud visualization"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Word Cloud Visualization</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Please load data first")
            return
        
        # Create tabs for positive and negative word clouds
        tab1, tab2 = st.tabs(["Positive Sentiment", "Negative Sentiment"])
        
        with tab1:
            st.markdown("### Positive Sentiment Word Cloud")
            positive_data = st.session_state.data[st.session_state.data['Label'] == 'Positif']
            if not positive_data.empty:
                # Combine all positive text
                positive_text = ' '.join(positive_data['text_clean'].astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='Greens',
                    max_words=100
                ).generate(positive_text)
                
                # Display
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                # Save option
                if st.button("Save Positive Word Cloud", key="save_pos_wordcloud"):
                    wordcloud.to_file("images/wordcloud_positif.png")
                    st.success("Word cloud saved as 'images/wordcloud_positif.png'")
            else:
                st.warning("No positive sentiment data available")
        
        with tab2:
            st.markdown("### Negative Sentiment Word Cloud")
            negative_data = st.session_state.data[st.session_state.data['Label'] == 'Negatif']
            if not negative_data.empty:
                # Combine all negative text
                negative_text = ' '.join(negative_data['text_clean'].astype(str))
                
                # Create word cloud
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='Reds',
                    max_words=100
                ).generate(negative_text)
                
                # Display
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                # Save option
                if st.button("Save Negative Word Cloud", key="save_neg_wordcloud"):
                    wordcloud.to_file("images/wordcloud_negatif.png")
                    st.success("Word cloud saved as 'images/wordcloud_negatif.png'")
            else:
                st.warning("No negative sentiment data available") 