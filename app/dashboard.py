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

logger = logging.getLogger(__name__)

st.markdown("""
<link href=\"https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap\" rel=\"stylesheet\">
<style>
body {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
    background-color: #19223a;
}
.main {
    padding: 2rem;
}
.stButton>button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 1.1em;
    font-weight: 600;
    background: #22305a;
    color: #fff;
    border: none;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
    transition: all 0.2s;
}
.stButton>button:hover {
    background: #2a3962;
    color: #fff;
    transform: translateY(-2px) scale(1.03);
}
.stTextInput>div>div>input, .stSelectbox>div>div>select, .stTextArea>div>div>textarea {
    border-radius: 12px;
    border: 1.5px solid #e0e6ed;
    padding: 0.7rem 1rem;
    font-size: 1rem;
    background: #fff;
    margin-bottom: 0.5rem;
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
}
.stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, .stTextArea>div>div>textarea:focus {
    border: 1.5px solid #28407a;
    outline: none;
}
.stFileUploader>div>div>button {
    border-radius: 12px;
    background: #22305a;
    color: #fff;
    font-weight: 600;
    border: none;
    box-shadow: 0 2px 8px rgba(44,62,80,0.08);
}
.stFileUploader>div>div>button:hover {
    background: #2a3962;
}
.css-1d391kg, .stAlert, .metric-card, .stDataFrame, .stExpander, .stTabs {
    border-radius: 16px !important;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
}
.metric-card {
    background-color: #22305a;
    padding: 1.2rem;
    border-radius: 16px;
    margin: 0.5rem;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
    color: #fff;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Poppins', 'Segoe UI', Arial, sans-serif;
    font-weight: 600;
    font-size: 1.05rem;
    color: #fff;
    border-radius: 10px 10px 0 0;
    background: #22305a;
    margin-right: 4px;
    padding: 0.7rem 1.2rem;
    transition: background 0.2s;
}
.stTabs [aria-selected="true"] {
    background: #28407a;
    color: #fff;
}
.stExpander {
    background: #22305a;
    border-radius: 14px !important;
    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
    color: #fff;
}
</style>
""", unsafe_allow_html=True)

class Dashboard:
    def __init__(self, analyzer: SentimentAnalyzer):
        self.analyzer = analyzer
        
        if st.session_state.db_connection is None:
            st.session_state.db_connection = create_db_connection()
        
        if not st.session_state.data_loaded and st.session_state.db_connection and st.session_state.db_connection.is_connected():
            self._load_database_data()
        
    def _load_database_data(self):
        """Otomatis memuat data dari database"""
        try:
            db_data = fetch_data_from_db(st.session_state.db_connection)
            if not db_data.empty:
                st.session_state.original_data = db_data.copy()
                
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
                logger.info(f"Otomatis memuat {len(db_data)} records dari database")
            else:
                st.session_state.data_source = "none"
                logger.info("Tidak ada data ditemukan di database")
        except Exception as e:
            logger.error(f"Error otomatis memuat data dari database: {str(e)}")
            st.session_state.data_source = "none"

    def render_login(self):
        """Render form login dengan opsi registrasi"""
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <h1 style='color: #2c3e50;'>Analisis Sentimen Mobile JKN</h1>
                </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["Login", "Register"])
            
            with tab1:
                with st.container():
                    st.markdown("### Login")
                    username = st.text_input('Username', placeholder='Masukkan username', key='login_username')
                    password = st.text_input('Password', type='password', placeholder='Masukkan password', key='login_password')
                    
                    if st.button('Login', use_container_width=True, key='login_button'):
                        if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                            success, user_data = authenticate_user(st.session_state.db_connection, username, password)
                            if success:
                                st.session_state.logged_in = True
                                st.session_state.role = user_data['role']
                                st.session_state.username = user_data['username']
                                st.session_state.user_id = user_data['id']
                                st.success("Login berhasil!")
                                st.rerun()
                            else:
                                st.error('Kredensial tidak valid')
                        else:
                            st.error('Koneksi database tidak tersedia')
            
            with tab2:
                with st.container():
                    st.markdown("### Registrasi Akun Baru")
                    
                    reg_username = st.text_input('Username', placeholder='Pilih username', key='reg_username')
                    reg_email = st.text_input('Email (optional)', placeholder='Masukkan email', key='reg_email')
                    reg_password = st.text_input('Password', type='password', placeholder='Pilih password', key='reg_password')
                    reg_confirm_password = st.text_input('Konfirmasi Password', type='password', placeholder='Konfirmasi password', key='reg_confirm_password')
                    
                    password_requirements = """
                    **Persyaratan Password:**
                    - Minimal 6 karakter
                    - Mengandung minimal satu huruf dan satu angka
                    """
                    st.markdown(password_requirements)
                    
                    if st.button('Register', use_container_width=True, key='register_button'):
                        if not reg_username or not reg_password:
                            st.error('Username dan password diperlukan')
                        elif reg_password != reg_confirm_password:
                            st.error('Password tidak cocok')
                        elif len(reg_password) < 6:
                            st.error('Password minimal 6 karakter')
                        elif reg_email and '@' not in reg_email:
                            st.error('Masukkan alamat email yang valid')
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
                                    st.info("Anda dapat sekarang login dengan akun baru Anda")
                                else:
                                    st.error(message)
                            else:
                                st.error('Koneksi database tidak tersedia')

    def render_data_input(self):
        """Render bagian data input"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Input</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.data_loaded:
            data_source = st.session_state.get('data_source', 'unknown')
            if data_source == "database":
                st.info("**Data Aktif Saat Ini:** Database (otomatis dimuat)")
            elif data_source == "csv":
                st.info("**Data Aktif Saat Ini:** File CSV yang diunggah")
            elif data_source == "scraped":
                st.info("**Data Aktif Saat Ini:** Data yang discrape")
            else:
                st.info("**Data Aktif Saat Ini:** Database (otomatis dimuat)")
        else:
            st.warning("Tidak ada data yang dimuat saat ini")
        
        tab1, tab2, tab3 = st.tabs(["Upload CSV", "Database Data", "Scrape Review"])
        
        with tab1:
            st.markdown("### Upload CSV File")
            st.info("Unggah file CSV dengan kolom 'Label'. Ini akan mengubah data aktif ke file yang diunggah.")
            
            uploaded_file = st.file_uploader("Pilih file CSV", type="csv", help="Pilih file CSV untuk diunggah")
            
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    
                    required_cols = ['content', 'Label']
                    if not all(col in df.columns for col in required_cols):
                        st.error("CSV harus mengandung kolom 'content' dan 'Label'")
                    else:
                        df['text_clean'] = df['content'].apply(self.analyzer.preprocess_text)
                        df['text_StopWord'] = df['text_clean']
                        df['text_tokens'] = df['text_StopWord']
                        df['text_steamindo'] = df['text_tokens']
                        
                        st.session_state.original_data = df.copy()

                        positif_samples = df[df['Label'] == 'Positif']
                        negatif_samples = df[df['Label'] == 'Negatif']
                        
                        if len(positif_samples) < 7000:
                            n_samples = 7000 - len(positif_samples)
                            synthetic_samples = positif_samples.sample(n=n_samples, replace=True, random_state=42)
                            df = pd.concat([df, synthetic_samples], ignore_index=True)
                            st.info(f"Oversampling label positif ke {len(df[df['Label'] == 'Positif'])} sampel")
                        
                        if st.checkbox("Save to database"):
                            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                                if batch_insert_to_db(st.session_state.db_connection, df):
                                    st.success(f"Berhasil menyimpan {len(df)} records ke database")
                                else:
                                    st.error("Gagal menyimpan ke database")
                            else:
                                st.error("Koneksi database tidak tersedia")
                        
                        st.session_state.data = df
                        st.session_state.data_loaded = True
                        st.session_state.sample_data_used = False
                        st.session_state.data_source = "csv"
                        st.success("Data berhasil dimuat! Data aktif berubah ke file CSV yang diunggah.")
                        
                        st.markdown("### Data Preview")
                        st.dataframe(df.head().style.set_properties(**{
                            'background-color': '#f8f9fa',
                            'border-radius': '10px',
                            'padding': '10px'
                        }))
                        
                except Exception as e:
                    st.error(f"Error memuat CSV: {str(e)}")
        
        with tab2:
            st.markdown("### Database Data")
            st.info("Data otomatis dimuat dari database ketika aplikasi dimulai.")
            
            if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                if st.session_state.data_loaded and st.session_state.get('data_source') == "database":
                    st.success("Database data saat ini aktif")
                    
                    if 'original_data' in st.session_state and st.session_state.original_data is not None:
                        st.markdown("#### Informasi Data Database Asli")
                        original_data = st.session_state.original_data
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Records (Asli)", len(original_data))
                        with col2:
                            positif_count = len(original_data[original_data['Label'] == 'Positif'])
                            st.metric("Positif (Asli)", positif_count)
                        with col3:
                            negatif_count = len(original_data[original_data['Label'] == 'Negatif'])
                            st.metric("Negatif (Asli)", negatif_count)
                        
                        if 'data' in st.session_state and st.session_state.data is not None:
                            st.info(f"ℹLabel positif di oversampling ke {len(st.session_state.data[st.session_state.data['Label'] == 'Positif'])} sampel untuk menyeimbangkan dataset.")

                        if st.button("Muat Ulang Data Database", use_container_width=True):
                            self._load_database_data()
                            st.success("Database data berhasil dimuat ulang!")
                            st.rerun()
                else:
                    st.info("ℹDatabase data saat ini tidak aktif. Upload CSV atau scrape data untuk mengubah data aktif.")
                    
                    if st.button("Switch to Database Data", use_container_width=True):
                        self._load_database_data()
                        if st.session_state.data_loaded:
                            st.success("Berhasil mengubah ke data database!")
                            st.rerun()
            else:
                st.error("Koneksi database tidak tersedia")

        with tab3:
            st.markdown("### Scrape Google Playstore Reviews")
            st.info("Scrape reviews dari Google Play Store. Ini akan mengubah data aktif ke data yang discrape.")
            
            app_id = st.text_input("Masukkan App ID", value="app.bpjs.mobile")
            num_reviews = st.slider("Jumlah Review", 1000, 10000, 5000)

            if st.button("Ambil Review", use_container_width=True):
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
                    st.session_state.data_source = "scraped"

                    st.success("Berhasil ambil dan proses data! Data aktif berubah ke data yang discrape.")
                    st.dataframe(df.head())

                    if st.checkbox("Simpan ke database"):
                        if st.session_state.db_connection and st.session_state.db_connection.is_connected():
                            from database import batch_insert_to_db
                            df['score'] = df['score'].astype(int)
                            if batch_insert_to_db(st.session_state.db_connection, df):
                                st.success("Data berhasil disimpan ke database")
                            else:
                                st.error("Gagal menyimpan ke database")
                except Exception as e:
                    st.error(f"Gagal scrape data: {e}")

    def render_data_overview(self):
        """Render bagian data overview"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Sentimen</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Silakan memuat atau memasukkan data terlebih dahulu")
            return
        
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "database":
            st.info("**Menganalisis:** Database Data (otomatis dimuat)")
        elif data_source == "csv":
            st.info("**Menganalisis:** File CSV yang diunggah")
        elif data_source == "scraped":
            st.info("**Menganalisis:** Data yang discrape")
        else:
            st.info("**Menganalisis:** Database Data (otomatis dimuat)")
        
        st.markdown("### Distribusi Sentimen")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            data_to_use = st.session_state.original_data if 'original_data' in st.session_state else st.session_state.data
            
            labels = ['Positif', 'Negatif']
            colors = ['#2E8B57', '#DC143C']
            
            sentiment_counts = data_to_use['Label'].value_counts()
            values = [sentiment_counts.get(label, 0) for label in labels]

            fig, ax = plt.subplots(figsize=(10, 8))
            
            explode = (0.05, 0.05)
            
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=colors, 
                startangle=90,
                explode=explode,
                shadow=True,
                textprops={'fontsize': 12, 'fontweight': 'bold'},
                pctdistance=0.85
            )
            
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
                autotext.set_fontsize(11)
            
            ax.set_title('Distribusi Sentimen', fontsize=16, fontweight='bold', pad=20)
            ax.axis('equal')
            
            ax.legend(wedges, labels, title="Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            
            plt.tight_layout()
            st.pyplot(fig)

            
        with col2:
            st.markdown("#### Statistik Jumlah")
            
            positif_count = sentiment_counts.get('Positif', 0)
            negatif_count = sentiment_counts.get('Negatif', 0)
            total_count = positif_count + negatif_count
            
            stat_col1, stat_col2, stat_col3 = st.columns(3)
            
            with stat_col1:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #2E8B57, #3CB371);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <h3 style="margin: 0; font-size: 24px;">{}</h3>
                        <p style="margin: 5px 0 0 0; font-size: 14px;">Positif</p>
                    </div>
                """.format(positif_count), unsafe_allow_html=True)
            
            with stat_col2:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #DC143C, #FF6347);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <h3 style="margin: 0; font-size: 24px;">{}</h3>
                        <p style="margin: 5px 0 0 0; font-size: 14px;">Negatif</p>
                    </div>
                """.format(negatif_count), unsafe_allow_html=True)
            
            with stat_col3:
                st.markdown("""
                    <div style="
                        background: linear-gradient(135deg, #4682B4, #5F9EA0);
                        padding: 20px;
                        border-radius: 15px;
                        text-align: center;
                        color: white;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    ">
                        <h3 style="margin: 0; font-size: 24px;">{}</h3>
                        <p style="margin: 5px 0 0 0; font-size: 14px;">Total</p>
                    </div>
                """.format(total_count), unsafe_allow_html=True)
            
        st.markdown("---")
        st.markdown("### Cari Data")
        
        data_to_view = st.session_state.original_data if 'original_data' in st.session_state else st.session_state.data
        
        search_query = st.text_input("Cari dalam konten review:", placeholder="Ketik di sini untuk mencari...")
        
        if search_query:
            filtered_data = data_to_view[data_to_view['content'].str.contains(search_query, case=False, na=False)]
            
            styled_data = filtered_data.style.set_properties(**{
                'background-color': '#f8f9fa',
                'border-radius': '8px',
                'padding': '12px',
                'border': '1px solid #e9ecef',
                'font-size': '14px'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#343a40'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '12px'),
                    ('border-radius', '8px 8px 0 0')
                ]},
                {'selector': 'td', 'props': [
                    ('border-bottom', '1px solid #dee2e6'),
                    ('text-align', 'left')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', '#e3f2fd'),
                    ('transition', 'background-color 0.3s ease')
                ]}
            ])
            
            st.dataframe(styled_data, use_container_width=True)
        else:
            styled_data = data_to_view.style.set_properties(**{
                'background-color': '#f8f9fa',
                'border-radius': '8px',
                'padding': '12px',
                'border': '1px solid #e9ecef',
                'font-size': '14px'
            }).set_table_styles([
                {'selector': 'th', 'props': [
                    ('background-color', '#343a40'),
                    ('color', 'white'),
                    ('font-weight', 'bold'),
                    ('text-align', 'center'),
                    ('padding', '12px'),
                    ('border-radius', '8px 8px 0 0')
                ]},
                {'selector': 'td', 'props': [
                    ('border-bottom', '1px solid #dee2e6'),
                    ('text-align', 'left')
                ]},
                {'selector': 'tr:hover', 'props': [
                    ('background-color', '#e3f2fd'),
                    ('transition', 'background-color 0.3s ease')
                ]}
            ])
            
            st.dataframe(styled_data, use_container_width=True)

    def render_model_performance(self):
        """Render metrik performa model"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Analisis Performa Model</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Silakan memuat atau memasukkan data terlebih dahulu")
            return
        
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "database":
            st.info("**Training Model pada:** Database Data (otomatis dimuat)")
        elif data_source == "csv":
            st.info("**Training Model pada:** File CSV yang diunggah")
        elif data_source == "scraped":
            st.info("**Training Model pada:** Data yang discrape")
        else:
            st.info("**Training Model pada:** Database Data (otomatis dimuat)")
        
        if len(st.session_state.data) < 10:
            st.warning("Data tidak cukup untuk pelatihan model. Silakan tambahkan data (minimal 10 entri).")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(
            st.session_state.data['content'], st.session_state.data['Label'], 
            test_size=0.2, random_state=42
        )
        
        try:
            model, vectorizer = self.analyzer.train_model(X_train, y_train)
            tfidf_test = vectorizer.transform(X_test)
            y_pred = model.predict(tfidf_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.markdown("### Performa Model Keseluruhan")
            col1, col2, col3 = st.columns(3)
            with col2:
                st.markdown(f'''
                    <div style="
                        background: linear-gradient(135deg, #4CAF50, #45a049);
                        padding: 32px 0 24px 0;
                        border-radius: 20px;
                        text-align: center;
                        color: #222;
                        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
                        margin: 20px 0;
                    ">
                        <span style="font-size: 2.5rem; font-weight: 700;">{accuracy:.1%}</span><br>
                        <span style="font-size: 1.1rem; font-weight: bold;">Akurasi</span>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown("### Laporan Klasifikasi")
            report = classification_report(y_test, y_pred, output_dict=True)
            labels = [lbl for lbl in report.keys() if lbl not in ['accuracy', 'macro avg', 'weighted avg']]
            metrics = ['precision', 'recall', 'f1-score', 'support']
            card_colors = {
                'Positif': 'linear-gradient(135deg, #2E8B57, #3CB371)',
                'Negatif': 'linear-gradient(135deg, #DC143C, #FF6347)',
                'avg / total': 'linear-gradient(135deg, #4682B4, #5F9EA0)'
            }
            for label in labels:
                st.markdown(f"#### {label}")
                card_cols = st.columns(4)
                for i, metric in enumerate(metrics):
                    value = report[label][metric]
                    if metric == 'support':
                        value_str = f"{int(value)}"
                    else:
                        value_str = f"{value:.3f}"
                    card_color = card_colors.get(label, 'linear-gradient(135deg, #4682B4, #5F9EA0)')
                    card_html = f'''
                        <div style="
                            background: {card_color};
                            padding: 28px 0 18px 0;
                            border-radius: 20px;
                            text-align: center;
                            color: #222;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.10);
                            margin: 10px 0;
                        ">
                            <span style="font-size: 2rem; font-weight: 700;">{value_str}</span><br>
                            <span style="font-size: 1rem; font-weight: bold;">{metric.capitalize()}</span>
                        </div>
                    '''
                    card_cols[i].markdown(card_html, unsafe_allow_html=True)
            st.markdown("### Matriks Konfusi")
            cm = confusion_matrix(y_test, y_pred, labels=['Negatif', 'Positif'])
            cm_labels = [['True Neg', 'False Pos'], ['False Neg', 'True Pos']]
            cm_colors = [['linear-gradient(135deg, #4682B4, #5F9EA0)', 'linear-gradient(135deg, #DC143C, #FF6347)'],
                         ['linear-gradient(135deg, #DC143C, #FF6347)', 'linear-gradient(135deg, #2E8B57, #3CB371)']]
            cm_cols = st.columns(2)
            for i in range(2):
                for j in range(2):
                    value = cm[i, j]
                    label = cm_labels[i][j]
                    color = cm_colors[i][j]
                    card_html = f'''
                        <div style="
                            background: {color};
                            padding: 32px 0 24px 0;
                            border-radius: 20px;
                            text-align: center;
                            color: #222;
                            box-shadow: 0 4px 8px rgba(0,0,0,0.10);
                            margin: 10px 0;
                        ">
                            <span style="font-size: 2.5rem; font-weight: 700;">{value}</span><br>
                            <span style="font-size: 1.1rem; font-weight: bold;">{label}</span>
                        </div>
                    '''
                    cm_cols[j].markdown(card_html, unsafe_allow_html=True)
            st.markdown("""
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin-top: 15px;">
                    <p style="margin: 0; font-size: 14px;">
                        <strong>Interpretasi Matriks:</strong><br>
                        • <strong>True Neg</strong>: Prediksi Negatif benar<br>
                        • <strong>False Pos</strong>: Prediksi Positif salah<br>
                        • <strong>False Neg</strong>: Prediksi Negatif salah<br>
                        • <strong>True Pos</strong>: Prediksi Positif benar
                    </p>
                </div>
            """, unsafe_allow_html=True)
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
        except Exception as e:
            logger.error(f"Error in model performance: {str(e)}")
            st.error("Error menganalisis performa model. Periksa data Anda.")

    def render_sentiment_prediction(self):
        """Render sentiment prediction interface"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Prediksi Sentimen</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Silakan memuat atau memasukkan data terlebih dahulu untuk melatih model")
            return
        
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "database":
            st.info("**Model dilatih pada:** Database Data (otomatis dimuat)")
        elif data_source == "csv":
            st.info("**Model dilatih pada:** File CSV yang diunggah")
        elif data_source == "scraped":
            st.info("**Model dilatih pada:** Data yang discrape")
        else:
            st.info("**Model dilatih pada:** Database Data (otomatis dimuat)")
        
        user_input = st.text_area(
            "Masukkan teks untuk analisis:",
            placeholder="Ketik teks Anda di sini...",
            help="Masukkan teks yang ingin Anda analisis untuk sentimen"
        )
        
        save_to_db = st.checkbox("Save result to database", value=True)
        
        if st.button('Analyze Sentiment', use_container_width=True):
            if not user_input:
                st.warning('Silakan masukkan beberapa teks untuk dianalisis')
                return
                
            try:
                text_clean = self.analyzer.preprocess_text(user_input)
                text_StopWord = text_clean
                text_tokens = text_StopWord
                text_steamindo = text_tokens
                
                if not st.session_state.model:
                    X_train, _, y_train, _ = train_test_split(
                        st.session_state.data['content'], st.session_state.data['Label'], 
                        test_size=0.2, random_state=42
                    )
                    st.session_state.model, st.session_state.vectorizer = (
                        self.analyzer.train_model(X_train, y_train)
                    )
                
                tfidf_input = st.session_state.vectorizer.transform([user_input])
                prediction = st.session_state.model.predict(tfidf_input)[0]
                
                if prediction == 'Positif':
                    st.success(f"Sentiment: {prediction}")
                else:
                    st.error(f"Sentiment: {prediction}")
                
                probs = st.session_state.model.predict_proba(tfidf_input)[0]
                st.markdown("### Skor Kepercayaan")
                prob_df = pd.DataFrame({
                    'Sentiment': st.session_state.model.classes_,
                    'Confidence': probs
                })
                st.dataframe(prob_df.style.set_properties(**{
                    'background-color': '#f8f9fa',
                    'border-radius': '10px',
                    'padding': '10px'
                }))
                
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
                        st.success("Hasil berhasil disimpan ke database")
                    else:
                        st.error("Gagal menyimpan hasil ke database")
                
            except Exception as e:
                logger.error(f"Error in sentiment prediction: {str(e)}")
                st.error("Terjadi kesalahan selama analisis. Silakan coba lagi.")

    def render_wordcloud(self):
        """Render wordcloud visualization"""
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Visualisasi Word Cloud</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("Silakan memuat data terlebih dahulu")
            return
        
        tab1, tab2 = st.tabs(["Positive Sentiment", "Negative Sentiment"])
        
        with tab1:
            st.markdown("### Word Cloud Sentimen Positif")
            positive_data = st.session_state.data[st.session_state.data['Label'] == 'Positif']
            if not positive_data.empty:
                positive_text = ' '.join(positive_data['text_clean'].astype(str))
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='Greens',
                    max_words=100
                ).generate(positive_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                if st.button("Simpan Word Cloud Sentimen Positif", key="save_pos_wordcloud"):
                    wordcloud.to_file("images/wordcloud_positif.png")
                    st.success("Word cloud berhasil disimpan sebagai 'images/wordcloud_positif.png'")
            else:
                st.warning("Tidak ada data sentimen positif yang tersedia")
        
        with tab2:
            st.markdown("### Word Cloud Sentimen Negatif")
            negative_data = st.session_state.data[st.session_state.data['Label'] == 'Negatif']
            if not negative_data.empty:
                negative_text = ' '.join(negative_data['text_clean'].astype(str))
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400, 
                    background_color='white',
                    colormap='Reds',
                    max_words=100
                ).generate(negative_text)
                
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                
                if st.button("Simpan Word Cloud Sentimen Negatif", key="save_neg_wordcloud"):
                    wordcloud.to_file("images/wordcloud_negatif.png")
                    st.success("Word cloud berhasil disimpan sebagai 'images/wordcloud_negatif.png'")
            else:
                st.warning("Tidak ada data sentimen negatif yang tersedia") 