import streamlit as st
import pandas as pd
from database import batch_insert_to_db

def render_data_input(dashboard):
    st.markdown(
        """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Input</h1>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.data_loaded:
        data_source = st.session_state.get("data_source", "unknown")
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
        st.info(
            "Unggah file CSV dengan kolom 'Label'. Ini akan mengubah data aktif ke file yang diunggah."
        )

        uploaded_file = st.file_uploader(
            "Pilih file CSV", type="csv", help="Pilih file CSV untuk diunggah"
        )

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                required_cols = ["content", "Label"]
                if not all(col in df.columns for col in required_cols):
                    st.error("CSV harus mengandung kolom 'content' dan 'Label'")
                else:
                    df["text_clean"] = df["content"].apply(dashboard.analyzer.preprocess_text)
                    df["text_StopWord"] = df["text_clean"]
                    df["text_tokens"] = df["text_StopWord"]
                    df["text_steamindo"] = df["text_tokens"]

                    st.session_state.original_data = df.copy()

                    positif_samples = df[df["Label"] == "Positif"]
                    negatif_samples = df[df["Label"] == "Negatif"]

                    if len(positif_samples) < 7000:
                        n_samples = 7000 - len(positif_samples)
                        synthetic_samples = positif_samples.sample(
                            n=n_samples, replace=True, random_state=42
                        )
                        df = pd.concat([df, synthetic_samples], ignore_index=True)
                        st.info(
                            f"Oversampling label positif ke {len(df[df['Label'] == 'Positif'])} sampel"
                        )

                    if st.checkbox("Save to database"):
                        if (
                            st.session_state.db_connection
                            and st.session_state.db_connection.is_connected()
                        ):
                            if batch_insert_to_db(st.session_state.db_connection, df):
                                st.success(
                                    f"Berhasil menyimpan {len(df)} records ke database"
                                )
                            else:
                                st.error("Gagal menyimpan ke database")
                        else:
                            st.error("Koneksi database tidak tersedia")

                    st.session_state.data = df
                    st.session_state.data_loaded = True
                    st.session_state.sample_data_used = False
                    st.session_state.data_source = "csv"
                    st.success(
                        "Data berhasil dimuat! Data aktif berubah ke file CSV yang diunggah."
                    )

                    st.markdown("### Data Preview")
                    st.dataframe(
                        df.head()
                        .style.set_properties(
                            **{
                                "background-color": "#f8f9fa",
                                "border-radius": "10px",
                                "padding": "10px",
                            }
                        )
                    )

            except Exception as e:
                st.error(f"Error memuat CSV: {str(e)}")

    with tab2:
        st.markdown("### Database Data")
        st.info("Data otomatis dimuat dari database ketika aplikasi dimulai.")

        if (
            st.session_state.db_connection
            and st.session_state.db_connection.is_connected()
        ):
            if (
                st.session_state.data_loaded
                and st.session_state.get("data_source") == "database"
            ):
                st.success("Database data saat ini aktif")

                if "original_data" in st.session_state and st.session_state.original_data is not None:
                    st.markdown("#### Informasi Data Database Asli")
                    original_data = st.session_state.original_data
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Records (Asli)", len(original_data))
                    with col2:
                        positif_count = len(
                            original_data[original_data["Label"] == "Positif"]
                        )
                        st.metric("Positif (Asli)", positif_count)
                    with col3:
                        negatif_count = len(
                            original_data[original_data["Label"] == "Negatif"]
                        )
                        st.metric("Negatif (Asli)", negatif_count)

                    if "data" in st.session_state and st.session_state.data is not None:
                        st.info(
                            f"ℹLabel positif di oversampling ke {len(st.session_state.data[st.session_state.data['Label'] == 'Positif'])} sampel untuk menyeimbangkan dataset."
                        )

                if st.button("Muat Ulang Data Database", use_container_width=True):
                    dashboard._load_database_data()
                    st.success("Database data berhasil dimuat ulang!")
                    st.rerun()
            else:
                st.info(
                    "ℹDatabase data saat ini tidak aktif. Upload CSV atau scrape data untuk mengubah data aktif."
                )

                if st.button("Switch to Database Data", use_container_width=True):
                    dashboard._load_database_data()
                    if st.session_state.data_loaded:
                        st.success("Berhasil mengubah ke data database!")
                        st.rerun()
        else:
            st.error("Koneksi database tidak tersedia")

    with tab3:
        st.markdown("### Scrape Google Playstore Reviews")
        st.info(
            "Scrape reviews dari Google Play Store. Ini akan mengubah data aktif ke data yang discrape."
        )

        app_id = st.text_input("Masukkan App ID", value="app.bpjs.mobile")
        num_reviews = st.slider("Jumlah Review", 1000, 10000, 5000)

        if st.button("Ambil Review", use_container_width=True):
            try:
                from google_play_scraper import Sort, reviews

                result, _ = reviews(
                    app_id, lang="id", country="id", sort=Sort.NEWEST, count=num_reviews
                )

                df = pd.DataFrame(result)[["content", "score"]]
                df["Label"] = df["score"].apply(lambda x: "Positif" if x >= 4 else "Negatif")
                df["text_clean"] = df["content"].apply(dashboard.analyzer.preprocess_text)
                df["text_StopWord"] = df["text_clean"]
                df["text_tokens"] = df["text_StopWord"]
                df["text_steamindo"] = df["text_tokens"]

                st.session_state.data = df
                st.session_state.original_data = df.copy()
                st.session_state.data_loaded = True
                st.session_state.data_source = "scraped"

                st.success(
                    "Berhasil ambil dan proses data! Data aktif berubah ke data yang discrape."
                )
                st.dataframe(df.head())

                if st.checkbox("Simpan ke database"):
                    if (
                        st.session_state.db_connection
                        and st.session_state.db_connection.is_connected()
                    ):
                        df["score"] = df["score"].astype(int)
                        if batch_insert_to_db(st.session_state.db_connection, df):
                            st.success("Data berhasil disimpan ke database")
                        else:
                            st.error("Gagal menyimpan ke database")
            except Exception as e:
                st.error(f"Gagal scrape data: {e}")
