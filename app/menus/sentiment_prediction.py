import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from database import insert_data_to_db
import logging

logger = logging.getLogger(__name__)


def render_sentiment_prediction(dashboard):
    st.markdown(
        """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Prediksi Sentimen</h1>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.data_loaded:
        st.warning("Silakan memuat atau memasukkan data terlebih dahulu untuk melatih model")
        return

    data_source = st.session_state.get("data_source", "unknown")
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
        help="Masukkan teks yang ingin Anda analisis untuk sentimen",
    )

    save_to_db = st.checkbox("Save result to database", value=True)

    if st.button("Analyze Sentiment", use_container_width=True):
        if not user_input:
            st.warning("Silakan masukkan beberapa teks untuk dianalisis")
            return

        try:
            text_clean = dashboard.analyzer.preprocess_text(user_input)
            text_StopWord = text_clean
            text_tokens = text_StopWord
            text_steamindo = text_tokens

            if not st.session_state.model:
                X_train, _, y_train, _ = train_test_split(
                    st.session_state.data["content"],
                    st.session_state.data["Label"],
                    test_size=0.2,
                    random_state=42,
                )
                st.session_state.model, st.session_state.vectorizer = (
                    dashboard.analyzer.train_model(X_train, y_train)
                )

            tfidf_input = st.session_state.vectorizer.transform([user_input])
            prediction = st.session_state.model.predict(tfidf_input)[0]

            if prediction == "Positif":
                st.success(f"Sentiment: {prediction}")
            else:
                st.error(f"Sentiment: {prediction}")

            probs = st.session_state.model.predict_proba(tfidf_input)[0]
            st.markdown("### Skor Kepercayaan")
            prob_df = pd.DataFrame(
                {"Sentiment": st.session_state.model.classes_, "Confidence": probs}
            )
            st.dataframe(
                prob_df.style.set_properties(
                    **{"background-color": "#f8f9fa", "border-radius": "10px", "padding": "10px"}
                )
            )

            if (
                save_to_db
                and st.session_state.db_connection
                and st.session_state.db_connection.is_connected()
            ):
                if insert_data_to_db(
                    st.session_state.db_connection,
                    user_input,
                    prediction,
                    text_clean,
                    text_StopWord,
                    text_tokens,
                    text_steamindo,
                ):
                    st.success("Hasil berhasil disimpan ke database")
                else:
                    st.error("Gagal menyimpan hasil ke database")

        except Exception as e:
            logger.error(f"Error in sentiment prediction: {str(e)}")
            st.error("Terjadi kesalahan selama analisis. Silakan coba lagi.")
