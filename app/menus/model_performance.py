import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def render_model_performance(dashboard):
    """Render metrik performa model"""
    st.markdown(
        """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Analisis Performa Model</h1>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.data_loaded:
        st.warning("Silakan memuat atau memasukkan data terlebih dahulu")
        return

    data_source = st.session_state.get("data_source", "unknown")
    if data_source == "database":
        st.info("**Training Model pada:** Database Data (otomatis dimuat)")
    elif data_source == "csv":
        st.info("**Training Model pada:** File CSV yang diunggah")
    elif data_source == "scraped":
        st.info("**Training Model pada:** Data yang discrape")
    else:
        st.info("**Training Model pada:** Database Data (otomatis dimuat)")

    if len(st.session_state.data) < 10:
        st.warning(
            "Data tidak cukup untuk pelatihan model. Silakan tambahkan data (minimal 10 entri)."
        )
        return

    X_train, X_test, y_train, y_test = train_test_split(
        st.session_state.data["content"],
        st.session_state.data["Label"],
        test_size=0.2,
        random_state=42,
    )

    try:
        model, vectorizer = dashboard.analyzer.train_model(X_train, y_train)
        tfidf_test = vectorizer.transform(X_test)
        y_pred = model.predict(tfidf_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.markdown("### Performa Model Keseluruhan")
        col1, col2, col3 = st.columns(3)
        with col2:
            st.markdown(
                f'''
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
                ''',
                unsafe_allow_html=True,
            )

        st.markdown("### Laporan Klasifikasi")
        report = classification_report(y_test, y_pred, output_dict=True)
        labels = [
            lbl for lbl in report.keys() if lbl not in ["accuracy", "macro avg", "weighted avg"]
        ]
        metrics = ["precision", "recall", "f1-score", "support"]
        card_colors = {
            "Positif": "linear-gradient(135deg, #2E8B57, #3CB371)",
            "Negatif": "linear-gradient(135deg, #DC143C, #FF6347)",
            "avg / total": "linear-gradient(135deg, #4682B4, #5F9EA0)",
        }
        for label in labels:
            st.markdown(f"#### {label}")
            card_cols = st.columns(4)
            for i, metric in enumerate(metrics):
                value = report[label][metric]
                if metric == "support":
                    value_str = f"{int(value)}"
                else:
                    value_str = f"{value:.3f}"
                card_color = card_colors.get(
                    label, "linear-gradient(135deg, #4682B4, #5F9EA0)"
                )
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
        cm = confusion_matrix(y_test, y_pred, labels=["Negatif", "Positif"])
        cm_labels = [["True Neg", "False Pos"], ["False Neg", "True Pos"]]
        cm_colors = [
            [
                "linear-gradient(135deg, #4682B4, #5F9EA0)",
                "linear-gradient(135deg, #DC143C, #FF6347)",
            ],
            [
                "linear-gradient(135deg, #DC143C, #FF6347)",
                "linear-gradient(135deg, #2E8B57, #3CB371)",
            ],
        ]
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
        st.markdown(
            """
                <div style="background-color: #e3f2fd; padding: 15px; border-radius: 8px; border-left: 4px solid #2196F3; margin-top: 15px;">
                    <p style="margin: 0; font-size: 14px;">
                        <strong>Interpretasi Matriks:</strong><br>
                        • <strong>True Neg</strong>: Prediksi Negatif benar<br>
                        • <strong>False Pos</strong>: Prediksi Positif salah<br>
                        • <strong>False Neg</strong>: Prediksi Negatif salah<br>
                        • <strong>True Pos</strong>: Prediksi Positif benar
                    </p>
                </div>
            """,
            unsafe_allow_html=True,
        )
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
    except Exception as e:
        logger.error(f"Error in model performance: {str(e)}")
        st.error("Error menganalisis performa model. Periksa data Anda.")
