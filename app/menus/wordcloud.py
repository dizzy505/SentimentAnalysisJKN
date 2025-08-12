import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def render_wordcloud(dashboard):
    """Render wordcloud visualization"""
    st.markdown(
        """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Visualisasi Word Cloud</h1>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.data_loaded:
        st.warning("Silakan memuat data terlebih dahulu")
        return

    tab1, tab2 = st.tabs(["Positive Sentiment", "Negative Sentiment"])

    with tab1:
        st.markdown("### Word Cloud Sentimen Positif")
        positive_data = st.session_state.data[st.session_state.data["Label"] == "Positif"]
        if not positive_data.empty:
            positive_text = " ".join(positive_data["text_clean"].astype(str))

            wordcloud = WordCloud(
                width=800, height=400, background_color="white", colormap="Greens", max_words=100
            ).generate(positive_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            if st.button("Simpan Word Cloud Sentimen Positif", key="save_pos_wordcloud"):
                wordcloud.to_file("images/wordcloud_positif.png")
                st.success(
                    "Word cloud berhasil disimpan sebagai 'images/wordcloud_positif.png'"
                )
        else:
            st.warning("Tidak ada data sentimen positif yang tersedia")

    with tab2:
        st.markdown("### Word Cloud Sentimen Negatif")
        negative_data = st.session_state.data[st.session_state.data["Label"] == "Negatif"]
        if not negative_data.empty:
            negative_text = " ".join(negative_data["text_clean"].astype(str))

            wordcloud = WordCloud(
                width=800, height=400, background_color="white", colormap="Reds", max_words=100
            ).generate(negative_text)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig)

            if st.button("Simpan Word Cloud Sentimen Negatif", key="save_neg_wordcloud"):
                wordcloud.to_file("images/wordcloud_negatif.png")
                st.success(
                    "Word cloud berhasil disimpan sebagai 'images/wordcloud_negatif.png'"
                )
        else:
            st.warning("Tidak ada data sentimen negatif yang tersedia")
