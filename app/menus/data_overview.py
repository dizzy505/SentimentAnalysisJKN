import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def render_data_overview(dashboard):
    """Render bagian data overview"""
    st.markdown(
        """
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h1 style='color: #2c3e50;'>Data Sentimen</h1>
            </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.data_loaded:
        st.warning("Silakan memuat atau memasukkan data terlebih dahulu")
        return

    data_source = st.session_state.get("data_source", "unknown")
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
        data_to_use = (
            st.session_state.original_data
            if "original_data" in st.session_state
            else st.session_state.data
        )

        labels = ["Positif", "Negatif"]
        colors = ["#2E8B57", "#DC143C"]

        sentiment_counts = data_to_use["Label"].value_counts()
        values = [sentiment_counts.get(label, 0) for label in labels]

        fig, ax = plt.subplots(figsize=(10, 8))

        explode = (0.05, 0.05)

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=90,
            explode=explode,
            shadow=True,
            textprops={"fontsize": 12, "fontweight": "bold"},
            pctdistance=0.85,
        )

        for autotext in autotexts:
            autotext.set_color("white")
            autotext.set_fontweight("bold")
            autotext.set_fontsize(11)

        ax.set_title("Distribusi Sentimen", fontsize=16, fontweight="bold", pad=20)
        ax.axis("equal")

        ax.legend(wedges, labels, title="Sentimen", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown("#### Statistik Jumlah")

        positif_count = sentiment_counts.get("Positif", 0)
        negatif_count = sentiment_counts.get("Negatif", 0)
        total_count = positif_count + negatif_count

        stat_col1, stat_col2, stat_col3 = st.columns(3)

        with stat_col1:
            st.markdown(
                """
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
                """.format(
                    positif_count
                ),
                unsafe_allow_html=True,
            )

        with stat_col2:
            st.markdown(
                """
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
                """.format(
                    negatif_count
                ),
                unsafe_allow_html=True,
            )

        with stat_col3:
            st.markdown(
                """
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
                """.format(
                    total_count
                ),
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("### Cari Data")

    data_to_view = (
        st.session_state.original_data
        if "original_data" in st.session_state
        else st.session_state.data
    )

    search_query = st.text_input(
        "Cari dalam konten review:", placeholder="Ketik di sini untuk mencari..."
    )

    if search_query:
        filtered_data = data_to_view[
            data_to_view["content"].str.contains(search_query, case=False, na=False)
        ]

        styled_data = (
            filtered_data.style.set_properties(
                **{
                    "background-color": "#f8f9fa",
                    "border-radius": "8px",
                    "padding": "12px",
                    "border": "1px solid #e9ecef",
                    "font-size": "14px",
                }
            )
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#343a40"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "12px"),
                        ("border-radius", "8px 8px 0 0"),
                    ],
                },
                {"selector": "td", "props": [("border-bottom", "1px solid #dee2e6"), ("text-align", "left")]},
                {
                    "selector": "tr:hover",
                    "props": [
                        ("background-color", "#e3f2fd"),
                        ("transition", "background-color 0.3s ease"),
                    ],
                },
            ]
        )

        st.dataframe(styled_data, use_container_width=True)
    else:
        styled_data = (
            data_to_view.style.set_properties(
                **{
                    "background-color": "#f8f9fa",
                    "border-radius": "8px",
                    "padding": "12px",
                    "border": "1px solid #e9ecef",
                    "font-size": "14px",
                }
            )
        ).set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#343a40"),
                        ("color", "white"),
                        ("font-weight", "bold"),
                        ("text-align", "center"),
                        ("padding", "12px"),
                        ("border-radius", "8px 8px 0 0"),
                    ],
                },
                {"selector": "td", "props": [("border-bottom", "1px solid #dee2e6"), ("text-align", "left")]},
                {
                    "selector": "tr:hover",
                    "props": [
                        ("background-color", "#e3f2fd"),
                        ("transition", "background-color 0.3s ease"),
                    ],
                },
            ]
        )

        st.dataframe(styled_data, use_container_width=True)
