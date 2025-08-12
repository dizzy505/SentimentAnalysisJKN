import streamlit as st
from database import register_user, authenticate_user

def render_login(dashboard):
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
            """
                <div style='text-align: center; margin-bottom: 2rem;'>
                    <h1 style='color: #2c3e50;'>Analisis Sentimen Mobile JKN</h1>
                </div>
            """,
            unsafe_allow_html=True,
        )

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.container():
                st.markdown("### Login")
                username = st.text_input(
                    "Username", placeholder="Masukkan username", key="login_username"
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Masukkan password",
                    key="login_password",
                )

                if st.button("Login", use_container_width=True, key="login_button"):
                    if (
                        st.session_state.db_connection
                        and st.session_state.db_connection.is_connected()
                    ):
                        success, user_data = authenticate_user(
                            st.session_state.db_connection, username, password
                        )
                        if success:
                            st.session_state.logged_in = True
                            st.session_state.role = user_data["role"]
                            st.session_state.username = user_data["username"]
                            st.session_state.user_id = user_data["id"]
                            st.success("Login berhasil!")
                            st.rerun()
                        else:
                            st.error("Kredensial tidak valid")
                    else:
                        st.error("Koneksi database tidak tersedia")

        with tab2:
            with st.container():
                st.markdown("### Registrasi Akun Baru")

                reg_username = st.text_input(
                    "Username", placeholder="Pilih username", key="reg_username"
                )
                reg_email = st.text_input(
                    "Email (optional)", placeholder="Masukkan email", key="reg_email"
                )
                reg_password = st.text_input(
                    "Password",
                    type="password",
                    placeholder="Pilih password",
                    key="reg_password",
                )
                reg_confirm_password = st.text_input(
                    "Konfirmasi Password",
                    type="password",
                    placeholder="Konfirmasi password",
                    key="reg_confirm_password",
                )

                password_requirements = """
                **Persyaratan Password:**
                - Minimal 6 karakter
                - Mengandung minimal satu huruf dan satu angka
                """
                st.markdown(password_requirements)

                if st.button("Register", use_container_width=True, key="register_button"):
                    if not reg_username or not reg_password:
                        st.error("Username dan password diperlukan")
                    elif reg_password != reg_confirm_password:
                        st.error("Password tidak cocok")
                    elif len(reg_password) < 6:
                        st.error("Password minimal 6 karakter")
                    elif reg_email and "@" not in reg_email:
                        st.error("Masukkan alamat email yang valid")
                    else:
                        if (
                            st.session_state.db_connection
                            and st.session_state.db_connection.is_connected()
                        ):
                            success, message = register_user(
                                st.session_state.db_connection,
                                reg_username,
                                reg_password,
                                reg_email if reg_email else None,
                            )
                            if success:
                                st.success(message)
                                st.info(
                                    "Anda dapat sekarang login dengan akun baru Anda"
                                )
                            else:
                                st.error(message)
                        else:
                            st.error("Koneksi database tidak tersedia")
