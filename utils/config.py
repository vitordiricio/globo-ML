# utils/config.py
import streamlit as st

def configurar_pagina():
    """
    Configura a página do Streamlit com título, layout e ícone.
    Também carrega estilos customizados se disponíveis.
    """
    st.set_page_config(
        page_title="ML Globo",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="./assets/globo-icone.png"
    )
    try:
        with open('styles/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass

    if 'models' not in st.session_state:
        st.session_state.models = {}
