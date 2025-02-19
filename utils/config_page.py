# utils/ui.py
import streamlit as st

def mostrar_cabecalho():
    """
    Exibe o cabeçalho com logos e título da aplicação.
    """
    col1, _, _ = st.columns([1, 2, 1])
    with col1:
        col_logo1, col_logo2 = st.columns([0.6, 4])
        col_logo1.image("assets/globo-icone.png", width=80)
        col_logo2.image("assets/fcamara-simple-logo.png", width=50)
    
    st.title("🔍 ML Globo")
    st.markdown("Faça o upload da sua base de dados CSV para entender melhor seus dados.")

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