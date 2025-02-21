# utils/ui.py
import streamlit as st

def mostrar_cabecalho():
    """
    Exibe o cabeçalho com logos e título da aplicação.
    """
    
    st.title("🔍 Globo dashboard")
    st.markdown("Faça o upload da sua base de dados CSV para entender melhor seus dados.")

def configurar_pagina():
    """
    Configura a página do Streamlit com título, layout e ícone.
    Também carrega estilos customizados se disponíveis.
    """
    st.set_page_config(
        page_title="Globo dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="./static/globo-icone.png"
    )
    try:
        with open('styles/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass

    if 'models' not in st.session_state:
        st.session_state.models = {}