import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt

from utils.config_page import configurar_pagina, mostrar_cabecalho

from paginas.setup import setup_page
from paginas.analise_crencas import analise_fatores_externos, analise_grandes_eventos, analise_percepcao_marca, analise_social_impacto, analise_streaming_vs_linear
from paginas.analise_tv_linear import analise_tv_linear
from paginas.analise_globoplay import analise_globoplay
from paginas.analise_redes_sociais import analise_redes_sociais
from paginas.playground import playground

def main():
    configurar_pagina()
    mostrar_cabecalho()

    col1, col2 = st.sidebar.columns(2, vertical_alignment='center', gap='large')
    col1.image("static/fcamara-simple-logo.png", width=40)
    col2.image("static/globo-icone.png", width=50)

    # Sidebar content
    st.sidebar.markdown(
        """
        <div class="sidebar-header">
            <h1>Globo Dashboard</h1>
            <p>Value Creation | Sales Boost</p>
        </div>
        <hr class="sidebar-hr">
        """,
        unsafe_allow_html=True
    )
    
    # Menu de navegação na sidebar - UPDATED to include Setup
    menu_options = ["⚙️ Setup", "1️⃣ TV LINEAR", "2️⃣ GLOBOPLAY", "3️⃣ REDES SOCIAIS", "4️⃣ CRENÇAS", "Playground"]
    page = st.sidebar.radio("Selecione a página", menu_options)
    
    # Initialize df_merged if not in session state
    if 'df_merged' not in st.session_state:
        st.session_state.df_merged = None
    
    # Get data from session state
    df_merged = st.session_state.df_merged
    
    # Display appropriate page based on selection
    if page == "⚙️ Setup":
        setup_page()
    
    # Playground page
    elif page == "Playground":
        playground(df_merged)

    # TV LINEAR page
    elif page == "1️⃣ TV LINEAR":
        st.title("1️⃣ TV LINEAR")
        if df_merged is not None:
            analise_tv_linear(df_merged)
        else:
            st.warning("Por favor, carregue os dados na página Setup primeiro.")

    # GLOBOPLAY page
    elif page == "2️⃣ GLOBOPLAY":
        st.title("2️⃣ GLOBOPLAY")
        if df_merged is not None:
            analise_globoplay(df_merged)
        else:
            st.warning("Por favor, carregue os dados na página Setup primeiro.")
    
    # REDES SOCIAIS page
    elif page == "3️⃣ REDES SOCIAIS":
        st.title("3️⃣ REDES SOCIAIS")
        if df_merged is not None:
            analise_redes_sociais(df_merged)
        else:
            st.warning("Por favor, carregue os dados na página Setup primeiro.")

    # CRENÇAS page
    elif page == "4️⃣ CRENÇAS":
        st.title("4️⃣ CRENÇAS")
        if df_merged is not None:
            analise_streaming_vs_linear(df_merged)
            analise_social_impacto(df_merged)
            analise_grandes_eventos(df_merged)
            analise_fatores_externos(df_merged)
            analise_percepcao_marca(df_merged)
        else:
            st.warning("Por favor, carregue os dados na página Setup primeiro.")
    
    st.markdown("---\nFeito com ❤️ FCamara | Value Creation | Sales Boost")

if __name__ == "__main__":
    main()