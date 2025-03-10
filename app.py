import streamlit as st

from utils.config_page import configurar_pagina, mostrar_cabecalho
from utils.data_processing import carregar_e_tratar_dados, merge_data
from utils.external_data import fetch_all_bcb_economic_indicators, join_futebol_external_data, join_tweets, join_eventos_externos

from paginas.analise_crencas import analise_fatores_externos, analise_grandes_eventos, analise_percepcao_marca, analise_social_impacto, analise_streaming_vs_linear
from paginas.analise_tv_linear import analise_tv_linear
from paginas.analise_globoplay import analise_globoplay
from paginas.analise_redes_sociais import analise_redes_sociais
from paginas.playground import playground
from paginas.setup import setup_page
from paginas.analise_linear_vs_concorrentes import analise_linear_vs_concorrentes
from paginas.analise_externos import analise_externos

def main():
    configurar_pagina()
    mostrar_cabecalho()

    col1, col2 = st.sidebar.columns(2, vertical_alignment='center', gap='large')
    col1.image("static/fcamara-simple-logo.png", width=40)
    col2.image("static/globo-icone.png", width=50)

    # Restante do conteúdo da sidebar
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
    
    # 1️⃣ 2️⃣ 3️⃣ 4️⃣ 5️⃣ 6️⃣ 7️⃣
    # Menu de navegação na sidebar - Adicionar "Setup" como primeira opção
    menu_options = ["⚙️ SETUP", "1️⃣ TV LINEAR", "2️⃣ TV LINEAR VS CONCORRENTES","3️⃣ GLOBOPLAY", "4️⃣ REDES SOCIAIS", "5️⃣ CRENÇAS", "6️⃣ EXTERNOS", "🛝 PLAYGROUND" ]
    page = st.sidebar.radio("Selecione a página", menu_options)
    
    # Página de Setup
    if page == "⚙️ SETUP":
        setup_page()
    else:
        # Outras páginas usam dados carregados
        df_redes_sociais, df_redes_sociais_canais, df_globoplay, df_tv_linear = carregar_e_tratar_dados()

        # Usar df_merged da session_state se disponível
        if 'df_merged' in st.session_state and st.session_state.df_merged is not None:
            df_merged = st.session_state.df_merged
        else:
            # Initialize df_merged as None
            df_merged = None
            
            # Create df_merged if all needed dataframes are available
            if df_redes_sociais is not None and df_redes_sociais_canais is not None and df_globoplay is not None and df_tv_linear is not None:
                df_merged = merge_data(df_redes_sociais, df_redes_sociais_canais, df_globoplay, df_tv_linear)
                
                # Apply external data processing only if df_merged was successfully created
                if df_merged is not None:
                    max_date = df_merged['data_hora'].max().strftime('%d/%m/%Y')
                    min_date = df_merged['data_hora'].min().strftime('%d/%m/%Y')
                    df_merged = fetch_all_bcb_economic_indicators(df_merged, 'data_hora', min_date, max_date)
                    df_merged = join_futebol_external_data(df_merged)
                    df_merged = join_tweets(df_merged)
                    df_merged = join_eventos_externos(df_merged)
                    df_merged = df_merged.fillna(0)
                    
                    # Store in session state
                    st.session_state.df_merged = df_merged
        
        # Página 🛝 PLAYGROUND
        if page == "🛝 PLAYGROUND" :
            playground(df_merged)
        # Página de TV LINEAR
        elif page == "1️⃣ TV LINEAR":
            st.title("1️⃣ TV LINEAR")
            if df_merged is not None:
                analise_tv_linear(df_merged)
            else:
                st.warning("Por favor, faça o upload dos dados na página 'Setup' primeiro.")

        elif page == "2️⃣ TV LINEAR VS CONCORRENTES":
            st.title("2️⃣ TV LINEAR VS CONCORRENTES")
            if df_merged is not None:
                analise_linear_vs_concorrentes(df_merged)
            else:
                st.warning("Por favor, faça o upload dos dados na página 'Setup' primeiro.")

        # Página de Análise de Redes Sociais
        elif page == "3️⃣ GLOBOPLAY":
            st.title("3️⃣ GLOBOPLAY")
            if df_merged is not None:
                analise_globoplay(df_merged)
            else:
                st.warning("Por favor, faça o upload dos dados na página 'Setup' primeiro.")
        # Página de Análise de Redes Sociais
        elif page == "4️⃣ REDES SOCIAIS":
            st.title("4️⃣ REDES SOCIAIS")
            if df_merged is not None:
                analise_redes_sociais(df_merged)
            else:
                st.warning("Por favor, faça o upload dos dados na página 'Setup' primeiro.")
        # Página Streaming vs TV Linear
        elif page == "5️⃣ CRENÇAS":
            st.title("5️⃣ CRENÇAS")
            if df_merged is not None:
                analise_streaming_vs_linear(df_merged)
                analise_social_impacto(df_merged)
                analise_grandes_eventos(df_merged)
                analise_fatores_externos(df_merged)
                analise_percepcao_marca(df_merged)
            else:
                st.warning("Por favor, faça o upload dos dados na página 'Setup' primeiro.")

        # Página Streaming vs TV Linear
        elif page == "6️⃣ EXTERNOS":
            st.title("6️⃣ EXTERNOS")
            if df_merged is not None:
                analise_externos(df_merged)
            else:
                st.warning("Por favor, faça o upload dos dados na página 'Setup' primeiro.")
    
    st.markdown("---\nFeito com ❤️ FCamara | Value Creation | Sales Boost")

if __name__ == "__main__":
    main()