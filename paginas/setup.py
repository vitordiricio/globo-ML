# paginas/setup.py
import streamlit as st
from utils.data_processing import carregar_e_tratar_dados, merge_data
from utils.external_data import fetch_all_bcb_economic_indicators, join_futebol_external_data, join_eventos_externos, join_tweets

def setup_page():
    """
    Dedicated page for file uploads and data processing.
    Stores processed data in the session state for other pages to use.
    """
    st.title("Setup - Carregamento de Dados")
    
    st.markdown("""
    ## Upload de Dados
    
    Faça o upload dos arquivos necessários para análise. Os dados processados serão 
    disponibilizados para todas as outras páginas do dashboard.
    """)
    
    # Explanation about the data merging process
    with st.expander("Como funcionam os uploads e junção dos dados?", expanded=False):
        st.markdown("""
        ### Como os dados são processados?

        1. **Upload de arquivos separados**: Você carrega cada fonte de dados como um arquivo CSV separado.
        
        2. **Processamento individual**: Cada arquivo é tratado separadamente para garantir o formato correto das datas, horas e valores.
        
        3. **Junção dos dados**: Depois de processar os arquivos individuais, o sistema combina todos em uma única tabela através de um "inner join".
        
        ### O que é um "inner join"?
        
        Imagine que cada arquivo representa uma parte do quebra-cabeça. O sistema junta essas partes, mas mantém **apenas os períodos onde todos os dados existem ao mesmo tempo**.
        
        Por exemplo:
        - Se você tem dados de TV Linear de janeiro a março
        - Dados de Redes Sociais de fevereiro a abril
        - Dados de GloboPlay de janeiro a abril
        
        O resultado final terá apenas dados de **fevereiro a março**, que é o período onde todos os dados existem simultaneamente.
        
        ### Por que fazemos isso?
        
        Esta abordagem garante que todas as análises tenham dados completos de todas as fontes, permitindo comparações mais precisas entre TV Linear, GloboPlay e Redes Sociais em cada momento.
        
        ### Enriquecimento de dados
        
        Após a junção principal, adicionamos automaticamente:
        - Indicadores econômicos (inflação, desemprego, dólar)
        - Eventos externos (como jogos de futebol em canais concorrentes)
        - Dados de tweets relacionados
        
        Isso permite analisar como fatores externos influenciam a audiência e o engajamento.
        """)

    # Load and process the data (moved from the main app)
    df_redes_sociais, df_redes_sociais_canais, df_globoplay, df_tv_linear = carregar_e_tratar_dados()

    # Initialize df_merged as None
    df_merged = None
    
    # Check which data sources are available
    data_sources = {
        "Redes Sociais GLOBO": df_redes_sociais is not None,
        "Redes Sociais CANAIS": df_redes_sociais_canais is not None,
        "GloboPlay": df_globoplay is not None,
        "TV Linear": df_tv_linear is not None
    }
    
    # Create df_merged if all needed dataframes are available
    all_loaded = all(data_sources.values())
    if all_loaded:
        
        with st.spinner("Juntando todos os dados em uma única tabela..."):
            df_merged = merge_data(df_redes_sociais, df_redes_sociais_canais, df_globoplay, df_tv_linear)
        
        # Apply external data processing only if df_merged was successfully created
        if df_merged is not None:
            
            # Show a spinner while processing external data
            max_date = df_merged['data_hora'].max().strftime('%d/%m/%Y')
            min_date = df_merged['data_hora'].min().strftime('%d/%m/%Y')
        
            df_merged = fetch_all_bcb_economic_indicators(df_merged, 'data_hora', min_date, max_date)
            df_merged = join_futebol_external_data(df_merged)
            df_merged = join_tweets(df_merged)
            df_merged = join_eventos_externos(df_merged)
            # Fill missing values and complete
            df_merged = df_merged.fillna(0)


            max_date_pos = df_merged['data_hora'].max().strftime('%d/%m/%Y')
            min_date_pos = df_merged['data_hora'].min().strftime('%d/%m/%Y')

            print(f"min date depois {min_date_pos} max date depois {max_date_pos}")
            
            # Display period information
            st.subheader("Períodos dos Dados")
            st.markdown(f"""
            📅 **Período de dados disponível**: De **{min_date}** até **{max_date}**
            
            > **Nota importante**: Este é o período onde todos os dados solicitados se sobrepõem.
            """)
            
            # Store the processed data in session state
            st.session_state.df_merged = df_merged
            st.success("🎉 Todos os dados foram processados com sucesso e estão disponíveis para análise!")
            
            # Show a preview of the data
            st.subheader("Pré-visualização dos Dados")
            st.dataframe(df_merged, use_container_width=True, hide_index=True)
            
            # Display some statistics
            st.subheader("Estatísticas dos Dados")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total de Registros", f"{len(df_merged):,}".replace(",", "."))
            with col2:
                st.metric("Total de Variáveis", len(df_merged.columns))
            with col3:
                st.metric("Período em Dias", (df_merged['data_hora'].max() - df_merged['data_hora'].min()).days + 1)
        else:
            st.error("""
            ⚠️ **Erro ao mesclar os dados**
            
            Não foi possível unir os arquivos. Isso geralmente acontece quando:
            
            1. Os arquivos não têm períodos de datas que se sobrepõem
            2. O formato de data/hora está inconsistente entre os arquivos
            3. Um dos arquivos tem problemas estruturais
            
            Por favor, verifique se todos os arquivos contêm pelo menos algum período de tempo em comum.
            """)
    else:
        st.info("""
        ℹ️ **Aguardando o upload de todos os arquivos**
        
        Por favor, faça o upload de todos os arquivos necessários para prosseguir com a análise.
        
        O dashboard precisa de todos os arquivos para criar uma visão completa dos dados.
        """)