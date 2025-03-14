import streamlit as st
import pandas as pd
from utils.data_processing import tratar_redes_sociais_linear, tratar_redes_sociais_canais, tratar_globoplay, tratar_tv_linear, tratar_tv_linear_semanal
from utils.external_data import fetch_all_bcb_economic_indicators, join_grade_external_data, join_tweets, join_eventos_externos
from utils.data_processing import merge_data

def setup_page():
    
    st.title("Configuração e Upload de Dados")
    
    st.markdown("""
    Esta página permite realizar o upload dos arquivos CSV necessários para o funcionamento do dashboard.
    Uma vez carregados, os dados permanecerão disponíveis em todas as outras páginas do aplicativo.
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
    
    # Initialize session state variables if they don't exist
    if 'df_redes_sociais' not in st.session_state:
        st.session_state.df_redes_sociais = None
    if 'df_redes_sociais_canais' not in st.session_state:
        st.session_state.df_redes_sociais_canais = None
    if 'df_globoplay' not in st.session_state:
        st.session_state.df_globoplay = None
    if 'df_tv_linear' not in st.session_state:
        st.session_state.df_tv_linear = None
    if 'df_tv_linear_semanal' not in st.session_state:
        st.session_state.df_tv_linear_semanal = None
    if 'df_merged' not in st.session_state:
        st.session_state.df_merged = None
    if 'granularidade' not in st.session_state:
        st.session_state.granularidade = "data_hora"
    
    # Initialize file upload status - using different keys
    if 'rs_status' not in st.session_state:
        st.session_state.rs_status = False
    if 'rs_canais_status' not in st.session_state:
        st.session_state.rs_canais_status = False
    if 'globoplay_status' not in st.session_state:
        st.session_state.globoplay_status = False
    if 'tv_linear_status' not in st.session_state:
        st.session_state.tv_linear_status = False
    if 'tv_linear_semanal_status' not in st.session_state:
        st.session_state.tv_linear_semanal_status = False
    if 'needs_merge' not in st.session_state:
        st.session_state.needs_merge = False
    
    # Granularity selection section
    st.subheader("Selecione a Granularidade dos Dados")
    
    # Dropdown for granularity selection
    granularidade = st.radio(
        "Escolha a granularidade para todas as análises:",
        options=["data_hora", "semana"],
        index=0,  # Default to data_hora
        horizontal=True,
        help="data_hora: dados originais por hora | semana: dados agrupados por semana"
    )
    
    # Save granularity to session state
    if granularidade != st.session_state.granularidade:
        st.session_state.granularidade = granularidade
        st.session_state.df_merged = None  # Reset merged data when granularity changes
        st.session_state.needs_merge = True
        st.rerun()
    
    # Display appropriate warning based on granularity
    if granularidade == "data_hora":
        st.warning("""
        ⚠️ **ATENÇÃO:** No modo de granularidade por hora, os dados do Globoplay são divididos igualmente 
        por 24 (horas do dia), o que pode causar inconsistências nas análises e nos modelos. Use esta 
        granularidade apenas quando necessário para análises horárias específicas.
        """)
    else:
        st.success("""
        ✅ No modo de granularidade por semana, os dados são agrupados semanalmente, o que proporciona
        análises mais consistentes e representativas das tendências gerais, especialmente para os dados
        do Globoplay que são originalmente diários.
        """)
    
    # Create columns for file uploads
    st.subheader("Upload de Arquivos")
    col1, col2, col3 = st.columns(3)
    
    # First column - Redes Sociais
    with col1:
        st.markdown("### Redes Sociais GLOBO")
        
        # Check if data is already loaded
        if st.session_state.rs_status:
            st.success("✅ Arquivo carregado")
            if st.button("Remover", key="remove_rs"):
                st.session_state.df_redes_sociais = None
                st.session_state.rs_status = False
                st.session_state.df_merged = None
                st.session_state.needs_merge = False
                st.rerun()
        else:
            arquivo_redes_sociais = st.file_uploader("Upload CSV", type=['csv'], key='uploader_rs')
            if arquivo_redes_sociais is not None:
                try:
                    df_redes_sociais = pd.read_csv(arquivo_redes_sociais)
                    st.session_state.df_redes_sociais = tratar_redes_sociais_linear(df_redes_sociais)
                    st.session_state.rs_status = True
                    st.session_state.df_merged = None  # Reset merged data
                    st.session_state.needs_merge = True
                    st.success("Arquivo processado com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {str(e)}")
    
        st.markdown("### Redes Sociais CANAIS")
        
        # Check if data is already loaded
        if st.session_state.rs_canais_status:
            st.success("✅ Arquivo carregado")
            if st.button("Remover", key="remove_rs_canais"):
                st.session_state.df_redes_sociais_canais = None
                st.session_state.rs_canais_status = False
                st.session_state.df_merged = None
                st.session_state.needs_merge = False
                st.rerun()
        else:
            arquivo_redes_sociais_canais = st.file_uploader("Upload CSV", type=['csv'], key='uploader_rs_canais')
            if arquivo_redes_sociais_canais is not None:
                try:
                    # Added low_memory=False to fix the DtypeWarning
                    df_redes_sociais_canais = pd.read_csv(arquivo_redes_sociais_canais, 
                              encoding='latin-1',
                              sep=';',
                              quotechar='"',
                              doublequote=True,
                              lineterminator='\n',
                              escapechar='\\',
                              on_bad_lines='skip',
                              low_memory=False)
                    st.session_state.df_redes_sociais_canais = tratar_redes_sociais_canais(df_redes_sociais_canais)
                    st.session_state.rs_canais_status = True
                    st.session_state.df_merged = None  # Reset merged data
                    st.session_state.needs_merge = True
                    st.success("Arquivo processado com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {str(e)}")
    
    # Second column - GloboPlay and TV Linear
    with col2:
        st.markdown("### GloboPlay")
        
        # Check if data is already loaded
        if st.session_state.globoplay_status:
            st.success("✅ Arquivo carregado")
            if st.button("Remover", key="remove_globoplay"):
                st.session_state.df_globoplay = None
                st.session_state.globoplay_status = False
                st.session_state.df_merged = None
                st.session_state.needs_merge = False
                st.rerun()
        else:
            arquivo_globoplay = st.file_uploader("Upload CSV", type=['csv'], key='uploader_globoplay')
            if arquivo_globoplay is not None:
                try:
                    df_globoplay = pd.read_csv(arquivo_globoplay)
                    st.session_state.df_globoplay = tratar_globoplay(df_globoplay)
                    st.session_state.globoplay_status = True
                    st.session_state.df_merged = None  # Reset merged data
                    st.session_state.needs_merge = True
                    st.success("Arquivo processado com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {str(e)}")
        
        st.markdown("### TV Linear (Dados Horários)")
        
        # Check if data is already loaded
        if st.session_state.tv_linear_status:
            st.success("✅ Arquivo carregado")
            if st.button("Remover", key="remove_tv_linear"):
                st.session_state.df_tv_linear = None
                st.session_state.tv_linear_status = False
                st.session_state.df_merged = None
                st.session_state.needs_merge = False
                st.rerun()
        else:
            arquivo_tv_linear = st.file_uploader("Upload CSV", type=['csv'], key='uploader_tv_linear')
            if arquivo_tv_linear is not None:
                try:
                    df_tv_linear = pd.read_csv(arquivo_tv_linear)
                    st.session_state.df_tv_linear = tratar_tv_linear(df_tv_linear)
                    st.session_state.tv_linear_status = True
                    st.session_state.df_merged = None  # Reset merged data
                    st.session_state.needs_merge = True
                    st.success("Arquivo processado com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {str(e)}")
    
    # Third column - TV Linear Semanal (New)
    with col3:
        st.markdown("### TV Linear (Dados Semanais)")
        
        # Check if data is already loaded
        if st.session_state.tv_linear_semanal_status:
            st.success("✅ Arquivo carregado")
            if st.button("Remover", key="remove_tv_linear_semanal"):
                st.session_state.df_tv_linear_semanal = None
                st.session_state.tv_linear_semanal_status = False
                st.session_state.df_merged = None
                st.session_state.needs_merge = False
                st.rerun()
        else:
            arquivo_tv_linear_semanal = st.file_uploader("Upload CSV", type=['csv'], key='uploader_tv_linear_semanal')
            if arquivo_tv_linear_semanal is not None:
                try:
                    df_tv_linear_semanal = pd.read_csv(arquivo_tv_linear_semanal)
                    st.session_state.df_tv_linear_semanal = tratar_tv_linear_semanal(df_tv_linear_semanal)
                    st.session_state.tv_linear_semanal_status = True
                    st.session_state.df_merged = None  # Reset merged data
                    st.session_state.needs_merge = True
                    st.success("Arquivo processado com sucesso!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {str(e)}")
    
    # Auto-merge if all required files are loaded and needs merge is set
    # The required files depend on the selected granularity
    should_merge = False
    
    if st.session_state.granularidade == "data_hora":
        # For hourly analysis, we need the regular TV Linear data
        should_merge = (st.session_state.rs_status and 
                        st.session_state.rs_canais_status and 
                        st.session_state.globoplay_status and 
                        st.session_state.tv_linear_status and
                        (st.session_state.needs_merge or st.session_state.df_merged is None))
    else:  # semana
        # For weekly analysis, we need the weekly TV Linear data
        should_merge = (st.session_state.rs_status and 
                        st.session_state.rs_canais_status and 
                        st.session_state.globoplay_status and 
                        st.session_state.tv_linear_semanal_status and
                        (st.session_state.needs_merge or st.session_state.df_merged is None))
    
    if should_merge:
        st.subheader("Processamento dos dados")
        
        with st.spinner(f"Processando dados automaticamente para granularidade '{st.session_state.granularidade}'..."):
            # Merge datasets based on the selected granularity
            if st.session_state.granularidade == "data_hora":
                df_merged = merge_data(
                    st.session_state.df_redes_sociais,
                    st.session_state.df_redes_sociais_canais,
                    st.session_state.df_globoplay,
                    st.session_state.df_tv_linear,
                    granularidade="data_hora"
                )
            else:  # semana
                df_merged = merge_data(
                    st.session_state.df_redes_sociais,
                    st.session_state.df_redes_sociais_canais,
                    st.session_state.df_globoplay,
                    st.session_state.df_tv_linear_semanal,
                    granularidade="semana"
                )
            
            if df_merged is not None:
                # Apply external data processing
                if st.session_state.granularidade == "data_hora":
                    max_date = df_merged['data_hora'].max().strftime('%d/%m/%Y')
                    min_date = df_merged['data_hora'].min().strftime('%d/%m/%Y')
                    
                    with st.status("Processando indicadores econômicos..."):
                        df_merged = fetch_all_bcb_economic_indicators(df_merged, 'data_hora', min_date, max_date)
                    
                    with st.status("Processando dados de grade..."):
                        eventos = {
                            'FUTEBOL' : ['FUTEBOL NOT', 'FUTEBOL MAT', 'FUTEBOL VES', 'FUTEBOL MAD'],
                            'BBB' : ['BIG BROTHER BRASIL'],
                            'AFAZENDA' : ['A FAZENDA'],
                            'OLIMPIADAS' : ['JOGOS OLIMPICOS MAT', 'JOGOS OLIMPICOS VES', 'JOGOS OLIMPICOS MAD']
                        }
                        df_merged = join_grade_external_data(df_merged, eventos=eventos)
                    
                    with st.status("Processando dados de tweets..."):
                        df_merged = join_tweets(df_merged)
                    
                    with st.status("Processando dados de eventos externos..."):
                        df_merged = join_eventos_externos(df_merged)
                else:  # semana
                    # For weekly data, we may need to adapt external data processing
                    # Here we're assuming the same functions can handle weekly data
                    # with appropriate modifications in the functions themselves
                    with st.status("Processando indicadores econômicos para dados semanais..."):
                        # Use 'semana' instead of 'data_hora' for weekly data
                        df_merged = fetch_all_bcb_economic_indicators(df_merged, 'data_hora', None, None)
                    
                    # Other external data may require special handling for weekly data
                    # We may need to implement specific versions for weekly data
                
                df_merged = df_merged.fillna(0).infer_objects(copy=False)
                st.session_state.df_merged = df_merged
                st.session_state.needs_merge = False
                
                st.success(f"✅ Todos os arquivos foram processados e mesclados com sucesso para granularidade '{st.session_state.granularidade}'!")
                
                st.subheader("Pré-visualização dos Dados Mesclados")
                st.dataframe(df_merged, hide_index=True)
                
                # Show information about the dataset
                st.subheader("Informações do Dataset")
                if st.session_state.granularidade == "data_hora":
                    max_date = df_merged['data_hora'].max().strftime('%d/%m/%Y')
                    min_date = df_merged['data_hora'].min().strftime('%d/%m/%Y')
                    st.write(f"**Período:** {min_date} a {max_date}")
                else:  # semana
                    if 'ano' in df_merged.columns and 'semana' in df_merged.columns:
                        min_year = df_merged['ano'].min()
                        max_year = df_merged['ano'].max()
                        min_week = df_merged.loc[df_merged['ano'] == min_year, 'semana'].min()
                        max_week = df_merged.loc[df_merged['ano'] == max_year, 'semana'].max()
                        st.write(f"**Período:** Semana {min_week}/{min_year} a Semana {max_week}/{max_year}")
                
                st.write(f"**Número de linhas:** {df_merged.shape[0]}")
                st.write(f"**Número de colunas:** {df_merged.shape[1]}")
                
                st.success("✅ Dados prontos para análise! Você pode proceder para as outras páginas do dashboard.")
            else:
                st.error("⚠️ Não foi possível mesclar os datasets. Verifique se os arquivos carregados contêm dados compatíveis.")
    
    # If already merged, show the preview
    elif st.session_state.df_merged is not None:
        df_merged = st.session_state.df_merged
        
        st.subheader("Pré-visualização dos Dados Mesclados")
        st.dataframe(df_merged, hide_index=True)
        
        # Show information about the dataset
        st.subheader("Informações do Dataset")
        if st.session_state.granularidade == "data_hora":
            max_date = df_merged['data_hora'].max().strftime('%d/%m/%Y')
            min_date = df_merged['data_hora'].min().strftime('%d/%m/%Y')
            st.write(f"**Período:** {min_date} a {max_date}")
        else:  # semana
            if 'ano' in df_merged.columns and 'semana' in df_merged.columns:
                min_year = df_merged['ano'].min()
                max_year = df_merged['ano'].max()
                min_week = df_merged.loc[df_merged['ano'] == min_year, 'semana'].min()
                max_week = df_merged.loc[df_merged['ano'] == max_year, 'semana'].max()
                st.write(f"**Período:** Semana {min_week}/{min_year} a Semana {max_week}/{max_year}")
        
        st.write(f"**Número de linhas:** {df_merged.shape[0]}")
        st.write(f"**Número de colunas:** {df_merged.shape[1]}")
        
        st.success("✅ Dados prontos para análise! Você pode proceder para as outras páginas do dashboard.")