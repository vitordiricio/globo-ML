# utils/data_processing.py
import pandas as pd
import streamlit as st

def tratar_redes_sociais(df):
    """
    Função específica para tratar os dados de redes sociais.
    """
    if df is None:
        return None
        
    # Tratamento inicial
    df = df.ffill()
    
    # Transformação de data
    df['dt_partition'] = pd.to_datetime(df['dt_partition'], format='%d/%m/%Y')
    df['dt_partition'] = pd.to_datetime(
        df['dt_partition'].dt.strftime('%d/%m/%Y') + ' ' +
        df['dt_hora'].astype(int).astype(str) + ':00:00',
        format='%d/%m/%Y %H:%M:%S'
    )
    
    # Seleção de colunas
    colunas = ['ds_platform', 'nr_total_interactions', 'nr_reactions', 'nr_shares', 
               'nr_comments', 'nr_saves', 'nr_link_clicks', 'nr_impressions_total', 
               'nr_reach', 'dt_partition', 'id_post']
    df = df[colunas]
    
    # Agregação
    df_agregado = (
        df.groupby(['ds_platform', 'dt_partition'])
        .agg(
            nr_total_interactions=('nr_total_interactions', 'sum'),
            nr_reactions=('nr_reactions', 'sum'),
            nr_shares=('nr_shares', 'sum'),
            nr_comments=('nr_comments', 'sum'),
            nr_saves=('nr_saves', 'sum'),
            nr_link_clicks=('nr_link_clicks', 'sum'),
            nr_impressions_total=('nr_impressions_total', 'sum'),
            nr_reach=('nr_reach', 'sum'),
            posts_quantity=('id_post', 'nunique')
        )
        .sort_values(by=['dt_partition', 'ds_platform'])
        .reset_index()
    )
    
    # Transformação para formato wide
    df_melted = df_agregado.melt(
        id_vars=["ds_platform", "dt_partition"], 
        var_name="metrica", 
        value_name="valor"
    )
    df_melted["nova_coluna"] = df_melted["ds_platform"] + "_" + df_melted["metrica"]
    
    # Pivot e limpeza final
    df_final = df_melted.pivot(
        index="dt_partition", 
        columns="nova_coluna", 
        values="valor"
    ).reset_index()
    df_final.columns.name = None
    df_final = df_final.fillna(0)
    
    # Adição de dias úteis
    df_final['dia_util'] = df_final['dt_partition'].dt.weekday
    
    return df_final

def tratar_globoplay(df):
    """
    Função específica para tratar os dados do GloboPlay.
    """
    if df is None:
        return None
    
    # Adicione aqui o tratamento específico para GloboPlay
    return df

def tratar_tv_linear(df):
    """
    Função específica para tratar os dados de TV Linear.
    """
    if df is None:
        return None
    
    # Adicione aqui o tratamento específico para TV Linear
    return df

def carregar_e_tratar_dados():
    """
    Função principal que carrega e trata todos os dados.
    Returns:
        tuple: DataFrames tratados (redes_sociais, globoplay, tv_linear)
    """
    st.subheader("Upload dos dados")
    
    # Criar três colunas
    col1, col2, col3 = st.columns(3)
    
    # Primeira coluna - Redes Sociais
    with col1:
        st.text("CSV de redes sociais:")
        arquivo_redes_sociais = st.file_uploader("", type=['csv'], key='upload1')
        df_redes_sociais = None
        if arquivo_redes_sociais is not None:
            try:
                df_redes_sociais = pd.read_csv(arquivo_redes_sociais)
                df_redes_sociais = tratar_redes_sociais(df_redes_sociais)
                st.success("Arquivo de redes sociais processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    # Segunda coluna - GloboPlay
    with col2:
        st.text("CSV de GloboPlay:")
        arquivo_globoplay = st.file_uploader("", type=['csv'], key='upload2')
        df_globoplay = None
        if arquivo_globoplay is not None:
            try:
                df_globoplay = pd.read_csv(arquivo_globoplay)
                df_globoplay = tratar_globoplay(df_globoplay)
                st.success("Arquivo do GloboPlay processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    # Terceira coluna - TV Linear
    with col3:
        st.text("CSV de TV Linear:")
        arquivo_tv_linear = st.file_uploader("", type=['csv'], key='upload3')
        df_tv_linear = None
        if arquivo_tv_linear is not None:
            try:
                df_tv_linear = pd.read_csv(arquivo_tv_linear)
                df_tv_linear = tratar_tv_linear(df_tv_linear)
                st.success("Arquivo de TV Linear processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    return df_redes_sociais, df_globoplay, df_tv_linear

def convert_non_numeric_to_codes(df):
    """
    Converte colunas não numéricas em códigos numéricos para cálculos de correlação.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype('category').cat.codes
    return df_copy

def filter_and_group_by_date(df, date_column, group_option, date_range=None):
    """
    Filtra o DataFrame com base em um intervalo de datas e agrupa os dados por uma granularidade específica.
    
    Args:
        df (DataFrame): DataFrame original.
        date_column (str): Nome da coluna com informações de data.
        group_option (str): Opção de agrupamento ("Data e hora", "Data", "Semana", "Mês", "Quarter", "Ano").
        date_range (list/tuple): Intervalo de datas [start_date, end_date] para filtrar o DataFrame.
    
    Returns:
        DataFrame: DataFrame filtrado e agrupado.
    """
    df = df.copy()
    try:
        df[date_column] = pd.to_datetime(df[date_column])
    except Exception as e:
        st.error(f"Erro ao converter '{date_column}' para datetime: {e}")
        return df

    if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        df = df[(df[date_column].dt.date >= start_date) & (df[date_column].dt.date <= end_date)]
    
    group_map = {
        "Data e hora": None,
        "Data": "D",
        "Semana": "W",
        "Mês": "ME",
        "Quarter": "QE",
        "Ano": "YE"
    }
    freq = group_map.get(group_option)
    if freq is not None:
        df = df.groupby(pd.Grouper(key=date_column, freq=freq)).mean().reset_index()
        if group_option == "Data":
            df[date_column] = df[date_column].dt.strftime('%Y-%m-%d')
        elif group_option == "Semana":
            df[date_column] = df[date_column].dt.strftime('%Y-W%U')
        elif group_option == "Mês":
            df[date_column] = df[date_column].dt.strftime('%Y-%m')
        elif group_option == "Quarter":
            df[date_column] = df[date_column].dt.to_period('Q').astype(str)
        elif group_option == "Ano":
            df[date_column] = df[date_column].dt.year.astype(str)
    return df

def calculate_correlation_series(df, target):
    """
    Calcula a correlação de cada coluna com a variável alvo.
    
    Args:
        df (DataFrame): DataFrame para análise.
        target (str): Nome da variável alvo.
    
    Returns:
        Series: Série com as correlações, ordenadas pelo valor absoluto em ordem decrescente.
    """
    df_corr = convert_non_numeric_to_codes(df)
    corr_matrix = df_corr.corr().fillna(0)
    if target not in corr_matrix.columns:
        return None
    corr_series = corr_matrix[target].drop(target).fillna(0)
    corr_series = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    return corr_series


def group_and_filter_by_date(df):

    """
    Pega o DataFrame original e filtra de acordo com o data range selecionado
    E também agrupa por diferentes tipos de agrupamentos de data (Data e hora, Data, Semana, Mês...)
    
    Args:
        df (DataFrame): DataFrame para análise.
    
    Returns:
        df_model (DataFrame): o DataFrame que mantém só os dados filtrados e agrupados de acordo
    """

    if 'dt_partition' in df.columns:
        try:
            df['dt_partition'] = pd.to_datetime(df['dt_partition'])
        except Exception as e:
            st.error(f"Erro ao converter 'dt_partition' para datetime: {e}")
            return
        
        min_date = df['dt_partition'].min().date()
        max_date = df['dt_partition'].max().date()
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            date_range = st.date_input(
                "Selecione o intervalo de datas:",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
        with col_date2:
            group_option = st.selectbox(
                "Agrupar dados por:",
                options=["Data e hora", "Data", "Semana", "Mês", "Quarter", "Ano"],
                help="Escolha a granularidade para agregação dos dados"
            )
        
        if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df[(df['dt_partition'].dt.date >= start_date) & (df['dt_partition'].dt.date <= end_date)]
        else:
            df_filtered = df.copy()
        
        group_map = {
            "Data e hora": None,
            "Data": "D",
            "Semana": "W",
            "Mês": "ME",
            "Quarter": "QE",
            "Ano": "YE"
        }
        freq = group_map[group_option]
        if freq is not None:
            # Agrupar dados utilizando a média para colunas numéricas
            df_model = df_filtered.groupby(pd.Grouper(key='dt_partition', freq=freq)).sum().reset_index()
            # Formatar a coluna dt_partition de acordo com a granularidade selecionada
            if group_option == "Data":
                df_model['dt_partition'] = df_model['dt_partition'].dt.strftime('%Y-%m-%d')
            elif group_option == "Semana":
                df_model['dt_partition'] = df_model['dt_partition'].dt.strftime('%Y-W%U')
            elif group_option == "Mês":
                df_model['dt_partition'] = df_model['dt_partition'].dt.strftime('%Y-%m')
            elif group_option == "Quarter":
                df_model['dt_partition'] = df_model['dt_partition'].dt.to_period('Q').astype(str)
            elif group_option == "Ano":
                df_model['dt_partition'] = df_model['dt_partition'].dt.year.astype(str)
        else:
            df_model = df_filtered.copy()
            
        st.markdown("Pré-visualização dos dados após tratamento:")
        st.dataframe(df_model, hide_index=True, height=250)
        return df_model
    else:
        st.error("A coluna 'dt_partition' não foi encontrada no dataframe.")
        return