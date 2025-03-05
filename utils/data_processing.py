# utils/data_processing.py
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
import numpy as np


@st.cache_data
def fill_hourly_gaps(df, datetime_column):
    """
    Fill gaps in hourly data with zeros for all metric columns using an optimized approach.
    
    Args:
        df: DataFrame containing time series data
        datetime_column: Name of the column containing datetime values
        
    Returns:
        DataFrame: A new dataframe with hourly gaps filled with zeros
    """
    import pandas as pd
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Ensure the datetime column is in datetime format
    df_copy[datetime_column] = pd.to_datetime(df_copy[datetime_column])
    
    # Set the datetime column as the index
    df_copy = df_copy.set_index(datetime_column)
    
    # Create a complete hourly range from min to max date
    complete_range = pd.date_range(
        start=df_copy.index.min(),
        end=df_copy.index.max(),
        freq='h'
    )
    
    # Reindex the dataframe with the complete range and fill missing values with 0
    filled_df = df_copy.reindex(complete_range, fill_value=0)
    
    # Reset index to turn the datetime back into a column
    filled_df = filled_df.reset_index()
    
    # Rename the index column back to the original datetime column name
    filled_df = filled_df.rename(columns={'index': datetime_column})
    
    return filled_df


@st.cache_data
def tratar_redes_sociais_canais(df):
    """
    Função específica para tratar os dados de redes sociais dos canais (G1, GE, GSHOW, ETC.)
    """

    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('[', '').str.replace(']', '')
    df['data_'] = pd.to_datetime(df['data_'], format='%d/%m/%Y %H:%M')
    df['data_'] = df['data_'].dt.round('h')
    df = df.copy(); df.rename(columns={'interacoes\r': 'interacoes'}, inplace=True); df.loc[:, 'interacoes'] = df['interacoes'].str.replace('\r', '').fillna('').replace('', '0').astype(int)
    df['plataforma'] = df['plataforma'].str.upper()
    df = df[['data_', 'perfil', 'plataforma', 'alcance', 'videoviews', 'impressoes', 'reacoes', 'comentarios', 'interacoes']]
    
    # Fix for the FutureWarning about downcasting
    df = df.infer_objects().fillna(0)
    

    df = df[['data_', 'perfil', 'plataforma', 'alcance', 'videoviews', 'impressoes', 'reacoes', 'comentarios', 'interacoes']]

    df = df.groupby(['data_', 'perfil', 'plataforma'], as_index=False).sum()

    redes_canais = df.melt(
        id_vars=["perfil", "data_", "plataforma"], 
        var_name="metrica", 
        value_name="valor"
    )
    redes_canais["nova_coluna"] = redes_canais["plataforma"] + "_" + redes_canais["perfil"] + "_" + redes_canais["metrica"]


    # Pivot e limpeza final
    result_df = redes_canais.pivot(
        index="data_", 
        columns="nova_coluna", 
        values="valor"
    ).reset_index()
    result_df.columns.name = None

    result_df = result_df.rename(columns={'data_': 'data_hora'})
    result_df = fill_hourly_gaps(result_df, 'data_hora')

    return result_df


@st.cache_data
def tratar_redes_sociais_linear(df):
    """
    Função específica para tratar os dados de redes sociais.
    """
    if df is None:
        return None
        
    
    # Transformação de data
    df['ts_published_brt'] = df['ts_published_brt'].str.replace('.000000 UTC', '')
    df['ts_published_brt'] = pd.to_datetime(df['ts_published_brt'])
    df['ts_published_brt'] = df['ts_published_brt'].dt.round('h')
    # Seleção de colunas
    colunas = ['ds_platform', 'total_interactions', 'nr_reactions', 'nr_shares', 
                'nr_comments', 'nr_saves', 'nr_views', 'nr_impressions',
                'nr_reach', 'ts_published_brt', 'id_post']
    df = df[colunas]
    
    # Agregação
    df_agregado = (
        df.groupby(['ds_platform', 'ts_published_brt'])
        .agg(
            total_interactions=('total_interactions', 'sum'),
            nr_reactions=('nr_reactions', 'sum'),
            nr_shares=('nr_shares', 'sum'),
            nr_comments=('nr_comments', 'sum'),
            nr_saves=('nr_saves', 'sum'),
            nr_views = ('nr_views', 'sum'),
            nr_reach=('nr_reach', 'sum'),
            nr_impressions = ('nr_impressions', 'sum'),
            posts_quantity=('id_post', 'nunique')
        )
        .sort_values(by=['ts_published_brt', 'ds_platform'])
        .reset_index()
    )
    
    # Transformação para formato wide
    df_melted = df_agregado.melt(
        id_vars=["ds_platform", "ts_published_brt"], 
        var_name="metrica", 
        value_name="valor"
    )
    df_melted["nova_coluna"] = df_melted["ds_platform"] + "_" + df_melted["metrica"]
    
    # Pivot e limpeza final
    df_final = df_melted.pivot(
        index="ts_published_brt", 
        columns="nova_coluna", 
        values="valor"
    ).reset_index()
    df_final.columns.name = None
    df_final = df_final.fillna(0)

 
    df_final = df_final.groupby('ts_published_brt', as_index=False).sum()
    df_final = fill_hourly_gaps(df_final, 'ts_published_brt')
    
    return df_final


@st.cache_data
def tratar_globoplay(df):
    """
    Função específica para tratar os dados do GloboPlay.
    """
    if df is None:
        return None
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    df = df.dropna()
    df['data'] = pd.to_datetime(df['data'])

    df = df[[col for col in df.columns if col not in ['mês', 'ano']]]

    # Seleciona as colunas numéricas automaticamente
    num_cols = df.select_dtypes(include='number').columns

    df_horas = pd.concat([
        pd.DataFrame({
            'data': pd.date_range(start=row['data'], periods=24, freq='h')
        }).assign(**{
            col: int(round(row[col] / 24, 0))
            for col in num_cols
        })
        for _, row in df.iterrows()
    ], ignore_index=True)

    df_horas = fill_hourly_gaps(df_horas, 'data')
    
    # Adicione aqui o tratamento específico para GloboPlay
    return df_horas


@st.cache_data
def tratar_tv_linear(df):
    """
    Process TV linear data by filtering, formatting, adjusting times and calculating metrics.
    
    Args:
        df: Input dataframe from 'tv_linear.csv'
        
    Returns:
        DataFrame: Processed dataframe with calculated metrics by hour
    """
    
    # Filter and format columns
    df = df.copy()
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('[', '').str.replace(']', '')
    df['hora_inicio'] = df['faixas_horárias'].str.split(' - ').str[0].apply(lambda x: f"{int(x.split(':')[0]) % 24:02d}:{x.split(':')[1]}")
    df['data_hora'] = pd.to_datetime(df['datas'] + ' ' + df['hora_inicio'], format='%d/%m/%Y %H:%M')
    df = df[['data_hora','emissoras', 'rat%', 'shr%', 'tvr%', 'rat#', 'avrch%_wavg', 'cov%', 'fid%_org', 'tvr#']]
    df['emissoras'] = df['emissoras'].str.replace(''', '').str.replace(''', '').str.replace(' - ', ' ').str.replace('é', 'e').str.replace(' ', '_')

    # Transformação para formato wide
    df_melted = df.melt(
        id_vars=["emissoras", "data_hora"], 
        var_name="metrica", 
        value_name="valor"
    )
    df_melted["nova_coluna"] = df_melted["emissoras"] + "_" + df_melted["metrica"]

    # Pivot e limpeza final
    result_df = df_melted.pivot(
        index="data_hora", 
        columns="nova_coluna", 
        values="valor"
    ).reset_index()
    result_df.columns.name = None
    result_df = result_df.fillna(0)
    
    return result_df


def carregar_e_tratar_dados():
    """
    Função principal que carrega e trata todos os dados.
    Returns:
        tuple: DataFrames tratados (redes_sociais, globoplay, tv_linear)
    """
    st.subheader("Upload dos dados")
    
    # Criar três colunas
    col1, col2, col3, col4 = st.columns(4)
    
    # Primeira coluna - Redes Sociais
    with col1:
        st.text("CSV de redes sociais GLOBO:")
        arquivo_redes_sociais = st.file_uploader("", type=['csv'], key='upload1')
        df_redes_sociais = None
        if arquivo_redes_sociais is not None:
            try:
                df_redes_sociais = pd.read_csv(arquivo_redes_sociais)
                df_redes_sociais = tratar_redes_sociais_linear(df_redes_sociais)
                st.success("Arquivo de redes sociais processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")

    with col2:
        st.text("CSV de redes sociais CANAIS:")
        arquivo_redes_sociais_canais = st.file_uploader("", type=['csv'], key='upload2')
        df_redes_sociais_canais = None
        if arquivo_redes_sociais_canais is not None:
            try:
                # Added low_memory=False to fix the DtypeWarning
                df_redes_sociais_canais = pd.read_csv('redes_sociais_canais.csv', 
                          encoding='latin-1',
                          sep=';',
                          quotechar='"',
                          doublequote=True,
                          lineterminator='\n',
                          escapechar='\\',
                          on_bad_lines='skip',
                          low_memory=False)
                df_redes_sociais_canais = tratar_redes_sociais_canais(df_redes_sociais_canais)
                st.success("Arquivo de redes sociais processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")

    
    # Segunda coluna - GloboPlay
    with col3:
        st.text("CSV de GloboPlay:")
        arquivo_globoplay = st.file_uploader("", type=['csv'], key='upload3')
        df_globoplay = None
        if arquivo_globoplay is not None:
            try:
                df_globoplay = pd.read_csv(arquivo_globoplay)
                df_globoplay = tratar_globoplay(df_globoplay)
                st.success("Arquivo do GloboPlay processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    # Terceira coluna - TV Linear
    with col4:
        st.text("CSV de TV Linear:")
        arquivo_tv_linear = st.file_uploader("", type=['csv'], key='upload4')
        df_tv_linear = None
        if arquivo_tv_linear is not None:
            try:
                df_tv_linear = pd.read_csv(arquivo_tv_linear)
                df_tv_linear = tratar_tv_linear(df_tv_linear)
                st.success("Arquivo de TV Linear processado!")
            except Exception as e:
                st.error(f"Erro ao processar arquivo: {str(e)}")
    
    return df_redes_sociais, df_redes_sociais_canais, df_globoplay, df_tv_linear


@st.cache_data
def merge_data(df_redes_sociais, df_redes_sociais_canais, df_globoplay, df_tv_linear):
    """
    Merge data from social media, Globoplay, and linear TV, keeping only data points
    that exist in all three dataframes, with appropriate prefixes for each source.
    
    Args:
        df_redes_sociais: Processed social media dataframe
        df_globoplay: Processed Globoplay dataframe
        df_tv_linear: Processed linear TV dataframe
        
    Returns:
        DataFrame: Merged dataframe containing only overlapping time periods with prefixed columns
    """
    # Check if all dataframes exist
    if df_redes_sociais is None or df_redes_sociais_canais is None or df_globoplay is None or df_tv_linear is None:
        return None
    
    # Round timestamps to the nearest hour for consistent merging
    df_redes_sociais['ts_published_brt'] = df_redes_sociais['ts_published_brt'].dt.round('h')
    df_globoplay['data'] = df_globoplay['data'].dt.round('h')
    # TV Linear already has data_hora at hourly intervals from your processing function
    
    # Rename timestamp columns for consistency
    df_redes = df_redes_sociais.rename(columns={'ts_published_brt': 'data_hora'})
    df_globo = df_globoplay.rename(columns={'data': 'data_hora'})
    # df_tv_linear already has 'data_hora' as column name
    
    # Add prefixes to column names except for 'data_hora'
    # 1. For tv_linear dataframe add LINEAR_ prefix
    prefix_columns = {col: f'LINEAR_{col}' for col in df_tv_linear.columns if col != 'data_hora'}
    df_tv_linear = df_tv_linear.rename(columns=prefix_columns)
    
    # 2. For redes_sociais add RS_ prefix
    prefix_columns = {col: f'RS_GLOBO_{col}' for col in df_redes.columns if col != 'data_hora'}
    df_redes = df_redes.rename(columns=prefix_columns)

    # 2. For redes_sociais add RS_ prefix
    prefix_columns = {col: f'RS_CANAIS_{col}' for col in df_redes_sociais_canais.columns if col != 'data_hora'}
    df_redes_sociais_canais = df_redes_sociais_canais.rename(columns=prefix_columns)
    
    # 3. For globoplay add GP_ prefix
    prefix_columns = {col: f'GP_{col}' for col in df_globo.columns if col != 'data_hora'}
    df_globo = df_globo.rename(columns=prefix_columns)
    
    # Otimização: Encontrar timestamps comuns antes de fazer o merge
    df_redes.set_index('data_hora', inplace=True)
    df_redes_sociais_canais.set_index('data_hora', inplace=True)
    df_tv_linear.set_index('data_hora', inplace=True)
    df_globo.set_index('data_hora', inplace=True)
    
    # Encontrar timestamps comuns
    common_dates = df_redes.index.intersection(df_redes_sociais_canais.index)
    common_dates = common_dates.intersection(df_tv_linear.index)
    common_dates = common_dates.intersection(df_globo.index)
    
    # Filtrar dataframes para timestamps comuns
    df_redes = df_redes.loc[common_dates]
    df_redes_sociais_canais = df_redes_sociais_canais.loc[common_dates]
    df_tv_linear = df_tv_linear.loc[common_dates]
    df_globo = df_globo.loc[common_dates]
    
    # Concatenar todos os dataframes
    df_merged = pd.concat([df_redes, df_redes_sociais_canais, df_tv_linear, df_globo], axis=1)
    df_merged = df_merged.reset_index()
    
    # Replace None with np.nan (if any)
    df_merged = df_merged.replace({None: np.nan})

    # Drop rows that are 100% 0's (or NaN's)
    df_merged = df_merged[~(df_merged.fillna(0) == 0).all(axis=1)]

    # Drop columns that are 100% 0's (or NaN's)
    df_merged = df_merged.loc[:, ~(df_merged.fillna(0) == 0).all(axis=0)]
    
    return df_merged

@st.cache_data
def convert_non_numeric_to_codes(df):
    """
    Converte colunas não numéricas em códigos numéricos para cálculos de correlação.
    """
    df_copy = df.copy()
    for col in df_copy.columns:
        if not pd.api.types.is_numeric_dtype(df_copy[col]):
            df_copy[col] = df_copy[col].astype('category').cat.codes
    return df_copy


@st.cache_data
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

    if 'data_hora' in df.columns:
        try:
            # Converter para datetime com UTC=True para lidar com timezone misto
            df = df.copy()  # Criar uma cópia para evitar SettingWithCopyWarning
            
            min_date = df['data_hora'].dt.date.min()
            max_date = df['data_hora'].dt.date.max()
            
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
                df_filtered = df[(df['data_hora'].dt.date >= start_date) & 
                               (df['data_hora'].dt.date <= end_date)]
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
                df_model = df_filtered.groupby(pd.Grouper(key='data_hora', freq=freq)).sum().reset_index()
                # Formatar a coluna data_hora de acordo com a granularidade selecionada
                if group_option == "Data":
                    df_model['data_hora'] = df_model['data_hora'].dt.strftime('%Y-%m-%d')
                elif group_option == "Semana":
                    df_model['data_hora'] = df_model['data_hora'].dt.strftime('%Y-W%U')
                elif group_option == "Mês":
                    df_model['data_hora'] = df_model['data_hora'].dt.strftime('%Y-%m')
                elif group_option == "Quarter":
                    df_model['data_hora'] = df_model['data_hora'].dt.to_period('Q').astype(str)
                elif group_option == "Ano":
                    df_model['data_hora'] = df_model['data_hora'].dt.year.astype(str)
            else:
                df_model = df_filtered.copy()
                
            st.markdown("Pré-visualização dos dados após tratamento:")
            st.dataframe(df_model, hide_index=True, height=250)
            return df_model
            
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {e}")
            return None
    else:
        st.error("A coluna 'data_hora' não foi encontrada no dataframe.")
        return None