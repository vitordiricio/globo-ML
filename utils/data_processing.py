# utils/data_processing.py
import pandas as pd
import streamlit as st

def carregar_dados():
    """
    Permite o upload de um arquivo CSV e retorna um DataFrame.
    """
    arquivo = st.file_uploader("", type=['csv'])
    if arquivo is not None:
        try:
            return pd.read_csv(arquivo)
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
    return None

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
    corr_matrix = df_corr.corr()
    if target not in corr_matrix.columns:
        return None
    corr_series = corr_matrix[target].drop(target).fillna(0)
    corr_series = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    df_corr_y = corr_series.to_frame(name="Correlação")
    df_corr_y.index.name = "Colunas"
    return df_corr_y
