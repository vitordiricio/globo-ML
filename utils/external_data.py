import pandas as pd
from datetime import datetime, time
import requests
import streamlit as st
import numpy as np
from utils.data_processing import fill_hourly_gaps
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor


@st.cache_data
def join_tweets(tabela_mae):
    """
    Função específica para tratar os dados de tweets coletados
    """
    tweets = pd.read_csv('tweets_23_24.csv')

    tweets['data_hora'] = pd.to_datetime(tweets['Data'] + ' ' + tweets['Hora'], format='%d/%m/%y %H:%M')
    tweets['data_hora'] = tweets['data_hora'].dt.round('h')
    tweets = tweets[(tweets['data_hora'] >= '2023-01-01') & (tweets['data_hora'] < '2025-01-01')]
    tweets = tweets.groupby('data_hora').size().reset_index(name='EXTERNO_quantidade_tweets')

    tweets = fill_hourly_gaps(tweets, 'data_hora')

    # Merge with TV linear data
    df_merged = pd.merge(
        tabela_mae,
        tweets,
        on='data_hora',
        how='left'  # Using inner join to keep only common timestamps
    )

    return df_merged


@st.cache_data
def join_grade_external_data(tabela_mae, eventos=None):
    """
    Adds columns to tabela_mae based on TV programming data.
    
    Args:
        tabela_mae (DataFrame): Main dataframe to add columns to
        eventos (dict, optional): Dictionary mapping event types to program names
                                 e.g. {'FUTEBOL': ['FUTEBOL NOT', 'FUTEBOL MAT'], 'BBB': ['BIG BROTHER BRASIL']}
    
    Returns:
        DataFrame: tabela_mae with added columns
    """
    # Make a copy of the input DataFrame to avoid warnings
    tabela_mae = tabela_mae.copy()
    
    # Ensure data_hora is datetime type
    if not pd.api.types.is_datetime64_dtype(tabela_mae['data_hora']):
        tabela_mae['data_hora'] = pd.to_datetime(tabela_mae['data_hora'])
    
    # Helper function to process a TV data file
    def process_tv_file(file_path):
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '').str.replace('[', '').str.replace(']', '')
        df = df[['nome_programa', 'detalhe', 'quadro', 'gênero', 'emissora', 'data', 'hora_início', 'hora_fim', 'duração_prg']]
        return df.drop_duplicates()
    
    # Process all TV data files in parallel
    tv_files = ['globo_tv_linear.csv', 'band_tv_linear.csv', 'sbt_tv_linear.csv', 'record_tv_linear.csv']
    
    with ThreadPoolExecutor() as executor:
        dfs = list(executor.map(process_tv_file, tv_files))
    
    grade_todas_emissoras = pd.concat(dfs, ignore_index=True)
    
    # Normalize hour format to 24h cycle (vetorizado)
    def normalize_hours(hour_series):
        # Separar horas e minutos em colunas
        hours_minutes = hour_series.str.split(':', expand=True)
        hours = hours_minutes[0].astype(int)
        minutes = hours_minutes[1].astype(int)
        
        # Calcular dias adicionais e horas normalizadas
        days_ahead = hours // 24
        normalized_hours = hours % 24
        
        # Formatar de volta para strings
        normalized_time_strs = normalized_hours.astype(str).str.zfill(2) + ':' + minutes.astype(str).str.zfill(2)
        
        return normalized_time_strs, days_ahead
    
    # Processar horários de início e fim
    grade_todas_emissoras['hora_normalizada_inicio'], grade_todas_emissoras['dias_adicionais_inicio'] = normalize_hours(grade_todas_emissoras['hora_início'])
    grade_todas_emissoras['hora_normalizada_fim'], grade_todas_emissoras['dias_adicionais_fim'] = normalize_hours(grade_todas_emissoras['hora_fim'])
    
    # Criar objetos datetime com ajustes de data
    grade_todas_emissoras['data_hora_inicio'] = pd.to_datetime(
        grade_todas_emissoras['data'] + ' ' + grade_todas_emissoras['hora_normalizada_inicio'],
        dayfirst=True
    ) + pd.to_timedelta(grade_todas_emissoras['dias_adicionais_inicio'], unit='days')
    
    grade_todas_emissoras['data_hora_fim'] = pd.to_datetime(
        grade_todas_emissoras['data'] + ' ' + grade_todas_emissoras['hora_normalizada_fim'],
        dayfirst=True
    ) + pd.to_timedelta(grade_todas_emissoras['dias_adicionais_fim'], unit='days')
    
    # Tratar casos especiais: hora fim anterior à hora início
    mask = (grade_todas_emissoras['data_hora_fim'] < grade_todas_emissoras['data_hora_inicio']) & (grade_todas_emissoras['dias_adicionais_fim'] == grade_todas_emissoras['dias_adicionais_inicio'])
    if mask.any():
        grade_todas_emissoras.loc[mask, 'data_hora_fim'] = grade_todas_emissoras.loc[mask, 'data_hora_fim'] + pd.Timedelta(days=1)
    
    # Calcular limites de horas (floor/ceiling)
    grade_todas_emissoras['data_hora_inicio_floor'] = grade_todas_emissoras['data_hora_inicio'].dt.floor('h')
    grade_todas_emissoras['data_hora_fim_ceil'] = grade_todas_emissoras['data_hora_fim'].dt.ceil('h')
    
    # Dicionário para armazenar todas as novas colunas
    new_columns = {}
    
    # Função para obter todos os timestamps horários em um intervalo
    def get_hourly_timestamps(start_time, end_time):
        return pd.date_range(start=start_time, end=end_time - pd.Timedelta(seconds=1), freq='h')
    
    # Processar tipos de eventos
    if eventos:
        for evento_key, programas in eventos.items():
            # Filtrar programas relevantes
            evento_programs = grade_todas_emissoras[grade_todas_emissoras['nome_programa'].isin(programas)]
            
            # Processar por emissora
            for emissora, emissora_programs in evento_programs.groupby('emissora'):
                col_name = f"EXTERNO_GRADE_{evento_key}_{emissora}_ON"
                
                # Obter todos os timestamps para esta combinação evento-emissora
                event_timestamps = set()
                for _, programa in emissora_programs.iterrows():
                    hourly_timestamps = get_hourly_timestamps(
                        programa['data_hora_inicio_floor'],
                        programa['data_hora_fim_ceil']
                    )
                    event_timestamps.update(hourly_timestamps)
                
                # Criar máscara booleana para tabela_mae
                new_columns[col_name] = tabela_mae['data_hora'].isin(event_timestamps).astype(int)
    
    # Processar combinações gênero-emissora
    for emissora in grade_todas_emissoras['emissora'].unique():
        emissora_data = grade_todas_emissoras[grade_todas_emissoras['emissora'] == emissora]
        
        for genero in emissora_data['gênero'].dropna().unique():
            if genero == '':
                continue
                
            # Criar nome de gênero sanitizado
            genero_sanitized = str(genero).replace(' ', '_').replace('-', '_').upper()
            col_name = f"EXTERNO_GRADE_GENERO_{emissora}_{genero_sanitized}_ON"
            
            # Obter programas para esta combinação gênero-emissora
            filtered_programs = grade_todas_emissoras[
                (grade_todas_emissoras['emissora'] == emissora) & 
                (grade_todas_emissoras['gênero'] == genero)
            ]
            
            # Obter todos os timestamps para esta combinação gênero-emissora
            genre_timestamps = set()
            for _, programa in filtered_programs.iterrows():
                hourly_timestamps = get_hourly_timestamps(
                    programa['data_hora_inicio_floor'],
                    programa['data_hora_fim_ceil']
                )
                genre_timestamps.update(hourly_timestamps)
            
            # Criar máscara booleana para tabela_mae
            new_columns[col_name] = tabela_mae['data_hora'].isin(genre_timestamps).astype(int)
    
    # Criar um DataFrame com todas as novas colunas de uma vez
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=tabela_mae.index)
        
        # Concatenar com o DataFrame original
        result = pd.concat([tabela_mae, new_df], axis=1)
        return result
    else:
        # Se nenhuma nova coluna foi criada, retornar o DataFrame original
        return tabela_mae


def join_eventos_externos(tabela_mae):
    df_eventos = pd.read_csv('eventos_externos.csv', sep = ";")

    # Garantir que data_hora em tabela_mae seja datetime
    if not pd.api.types.is_datetime64_dtype(tabela_mae['data_hora']):
        tabela_mae['data_hora'] = pd.to_datetime(tabela_mae['data_hora'])
    
    # Funções auxiliares para conversão
    def converter_data(data_str):
        if pd.isna(data_str) or data_str == '-':
            return None
        try:
            return pd.to_datetime(data_str, format='%d/%m/%y')
        except:
            try:
                return pd.to_datetime(data_str, format='%d/%m/%Y')
            except:
                return None
    
    def converter_hora(hora_str, tipo):
        if pd.isna(hora_str) or hora_str == '-':
            return time(0, 0) if tipo == 'inicio' else time(23, 59, 59)
        if hora_str == '00:00' and tipo == 'fim':
            return time(23, 59, 59)  # Tratar 00:00 como final do dia para hora_fim
        try:
            return datetime.strptime(hora_str, '%H:%M').time()
        except:
            return time(0, 0) if tipo == 'inicio' else time(23, 59, 59)
    
    # Para cada evento no dataframe de eventos
    for _, evento in df_eventos.iterrows():
        nome_evento = evento['evento'].strip()
        nome_coluna = f"EXTERNO_{nome_evento}"
        tabela_mae[nome_coluna] = 0  # Inicializar com 0
        
        # Processar dados do evento
        data_inicio = converter_data(evento['data_inicio'])
        if data_inicio is None:
            continue  # Pular evento sem data de início
        
        data_fim = converter_data(evento['data_fim'])
        if data_fim is None:
            data_fim = pd.Timestamp.max  # Sem fim = infinito
        
        hora_inicio = converter_hora(evento['hora_inicio'], 'inicio')
        hora_fim = converter_hora(evento['hora_fim'], 'fim')
        atravessa_meia_noite = hora_fim < hora_inicio
        
        # Definir a função para verificar se um registro está dentro do evento
        def esta_no_evento(timestamp):
            data = timestamp.date()
            hora = timestamp.time()
            
            # Verificar se está dentro do intervalo de datas
            if data < data_inicio.date() or (data_fim != pd.Timestamp.max and data > data_fim.date()):
                return False
            
            # Caso 1: Evento normal (não atravessa a meia-noite)
            if not atravessa_meia_noite:
                return hora_inicio <= hora <= hora_fim
            
            # Caso 2: Evento que atravessa a meia-noite
            
            # Caso 2.1: Primeiro dia do evento
            if data == data_inicio.date():
                return hora >= hora_inicio
            
            # Caso 2.2: Último dia do evento (se tiver data_fim)
            if data_fim != pd.Timestamp.max and data == data_fim.date():
                return hora <= hora_fim
            
            # Caso 2.3: Dias intermediários ou continuação em evento sem fim
            if data > data_inicio.date() and (data_fim == pd.Timestamp.max or data < data_fim.date()):
                # Em dias intermediários, considerar tanto a continuação da noite anterior
                # quanto o início de um novo ciclo do evento
                return hora <= hora_fim or hora >= hora_inicio
            
            return False
        
        # Aplicar a função a cada registro
        tabela_mae[nome_coluna] = tabela_mae['data_hora'].apply(esta_no_evento).astype(int)
    
    return tabela_mae


@st.cache_data(ttl=3600)  # Cache por 1 hora
def fetch_all_bcb_economic_indicators(df=None, date_column='data_hora', start_date=None, end_date=None):
    """
    Busca todos os indicadores econômicos do Banco Central e retorna um dataframe final
    com prefixo 'EXTERNO_' em todas as colunas exceto a coluna de data.
    Opcionalmente faz merge com um dataframe existente.
    
    Parâmetros:
    df (pandas.DataFrame, opcional): DataFrame para fazer merge com os indicadores
    date_column (str): Nome da coluna de data no DataFrame df
    start_date (str ou datetime, opcional): Data inicial no formato 'dd/mm/yyyy' ou objeto datetime
    end_date (str ou datetime, opcional): Data final no formato 'dd/mm/yyyy' ou objeto datetime
    
    Retorna:
    pandas.DataFrame: DataFrame com indicadores econômicos, opcionalmente mesclado com df
    """
    
    # Se um dataframe foi fornecido e não foram especificadas datas, usa o min/max do dataframe
    if df is not None and (start_date is None or end_date is None):
        df[date_column] = pd.to_datetime(df[date_column])
        
        if start_date is None:
            start_date = df[date_column].min()
        
        if end_date is None:
            end_date = df[date_column].max()
    
    # Se não for fornecida uma data inicial, use 01/01/2022 como padrão
    if start_date is None:
        start_date = "01/01/2022"
    elif isinstance(start_date, datetime):
        start_date = start_date.strftime('%d/%m/%Y')
    
    # Se não for fornecida uma data final, use a data atual como padrão
    if end_date is None:
        end_date = datetime.today().strftime('%d/%m/%Y')
    elif isinstance(end_date, datetime):
        end_date = end_date.strftime('%d/%m/%Y')
    
    # Função auxiliar para buscar uma série específica
    def fetch_bcb_series(series_code, series_name):
        try:
            url = f'https://api.bcb.gov.br/dados/serie/bcdata.sgs.{series_code}/dados?formato=json&dataInicial={start_date}&dataFinal={end_date}'
            response = requests.get(url)
            response.raise_for_status()  # Levanta exceção para códigos de status ruins

            try:
                data = response.json()
            except requests.exceptions.JSONDecodeError:
                print(f"Erro ao decodificar JSON para série {series_name} (código: {series_code})")
                return pd.DataFrame(columns=['data_hora', series_name])

            series_df = pd.DataFrame(data)
            if series_df.empty:
                return pd.DataFrame(columns=['data_hora', series_name])
                
            series_df['data_hora'] = pd.to_datetime(series_df['data'], dayfirst=True)
            series_df[series_name] = series_df['valor'].astype(float)
            return series_df[['data_hora', series_name]]

        except Exception as e:
            print(f"Erro ao buscar série {series_name} (código: {series_code}): {str(e)}")
            return pd.DataFrame(columns=['data_hora', series_name])

    # Mapeamento das séries
    series_mapping = {
        1: 'dolar',
        24369: 'unemployment_rate',
        433: 'inflation_ipca',
        11: 'selic_rate',
        4394: 'indice_cond_economicas'
    }
    
    # Se temos um dataframe, usamos suas datas, senão criamos um range de datas
    if df is not None:
        # Extrai apenas a parte da data (sem hora) para usar no merge
        df = df.copy()  # Create a copy to avoid SettingWithCopyWarning
        df['data_apenas'] = pd.to_datetime(df[date_column]).dt.date
        # Convertemos para datetime para garantir o formato correto
        df['data_apenas'] = pd.to_datetime(df['data_apenas'])
        # Obtemos datas únicas
        unique_dates = pd.DataFrame({'data_hora': pd.to_datetime(df['data_apenas'].unique())})
        dados_externos = unique_dates
    else:
        # Crie um dataframe de datas para garantir continuidade
        date_range = pd.date_range(
            start=pd.to_datetime(start_date, dayfirst=True),
            end=pd.to_datetime(end_date, dayfirst=True),
            freq='D'
        )
        dados_externos = pd.DataFrame({'data_hora': date_range})
    
    # Busca paralela de dados para todas as séries
    series_data = {}
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submete jobs
        future_to_series = {
            executor.submit(fetch_bcb_series, code, name): (code, name) 
            for code, name in series_mapping.items()
        }
        
        # Processa resultados conforme chegam
        for future in concurrent.futures.as_completed(future_to_series):
            code, name = future_to_series[future]
            try:
                temp_df = future.result()
                series_data[(code, name)] = temp_df
            except Exception as e:
                print(f"Erro ao buscar série {name}: {e}")
                series_data[(code, name)] = pd.DataFrame(columns=['data_hora', name])
    
    # Processa cada série
    for (code, name), temp_df in series_data.items():
        # Adiciona prefixo ao nome da coluna
        prefixed_name = f"EXTERNO_{name}"
        
        if temp_df.empty:
            dados_externos[prefixed_name] = None
            continue
        
        if code in [24369, 433]:  # Séries mensais
            # Criar referência ano-mês
            temp_df['year_month'] = temp_df['data_hora'].dt.to_period('M')
            # Obter último valor por mês
            temp_df = temp_df.groupby('year_month').last().reset_index()
            # Renomear a coluna para incluir o prefixo
            temp_df.rename(columns={name: prefixed_name}, inplace=True)
            
            # Mesclar com o dataframe principal
            dados_externos['year_month'] = dados_externos['data_hora'].dt.to_period('M')
            dados_externos = dados_externos.merge(temp_df[['year_month', prefixed_name]], on='year_month', how='left')
            # Forward fill para valores ausentes
            dados_externos[prefixed_name] = dados_externos[prefixed_name].ffill()
            dados_externos = dados_externos.drop(columns=['year_month'])
        else:  # Séries diárias
            # Renomear a coluna para incluir o prefixo
            temp_df.rename(columns={name: prefixed_name}, inplace=True)
            dados_externos = dados_externos.merge(temp_df[['data_hora', prefixed_name]], on='data_hora', how='left')
            # Forward fill para valores ausentes
            dados_externos[prefixed_name] = dados_externos[prefixed_name].ffill()
    
    # Se temos um dataframe, fazemos o merge
    if df is not None:
        # Extraímos apenas a parte da data da coluna data_hora para fazer o merge
        dados_externos['data_apenas'] = dados_externos['data_hora'].dt.date
        dados_externos['data_apenas'] = pd.to_datetime(dados_externos['data_apenas'])
        
        # Fazemos o merge baseado na data (sem a hora)
        resultado = df.merge(dados_externos.drop('data_hora', axis=1), 
                           left_on='data_apenas', right_on='data_apenas', how='inner')
        
        # Removemos colunas temporárias
        resultado = resultado.drop('data_apenas', axis=1)
        
        return resultado
    else:
        return dados_externos


@st.cache_data
def join_eventos_external_data(tabela_mae):
    """
    Versão otimizada da função join_eventos_external_data que processa
    dados de eventos externos de forma mais eficiente.
    """
    df_eventos_externos = pd.read_csv('eventos_externos.csv', sep=";")

    # Garantir que data_hora em tabela_mae seja datetime
    if not pd.api.types.is_datetime64_dtype(tabela_mae['data_hora']):
        tabela_mae['data_hora'] = pd.to_datetime(tabela_mae['data_hora'])
    
    # Criar uma cópia para evitar SettingWithCopyWarning
    tabela_mae = tabela_mae.copy()
    
    # Pré-processar dados de eventos
    df_eventos_externos = df_eventos_externos.copy()
    
    # Criar todas as colunas de eventos de uma vez
    event_columns = {}
    for evento in df_eventos_externos['evento'].unique():
        col_name = f"EXTERNO_{evento.strip()}"
        event_columns[col_name] = np.zeros(len(tabela_mae))
    
    # Criar um DataFrame com as novas colunas
    evento_df = pd.DataFrame(event_columns, index=tabela_mae.index)
    
    # Adicionar colunas de eventos ao tabela_mae
    tabela_mae = pd.concat([tabela_mae, evento_df], axis=1)
    
    # Converter colunas de data em df_eventos_externos
    def convert_date(x):
        if pd.isna(x) or x == '-':
            return None
        try:
            return pd.to_datetime(x, format='%d/%m/%y')
        except:
            try:
                return pd.to_datetime(x, format='%d/%m/%Y')
            except:
                return None
    
    # Conversão vetorizada para colunas de data
    df_eventos_externos['data_inicio'] = df_eventos_externos['data_inicio'].apply(convert_date)
    df_eventos_externos['data_fim'] = df_eventos_externos['data_fim'].apply(convert_date)
    
    # Definir data final ausente para data máxima
    df_eventos_externos.loc[df_eventos_externos['data_fim'].isna(), 'data_fim'] = pd.Timestamp.max
    
    # Processar horários
    def get_hour_from_time(time_str, is_end=False):
        if pd.isna(time_str) or time_str == '-':
            return 0 if not is_end else 23
        if time_str == '00:00' and is_end:
            return 23
        try:
            return int(time_str.split(':')[0])
        except:
            return 0 if not is_end else 23
    
    df_eventos_externos['hora_inicio_num'] = df_eventos_externos['hora_inicio'].apply(
        lambda x: get_hour_from_time(x, False))
    df_eventos_externos['hora_fim_num'] = df_eventos_externos['hora_fim'].apply(
        lambda x: get_hour_from_time(x, True))
    
    # Processar cada evento
    for _, evento in df_eventos_externos.iterrows():
        col_name = f"EXTERNO_{evento['evento'].strip()}"
        
        # Pular eventos sem data de início
        if pd.isna(evento['data_inicio']):
            continue
        
        start_date = evento['data_inicio']
        end_date = evento['data_fim']
        start_hour = evento['hora_inicio_num']
        end_hour = evento['hora_fim_num']
        
        # Verificar se o evento atravessa a meia-noite
        spans_midnight = end_hour < start_hour
        
        # Para cada timestamp em tabela_mae, verificar se está dentro deste evento
        # Abordagem vetorizada usando máscaras
        date_mask = (tabela_mae['data_hora'].dt.date >= start_date.date()) & \
                   ((end_date == pd.Timestamp.max) | (tabela_mae['data_hora'].dt.date <= end_date.date()))
        
        if not spans_midnight:
            # Evento no mesmo dia, não atravessa a meia-noite
            hour_mask = (tabela_mae['data_hora'].dt.hour >= start_hour) & \
                        (tabela_mae['data_hora'].dt.hour <= end_hour)
            final_mask = date_mask & hour_mask
        else:
            # Evento atravessa a meia-noite
            first_day_mask = (tabela_mae['data_hora'].dt.date == start_date.date()) & \
                            (tabela_mae['data_hora'].dt.hour >= start_hour)
            
            last_day_mask = ((end_date != pd.Timestamp.max) & \
                            (tabela_mae['data_hora'].dt.date == end_date.date()) & \
                            (tabela_mae['data_hora'].dt.hour <= end_hour))
            
            middle_days_mask = (tabela_mae['data_hora'].dt.date > start_date.date()) & \
                              ((end_date == pd.Timestamp.max) | (tabela_mae['data_hora'].dt.date < end_date.date()))
            
            final_mask = first_day_mask | last_day_mask | middle_days_mask
        
        tabela_mae.loc[final_mask, col_name] = 1
    
    return tabela_mae