import pandas as pd
from datetime import datetime, time
import requests

def join_futebol_external_data(tabela_mae):
    dados_externos_futebol = pd.read_csv('futebol_externo.csv', sep = ";")

    # Garantir que data_hora em tabela_mae seja datetime
    if not pd.api.types.is_datetime64_dtype(tabela_mae['data_hora']):
        tabela_mae['data_hora'] = pd.to_datetime(tabela_mae['data_hora'])
    
    # Criar a coluna no tabela_mae inicializada com 0
    tabela_mae['EXTERNO_FUTEBOL_CONCORRENTE_ON'] = 0
    
    # Processar dados_externos_futebol
    # Converter data_hora para datetime
    dados_externos_futebol['data_datetime'] = pd.to_datetime(dados_externos_futebol['data_hora'], format='%d/%m/%Y %H:%M')
    
    # Extrair apenas a data
    dados_externos_futebol['data'] = dados_externos_futebol['data_datetime'].dt.date
    
    # Extrair apenas a parte da hora (arredondando para hora cheia)
    def extrair_hora(hora_str):
        partes = hora_str.split(':')
        return int(partes[0])  # Pegar apenas a parte da hora
    
    # Criar data_inicio e data_fim com horas arredondadas
    dados_externos_futebol['hora_inicio_h'] = dados_externos_futebol['hora_inicio'].apply(extrair_hora)
    dados_externos_futebol['hora_fim_h'] = dados_externos_futebol['hora_fim'].apply(extrair_hora)
    
    # Combinar data com hora arredondada
    dados_externos_futebol['data_inicio'] = dados_externos_futebol.apply(lambda x: pd.Timestamp(x['data']).replace(hour=x['hora_inicio_h'], minute=0, second=0), axis=1)
    dados_externos_futebol['data_fim'] = dados_externos_futebol.apply(lambda x: pd.Timestamp(x['data']).replace(hour=x['hora_fim_h'], minute=59, second=59), axis=1)
    
    # Para cada evento em dados_externos_futebol, marcar os registros em tabela_mae
    for _, evento in dados_externos_futebol.iterrows():
        inicio = evento['data_inicio']
        fim = evento['data_fim']
        
        # Marcar registros dentro do intervalo
        mascara = (tabela_mae['data_hora'] >= inicio) & (tabela_mae['data_hora'] <= fim)
        tabela_mae.loc[mascara, 'EXTERNO_FUTEBOL_CONCORRENTE_ON'] = 1
    
    return tabela_mae

def join_eventos_external_data(tabela_mae):

    df_eventos_externos = pd.read_csv('eventos_externos.csv', sep = ";")

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
    for _, evento in df_eventos_externos.iterrows():
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
    
    # Buscar e processar cada série
    for code, name in series_mapping.items():
        # Adicionando prefixo EXTERNO_ ao nome da série
        prefixed_name = f"EXTERNO_{name}"
        temp_df = fetch_bcb_series(code, name)
        
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
            # FIX: Substituir ffill com inplace=True
            dados_externos[prefixed_name] = dados_externos[prefixed_name].ffill()
            dados_externos = dados_externos.drop(columns=['year_month'])
        else:  # Séries diárias
            # Renomear a coluna para incluir o prefixo
            temp_df.rename(columns={name: prefixed_name}, inplace=True)
            dados_externos = dados_externos.merge(temp_df[['data_hora', prefixed_name]], on='data_hora', how='left')
            # FIX: Substituir ffill com inplace=True
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