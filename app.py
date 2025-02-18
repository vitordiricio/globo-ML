# main.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px
import plotly.graph_objects as go
from utils.model_evaluation import ModelEvaluator
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Configuração inicial
def configurar_pagina():
    st.set_page_config(
        page_title="ML Globo",
        layout="wide",
        initial_sidebar_state="expanded",
        page_icon="./assets/globo-icone.png"
    )
    try:
        with open('styles/custom.css') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except Exception:
        pass

    if 'models' not in st.session_state:
        st.session_state.models = {}

# Cabeçalho com logos
def mostrar_cabecalho():
    col1, _, _ = st.columns([1, 2, 1])
    with col1:
        col_logo1, col_logo2 = st.columns([0.6, 4])
        col_logo1.image("assets/globo-icone.png", width=80)
        col_logo2.image("assets/fcamara-simple-logo.png", width=50)
    
    st.title("🔍 ML Globo")
    st.markdown("Faça o upload da sua base de dados CSV para entender melhor seus dados.")

# Processamento do arquivo
def carregar_dados():
    arquivo = st.file_uploader("", type=['csv'])
    if arquivo is not None:
        try:
            return pd.read_csv(arquivo)
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
    return None

# Métricas com textos de ajuda
def mostrar_metricas(y_test, y_pred):
    ajuda = {
        'R²': "Explica a porcentagem de variação da variável alvo que o modelo consegue prever",
        'MSE': "Média dos erros quadrados entre valores reais e previstos",
        'RMSE': "Raiz quadrada do MSE, na mesma unidade da variável original"
    }
    
    metricas = {
        'R²': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    cols = st.columns(3)
    for (nome, valor), col in zip(metricas.items(), cols):
        col.metric(
            label=str(nome),
            value=round(valor, 3),
            help=ajuda[nome],
            label_visibility="visible"
        )

# Gráfico de Previsões vs. Valores Reais
def plotar_previsoes_vs_reais(y_test, y_pred):
    fig = px.scatter(
        x=y_test,
        y=y_pred,
        labels={'x': 'Valores Reais', 'y': 'Valores Previstos'},
        title='Comparação entre Valores Reais e Previstos'
    )
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Previsão Ideal',
        line=dict(color='#50E3C2', dash='dash')
    ))
    st.plotly_chart(fig, use_container_width=True)

# Modelos disponíveis
MODELOS = {
    'Regressão Linear': {
        'classe': LinearRegression,
        'descricao': """
        Modelo estatístico fundamental que assume uma relação linear entre as variáveis.
        Ideal para:
        - Relações diretas entre variáveis
        - Previsões simples e interpretáveis
        - Identificar força de impacto das variáveis
        """,
        'equacao': True
    },
    'XGBoost': {
        'classe': xgb.XGBRegressor,
        'parametros': {'objective': 'reg:squarederror', 'n_estimators': 100, 'random_state': 42},
        'descricao': """
        Algoritmo avançado de aprendizado de máquina baseado em árvores de decisão.
        Vantagens:
        - Captura relações não lineares
        - Robustez a outliers
        - Alta precisão nas previsões
        """
    }
}

def main():
    configurar_pagina()
    mostrar_cabecalho()
    
    df = carregar_dados()
    if df is not None:
        # 1) Exibição da base e matriz de correlação completa (todas as colunas)
        st.subheader("Pré-visualização dos Dados")
        st.dataframe(df.head(), hide_index=True)
        
        # Converter colunas não numéricas para códigos numéricos para cálculo da correlação
        df_corr = df.copy()
        for col in df_corr.columns:
            if not pd.api.types.is_numeric_dtype(df_corr[col]):
                df_corr[col] = df_corr[col].astype('category').cat.codes
        corr_matrix_full = df_corr.corr().fillna(0)
        
        st.subheader("Matriz de Correlação Completa (Todas as Colunas)")
        fig_corr_full = px.imshow(
            corr_matrix_full, 
            text_auto=True, 
            aspect="auto", 
            color_continuous_scale=[
                (0.0, "lightcoral"),
                (0.5, "yellow"),
                (1.0, "lightgreen")
            ]
        )
        st.plotly_chart(fig_corr_full, use_container_width=True)
        
        # 2 & 3) Configuração de Datas e Agrupamento
        st.subheader("Configuração de Datas")
        st.markdown("Escolha o intervalo de datas e a granularidade para a análise.")
        
        if 'dt_partition' in df.columns:
            try:
                df['dt_partition'] = pd.to_datetime(df['dt_partition'])
            except Exception as e:
                st.error(f"Erro ao converter 'dt_partition' para datetime: {e}")
                return
            
            min_date = df['dt_partition'].min().date()
            max_date = df['dt_partition'].max().date()
            date_range = st.date_input(
                "Selecione o intervalo de datas:",
                value=[min_date, max_date],
                min_value=min_date,
                max_value=max_date
            )
            if isinstance(date_range, list) and len(date_range) == 2:
                start_date, end_date = date_range
                df_filtered = df[(df['dt_partition'].dt.date >= start_date) & (df['dt_partition'].dt.date <= end_date)]
            else:
                df_filtered = df.copy()
            
            group_option = st.selectbox(
                "Agrupar dados por:",
                options=["Data e hora", "Data", "Semana", "Mês", "Quarter", "Ano"],
                help="Escolha a granularidade para agregação dos dados"
            )
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
                df_model = df_filtered.groupby(pd.Grouper(key='dt_partition', freq=freq)).mean().reset_index()
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
            st.dataframe(df_model.head(), hide_index=True)
        else:
            st.error("A coluna 'dt_partition' não foi encontrada no dataframe.")
            return
        
        # 4) Seleção da variável alvo (Y) e análise de correlação
        st.subheader("Seleção de Variáveis")
        colunas_model = df_model.columns.tolist()
        if 'dt_partition' in colunas_model:
            colunas_model.remove('dt_partition')
        alvo = st.selectbox(
            "Selecione a variável que deseja prever (Y):",
            options=colunas_model
        )
        
        # Calcular a correlação de Y com as demais colunas (convertendo para numérico se necessário)
        df_corr_model = df_model.copy()
        for col in df_corr_model.columns:
            if not pd.api.types.is_numeric_dtype(df_corr_model[col]):
                df_corr_model[col] = df_corr_model[col].astype('category').cat.codes
        corr_matrix_model = df_corr_model.corr()
        
        # Extrair a correlação de Y com as demais colunas, ordenada por valor absoluto decrescente
        corr_series = corr_matrix_model[alvo].drop(alvo).fillna(0)
        corr_series = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
        df_corr_y = corr_series.to_frame(name="Correlação")

        # Adicionar nome ao índice
        df_corr_y.index.name = "Colunas"

        custom_cmap = LinearSegmentedColormap.from_list(
            "custom_cmap", 
            [(0.0, "lightcoral"), (0.5, "yellow"), (1.0, "lightgreen")]
        )
        
        st.markdown(f"**Correlação das variáveis com `{alvo}`:**")
        st.dataframe(
            df_corr_y.style.format("{:.5f}").background_gradient(cmap=custom_cmap),
            height=400
        )
        
        # 5) Seleção das variáveis explicativas (X)
        opcoes_features = ["Selecionar todas as colunas"] + [col for col in df_model.columns if col not in ['dt_partition', alvo]]
        selecionadas = st.multiselect(
            "Selecione as variáveis explicativas (X):",
            options=opcoes_features,
            help="Selecione as features desejadas. Se escolher 'Selecionar todas as colunas', todas as colunas serão usadas."
        )
        if "Selecionar todas as colunas" in selecionadas:
            features = [col for col in df_model.columns if col not in ['dt_partition', alvo]]
        else:
            features = selecionadas
        
        if not features:
            st.warning("Selecione pelo menos uma variável para X.")
            return
        
        # 6) Botão para executar os modelos
        if st.button("Rodar Modelos"):
            X = df_model[features].fillna(df_model[features].mean())
            y = df_model[alvo]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            abas = st.tabs(list(MODELOS.keys()) + [
                "CatBoost", 
                "LightGBM", 
                "Modelo de Marketing Mix", 
                "Deep Learning (LSTM)",
                "Comparação de Modelos"
            ])
            
            for i, (nome_modelo, config) in enumerate(MODELOS.items()):
                with abas[i]:
                    st.markdown(f"### {nome_modelo}")
                    st.markdown(config['descricao'])
                    
                    with st.spinner(f"Treinando {nome_modelo}..."):
                        if 'parametros' in config:
                            modelo = config['classe'](**config['parametros'])
                        else:
                            modelo = config['classe']()
                            
                        modelo.fit(X_train, y_train)
                        st.session_state.models[nome_modelo] = modelo
                        
                        y_pred = modelo.predict(X_test)
                        
                        mostrar_metricas(y_test, y_pred)
                        
                        if config.get('equacao', False):
                            st.subheader("Equação do Modelo")
                            equacao = f"{alvo} = {modelo.intercept_:.4f}"
                            for coef, feature in zip(modelo.coef_, features):
                                equacao += f" + ({coef:.4f} × {feature})"
                            st.code(equacao)
                        
                        plotar_previsoes_vs_reais(y_test, y_pred)
                        
                        st.subheader("Importância das Variáveis")
                        if hasattr(modelo, 'feature_importances_'):
                            importancia = modelo.feature_importances_
                        else:
                            importancia = np.abs(modelo.coef_)
                            
                        df_importancia = pd.DataFrame({
                            'Variável': features,
                            'Importância': importancia
                        }).sort_values('Importância', ascending=True)
                        
                        fig = px.bar(
                            df_importancia,
                            x='Importância',
                            y='Variável',
                            orientation='h'
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Abas para funcionalidades futuras
            for i in range(len(MODELOS), len(abas)-1):
                with abas[i]:
                    st.info("🚧 Funcionalidade em desenvolvimento!")
                    st.markdown("""
                    **Em breve:**
                    - Treinamento do modelo
                    - Métricas de performance
                    - Visualizações detalhadas
                    - Exportação de resultados
                    """)
            
            with abas[-1]:
                st.markdown("""
                ### Comparação de Modelos
                Compare o desempenho de diferentes modelos usando validação cruzada.
                Isso nos ajuda a entender:
                - Qual modelo performa melhor em média
                - Quão estáveis são as previsões de cada modelo
                - Se as diferenças de performance são estatisticamente significativas
                """)
                
                if len(st.session_state.models) > 0:
                    with st.spinner("Executando validação cruzada..."):
                        avaliador = ModelEvaluator(st.session_state.models)
                        avaliador.perform_cross_validation(X, y, cv=5)
                        
                        st.subheader("Resumo da Validação Cruzada")
                        resumo = avaliador.get_summary_stats()
                        st.dataframe(resumo.style.format("{:.4f}"))
                        
                        st.subheader("Distribuição de Performance")
                        figura_cv = avaliador.plot_cv_comparison()
                        st.plotly_chart(figura_cv, use_container_width=True)
                        
                        st.subheader("Comparação Estatística")
                        st.markdown("""
                        Valores-p para comparações pareadas entre modelos.
                        Valores < 0.05 indicam diferenças estatisticamente significativas.
                        """)
                        valores_p = avaliador.perform_statistical_test()
                        st.dataframe(valores_p.style.format("{:.4f}"))
                else:
                    st.info("Treine pelo menos um modelo para ver a comparação.")
    
    st.markdown("---\nFeito com ❤️ FCamara | Value Creation | Sales Boost")

if __name__ == "__main__":
    main()
