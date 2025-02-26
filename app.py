import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt


from utils.model_evaluation import ModelEvaluator
from utils.config_page import configurar_pagina, mostrar_cabecalho
from utils.data_processing import carregar_e_tratar_dados, group_and_filter_by_date, merge_data
from utils.graphs_views import (
    mostrar_metricas, 
    plot_previsoes_vs_reais, 
    plot_heatmap_correlation_total, 
    cria_dataframe_correlacao_com_target,
    plot_feature_importance
)
from utils.ml_models import AVAILABLE_MODELS

from utils.analises_estaticas import analise_redes_sociais

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
    
    # Menu de navegação na sidebar
    menu_options = ["Home", "Análise de Redes Sociais"]
    page = st.sidebar.radio("Selecione a página", menu_options)
    
    # --- Carregamento dos dados ---
    df_redes_sociais, df_globoplay, df_tv_linear = carregar_e_tratar_dados()

    
    
    # Página Home
    if page == "Home":
        st.title("Home")
        
        if df_redes_sociais is not None and df_globoplay is not None and df_tv_linear is not None:
            df_merged = merge_data(df_redes_sociais, df_globoplay, df_tv_linear)

            st.subheader("Pré-visualização dos Dados juntos")
            st.dataframe(df_merged, hide_index=True, height=250)
            
            # Exibe o heatmap de correlação
            plot_heatmap_correlation_total(df_merged)
            
            st.subheader("Configuração de Datas")
            st.markdown("Escolha o intervalo de datas e a granularidade para a análise.")
            df_model = group_and_filter_by_date(df_merged)
            
            st.subheader("Seleção de Variáveis")
            colunas_model = df_model.columns.tolist()
            if 'data_hora' in colunas_model:
                colunas_model.remove('data_hora')
            col_y, col_x = st.columns(2)
            with col_y:
                alvo = st.selectbox(
                    "Selecione a variável que deseja prever (Y):",
                    options=colunas_model
                )
            with col_x:
                opcoes_features = ["Selecionar todas as colunas"] + [col for col in df_model.columns if col not in ['ts_published_brt', alvo]]
                selecionadas = st.multiselect(
                    "Selecione as variáveis explicativas (X):",
                    options=opcoes_features,
                    help="Selecione as features desejadas. Se escolher 'Selecionar todas as colunas', todas as colunas serão usadas."
                )
            
            cria_dataframe_correlacao_com_target(df_model, alvo)
            
            if st.button("Rodar Modelos"):
                if "models" not in st.session_state:
                    st.session_state.models = {}
                if "results" not in st.session_state:
                    st.session_state.results = {}
                
                selected_columns = (
                    selecionadas
                    if "Selecionar todas as colunas" not in selecionadas 
                    else [col for col in df_model.columns if col not in ['data_hora', alvo]]
                )
                features_used = selected_columns
                X = df_model[selected_columns].fillna(
                    df_model[selected_columns].select_dtypes(include=[np.number]).mean()
                )
                y = df_model[alvo]
                st.session_state.X = X
                st.session_state.y = y
                st.session_state.features_used = features_used
                
                # Treinamento dos modelos
                for model_class in AVAILABLE_MODELS:
                    with st.spinner(f"Treinando {model_class.model_name}..."):
                        modelo = model_class()
                        resultado = modelo.run(X, y)
                        st.session_state.models[model_class.model_name] = modelo
                        st.session_state.results[model_class.model_name] = resultado
                st.success("Modelos treinados com sucesso!")
                
                # Criação das abas para visualização dos modelos e comparação
                model_tabs = [model_class.model_name for model_class in AVAILABLE_MODELS] + ["Comparação de Modelos"]
                abas = st.tabs(model_tabs)
                
                for i, model_class in enumerate(AVAILABLE_MODELS):
                    with abas[i]:
                        st.markdown(f"### {model_class.model_name}")
                        st.markdown(model_class.description)
                        
                        with st.spinner(f"Processando {model_class.model_name}..."):
                            modelo = st.session_state.models[model_class.model_name]
                            resultado = st.session_state.results[model_class.model_name]
                            y_test = resultado["y_test"]
                            y_pred = resultado["y_pred"]
                            mostrar_metricas(y_test, y_pred)
                            
                            if model_class.model_name == "Regressão Linear":
                                st.subheader("Equação do Modelo")
                                equacao = f"{alvo} = {modelo.intercept_:.4f}"
                                for coef, feature in zip(modelo.coef_, st.session_state.features_used):
                                    equacao += f" + ({coef:.4f} × {feature})"
                                st.code(equacao)
                            
                            plot_previsoes_vs_reais(y_test, y_pred)
                            
                            st.subheader("Importância das Variáveis")
                            if hasattr(modelo, 'feature_importances_'):
                                importancia = modelo.feature_importances_
                            else:
                                importancia = np.abs(modelo.coef_)

                            plot_feature_importance(st.session_state.features_used, importancia)
                                
                
                with abas[-1]:
                    st.markdown("""
                    ### Comparação de Modelos
                    Compare o desempenho de diferentes modelos usando validação cruzada.
                    Isso nos ajuda a entender:
                    - Qual modelo performa melhor em média
                    - Quão estáveis são as previsões de cada modelo
                    - Se as diferenças de performance são estatisticamente significativas
                    """)
                    if st.session_state.models:
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
        else:
            st.info("Por favor, faça o upload dos dados para prosseguir.")
    
    # Página de Análise de Redes Sociais
    elif page == "Análise de Redes Sociais":
        st.title("Análise de Redes Sociais")
        if df_redes_sociais is not None:
            analise_redes_sociais(df_redes_sociais)
        else:
            st.warning("Por favor, faça o upload dos dados de redes sociais na Home primeiro.")
    
    st.markdown("---\nFeito com ❤️ FCamara | Value Creation | Sales Boost")

if __name__ == "__main__":
    main()
