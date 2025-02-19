# app.py
import streamlit as st
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from utils.config import configurar_pagina
from utils.ui import mostrar_cabecalho
from utils.data_processing import (
    carregar_dados,
    filter_and_group_by_date,
    convert_non_numeric_to_codes,
    calculate_correlation_series
)
from utils.metrics import mostrar_metricas
from utils.visualization import (
    plot_previsoes_vs_reais,
    plot_feature_importance,
    plot_correlation_heatmap
)
from utils.model_evaluation import ModelEvaluator

# Import model classes directly (each model encapsulates its own logic)
from models.linear_regression import LinearRegressionModel
from models.xgboost_model import XGBoostModel

def main():
    configurar_pagina()
    mostrar_cabecalho()
    
    # Carrega a base de dados
    df = carregar_dados()
    if df is not None:
        st.subheader("Pré-visualização dos Dados")
        st.dataframe(df, hide_index=True, height=250)
        
        # Matriz de correlação completa
        df_corr_numeric = convert_non_numeric_to_codes(df)
        corr_matrix_full = df_corr_numeric.corr().fillna(0)
        
        st.subheader("Matriz de Correlação Completa (Todas as Colunas)")
        plot_correlation_heatmap(corr_matrix_full, "Matriz de Correlação Completa")
        
        # Configuração de datas e agrupamento
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
            
            df_model = filter_and_group_by_date(df, 'dt_partition', group_option, date_range)
            
            st.markdown("Pré-visualização dos dados após tratamento:")
            st.dataframe(df_model, hide_index=True, height=250)
        else:
            st.error("A coluna 'dt_partition' não foi encontrada no dataframe.")
            return
        
        # Seleção de variáveis
        st.subheader("Seleção de Variáveis")
        colunas_model = df_model.columns.tolist()
        if 'dt_partition' in colunas_model:
            colunas_model.remove('dt_partition')
        col_y, col_x = st.columns(2)
        with col_y:
            alvo = st.selectbox(
                "Selecione a variável que deseja prever (Y):",
                options=colunas_model
            )
        with col_x:
            opcoes_features = ["Selecionar todas as colunas"] + [col for col in df_model.columns if col not in ['dt_partition', alvo]]
            selecionadas = st.multiselect(
                "Selecione as variáveis explicativas (X):",
                options=opcoes_features,
                help="Selecione as features desejadas. Se escolher 'Selecionar todas as colunas', todas as colunas serão usadas."
            )
        
        # Exibe correlação entre as variáveis e o alvo
        corr_series = calculate_correlation_series(df_model, alvo)
        if corr_series is not None:
            st.markdown(f"**Correlação das variáveis com `{alvo}`:**")
            custom_cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", 
                [(0.0, "lightcoral"), (0.5, "yellow"), (1.0, "lightgreen")]
            )
            st.dataframe(
                corr_series.style.format("{:.5f}").background_gradient(cmap=custom_cmap),
                height=400
            )
        
        # Define as features a serem usadas
        features_used = (selecionadas if "Selecionar todas as colunas" not in selecionadas 
                         else [col for col in df_model.columns if col not in ['dt_partition', alvo]])
        X = df_model[features_used].fillna(
            df_model[features_used].select_dtypes(include=[np.number]).mean()
        )
        y = df_model[alvo]
        
        # Ao clicar no botão, executa os modelos
        if st.button("Rodar Modelos"):
            # Lista de classes de modelos a serem executados
            model_classes = [LinearRegressionModel, XGBoostModel]
            tabs_titles = [cls.__name__ for cls in model_classes] + [
                "CatBoost", "LightGBM", "Modelo de Marketing Mix", "Deep Learning (LSTM)", "Comparação de Modelos"
            ]
            abas = st.tabs(tabs_titles)
            
            for i, ModelClass in enumerate(model_classes):
                with abas[i]:
                    # Instancia o modelo e exibe sua descrição
                    model_instance = ModelClass()
                    st.markdown(f"### {ModelClass.__name__}")
                    st.markdown(ModelClass.description)
                    
                    with st.spinner(f"Treinando {ModelClass.__name__}..."):
                        result = model_instance.run(X, y)
                    
                    # Armazena o modelo na sessão
                    st.session_state.models[ModelClass.__name__] = model_instance
                    
                    y_test = result["y_test"]
                    y_pred = result["y_pred"]
                    
                    mostrar_metricas(y_test, y_pred)
                    
                    # Exibe a equação do modelo se aplicável (ex.: Regressão Linear)
                    if hasattr(model_instance, 'intercept_') and hasattr(model_instance, 'coef_'):
                        st.subheader("Equação do Modelo")
                        equacao = f"{alvo} = {model_instance.intercept_:.4f}"
                        for coef, feature in zip(model_instance.coef_, features_used):
                            equacao += f" + ({coef:.4f} × {feature})"
                        st.code(equacao)
                    
                    plot_previsoes_vs_reais(y_test, y_pred)
                    
                    st.subheader("Importância das Variáveis")
                    if hasattr(model_instance, 'feature_importances_'):
                        importancia = model_instance.feature_importances_
                        plot_feature_importance(features_used, importancia)
            
            # Abas para funcionalidades futuras
            for i in range(len(model_classes), len(abas)-1):
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
