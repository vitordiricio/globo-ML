
import streamlit as st
import numpy as np

from utils.graphs_views import (
    mostrar_metricas, 
    plot_previsoes_vs_reais, 
    plot_heatmap_correlation_total, 
    cria_dataframe_correlacao_com_target,
    plot_feature_importance
)

from utils.data_processing import group_and_filter_by_date, prepare_features_for_modeling

from utils.ml_models import AVAILABLE_MODELS

from utils.model_evaluation import ModelEvaluator


def playground(df_merged):
    st.title("Playground" )

    if df_merged is not None:
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

        # Initialize session state for storing selections
        if 'previous_alvo' not in st.session_state:
            st.session_state.previous_alvo = None
        if 'selected_features_rs' not in st.session_state:
            st.session_state.selected_features_rs = []
        if 'selected_features_gp' not in st.session_state:
            st.session_state.selected_features_gp = []
        if 'selected_features_linear' not in st.session_state:
            st.session_state.selected_features_linear = []
        # Add session state for external features
        if 'selected_features_external' not in st.session_state:
            st.session_state.selected_features_external = []

        # Y selection gets full width
        alvo = st.selectbox(
            "Selecione a variável que deseja prever (Y):",
            options=colunas_model,
            key="alvo_selectbox"
        )
        
        # Check if Y changed and update the X variables accordingly
        if st.session_state.previous_alvo != alvo:
            # If Y changed, we need to update our X options
            # But we want to preserve the selection when possible
            
            # Filter out the new Y variable from the previous selections
            updated_features_rs = [feat for feat in st.session_state.selected_features_rs 
                                if feat != alvo and feat != "Selecionar todas as colunas RS"]
            updated_features_gp = [feat for feat in st.session_state.selected_features_gp 
                                if feat != alvo and feat != "Selecionar todas as colunas GP"]
            updated_features_linear = [feat for feat in st.session_state.selected_features_linear 
                                    if feat != alvo and feat != "Selecionar todas as colunas LINEAR"]
            updated_features_external = [feat for feat in st.session_state.selected_features_external
                                        if feat != alvo and feat != "Selecionar todas as colunas EXTERNO"]
            
            st.session_state.selected_features_rs = updated_features_rs
            st.session_state.selected_features_gp = updated_features_gp
            st.session_state.selected_features_linear = updated_features_linear
            st.session_state.selected_features_external = updated_features_external
            st.session_state.previous_alvo = alvo

        # Filter columns by prefix
        rs_columns = ["Selecionar todas as colunas RS"] + [col for col in df_model.columns 
                    if col.startswith('RS_') and col != alvo]
        gp_columns = ["Selecionar todas as colunas GP"] + [col for col in df_model.columns 
                    if col.startswith('GP_') and col != alvo]
        linear_columns = ["Selecionar todas as colunas LINEAR"] + [col for col in df_model.columns 
                        if col.startswith('LINEAR_') and col != alvo]
        external_columns = ["Selecionar todas as colunas EXTERNO"] + [col for col in df_model.columns
                            if col.startswith('EXTERNO_') and col != alvo]

        # Create four separate selectors for each prefix
        st.markdown("### Selecione as variáveis explicativas (X):")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Variáveis de Redes Sociais")
            selected_rs = st.multiselect(
                "Selecione as variáveis de Redes Sociais:",
                options=rs_columns,
                default=st.session_state.selected_features_rs,
                help="Selecione as features desejadas de Redes Sociais."
            )
            st.session_state.selected_features_rs = selected_rs

        with col2:
            st.markdown("#### Variáveis de GloboPlay")
            selected_gp = st.multiselect(
                "Selecione as variáveis do GloboPlay:",
                options=gp_columns,
                default=st.session_state.selected_features_gp,
                help="Selecione as features desejadas do GloboPlay."
            )
            st.session_state.selected_features_gp = selected_gp

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### Variáveis de TV Linear")
            selected_linear = st.multiselect(
                "Selecione as variáveis de TV Linear:",
                options=linear_columns,
                default=st.session_state.selected_features_linear,
                help="Selecione as features desejadas de TV Linear."
            )
            st.session_state.selected_features_linear = selected_linear
        
        with col4:
            st.markdown("#### Variáveis Externas")
            selected_external = st.multiselect(
                "Selecione as variáveis externas:",
                options=external_columns,
                default=st.session_state.selected_features_external,
                help="Selecione dados econômicos, eventos externos e outros fatores."
            )
            st.session_state.selected_features_external = selected_external

        # Create separate correlation dataframes for each prefix
        st.markdown("### Correlação com variável alvo por tipo de dados:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Redes Sociais")
            cria_dataframe_correlacao_com_target(df_model, alvo, prefix='RS_')

        with col2:
            st.markdown("#### GloboPlay")
            cria_dataframe_correlacao_com_target(df_model, alvo, prefix='GP_')

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("#### TV Linear")
            cria_dataframe_correlacao_com_target(df_model, alvo, prefix='LINEAR_')
        
        with col4:
            st.markdown("#### Fatores Externos")
            cria_dataframe_correlacao_com_target(df_model, alvo, prefix='EXTERNO_')
        
        if st.button("Rodar Modelos"):
            if "models" not in st.session_state:
                st.session_state.models = {}
            if "results" not in st.session_state:
                st.session_state.results = {}
            
            # Process "Selecionar todas" options for each category
            rs_features = []
            if "Selecionar todas as colunas RS" in selected_rs:
                rs_features = [col for col in df_model.columns if col.startswith('RS_') and col != alvo]
            else:
                rs_features = selected_rs
                
            gp_features = []
            if "Selecionar todas as colunas GP" in selected_gp:
                gp_features = [col for col in df_model.columns if col.startswith('GP_') and col != alvo]
            else:
                gp_features = selected_gp
                
            linear_features = []
            if "Selecionar todas as colunas LINEAR" in selected_linear:
                linear_features = [col for col in df_model.columns if col.startswith('LINEAR_') and col != alvo]
            else:
                linear_features = selected_linear
            
            external_features = []
            if "Selecionar todas as colunas EXTERNO" in selected_external:
                external_features = [col for col in df_model.columns if col.startswith('EXTERNO_') and col != alvo]
            else:
                external_features = selected_external
            
            # Combine all selected features
            selected_columns = rs_features + gp_features + linear_features + external_features
            
            if not selected_columns:
                st.error("Por favor, selecione pelo menos uma variável explicativa (X).")
            else:
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