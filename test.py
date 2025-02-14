# main.py
import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processor import DataProcessor
from utils.ml_models import RegressionAnalyzer
from utils.model_evaluation import ModelEvaluator
import xgboost as xgb

# Inicializa o estado da sessão para armazenar modelos
if 'models' not in st.session_state:
    st.session_state.models = {}

# Configuração da página
st.set_page_config(
    page_title="ML Globo",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="./assets/globo-icone.png"
)

col1, col2, col3 = st.columns([1, 2, 1])

# Criar duas colunas com proporções diferentes - a primeira um pouco maior que a segunda
logo1, logo2 = col1.columns([0.6, 4])  # Ajuste esses valores conforme necessário

with logo1:
    st.image("assets/globo-icone.png", width=80)
    
with logo2:
    st.image("assets/fcamara-simple-logo.png", width=50)

# Carrega CSS personalizado
with open('styles/custom.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Cabeçalho
st.title("🔍 ML Globo")
st.markdown("""
    Faça o upload da sua base de dados CSV para entender melhor seus dados.
""")

# Funções auxiliares
def display_metrics(metrics):
    st.subheader("Métricas de Performance do Modelo")
    metric_cols = st.columns(3)
    help_texts = {
        'R²': "O Coeficiente de Determinação (R²) mede a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Um valor mais próximo de 1 indica um modelo melhor.",
        'MSE': "O Erro Quadrático Médio (MSE) é a média dos quadrados dos erros, ou seja, a diferença média ao quadrado entre os valores previstos e os valores reais. Um valor menor indica um modelo melhor.",
        'RMSE': "A Raiz do Erro Quadrático Médio (RMSE) é a raiz quadrada do MSE. Ela representa, aproximadamente, a diferença média entre os valores previstos e os valores reais. Um valor menor indica um modelo melhor."
    }
    for i, (metric, value) in enumerate(metrics.items()):
        with metric_cols[i]:
            st.metric(metric, value, help=help_texts[metric])

def display_visualizations(y_true, y_pred, feature_columns, model):
    st.subheader("Visualizações")

    # Gráfico de Valores Reais vs Preditos
    fig = model.create_prediction_plot(y_true, y_pred)
    st.plotly_chart(fig, use_container_width=True)

    # Importância das Features
    if hasattr(model.model, 'coef_'):
        importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importância': np.abs(model.model.coef_)
        })
    elif hasattr(model.model, 'feature_importances_'):
        importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importância': model.model.feature_importances_
        })
    else:
        importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importância': [0]*len(feature_columns)
        })
    importance = importance.sort_values('Importância', ascending=True)

    fig_importance = px.bar(
        importance,
        x='Importância',
        y='Feature',
        orientation='h',
        title='Importância das Features'
    )
    fig_importance.update_layout(template='plotly_white')
    st.plotly_chart(fig_importance, use_container_width=True)

def export_results(y_true, y_pred, filename):
    st.subheader("Exportar Resultados")
    results_df = pd.DataFrame({
        'Valor Real': y_true,
        'Valor Predito': y_pred
    })
    csv = results_df.to_csv(index=False)
    st.download_button(
        label="Baixar Resultados em CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

# Upload de arquivo
uploaded_file = st.file_uploader("Faça o upload dos seus dados para começar a análise :)", type=['csv'])

if uploaded_file is not None:
    try:
        # Inicializa o DataProcessor
        data_processor = DataProcessor()

        # Carrega e valida os dados
        df, error = data_processor.validate_csv(uploaded_file)
        if error:
            st.error(f"Erro ao carregar o arquivo: {error}")
            st.stop()

        # Visualização dos dados
        st.subheader("Visualizar Dados")
        st.dataframe(df.head(), hide_index=True)

        # Seleção de colunas
        numeric_columns = data_processor.get_numeric_columns(df)

        col1, col2 = st.columns(2)
        with col1:
            target_column = st.selectbox(
                "Selecione a variável que quer entender (Y)",
                options=numeric_columns,
                help="A coluna que você quer entender melhor a partir de outras"
            )

        with col2:
            available_features = [col for col in numeric_columns if col != target_column]

            # Adiciona opção de selecionar todas as features
            all_option = "Selecionar Todas as Features"
            options = [all_option] + available_features

            selected = st.multiselect(
                "Selecione as métricas explicativas (X)",
                options=options,
                help="As métricas/colunas que você quer usar para tentar predizer e entender Y"
            )

            # Trata a opção de selecionar todas as features
            if all_option in selected or not selected:
                feature_columns = available_features
            else:
                feature_columns = selected

        if feature_columns and target_column:
            # Prepara os dados
            X_train, X_test, y_train, y_test, scaler = data_processor.prepare_data(df, target_column, feature_columns)

            # Aba de seleção de modelos
            st.subheader("Modelos")
            tabs = st.tabs([
                "Regressão Linear",
                "XGBoost",
                "CatBoost",
                "LightGBM",
                "Market Mix Modeling",
                "Deep Learning (LSTM)",
                "Comparação de modelos",
            ])

            # Aba de Regressão Linear
            with tabs[0]:
                st.markdown("""
                    ### Regressão Linear
                    A regressão linear é um modelo estatístico fundamental que assume uma relação linear entre as variáveis de entrada e saída.
                    É útil para:

                    - Compreender relações diretas entre variáveis
                    - Fazer previsões simples e interpretáveis
                    - Identificar a força do impacto das variáveis explicativas sobre a variável alvo
                """)

                with st.spinner("Treinando modelo de Regressão Linear..."):
                    # Inicializa o RegressionAnalyzer
                    regression_analyzer = RegressionAnalyzer()

                    # Treina o modelo
                    regression_analyzer.train_model(X_train, y_train)
                    st.session_state.models['Regressão Linear'] = regression_analyzer.model

                    # Faz predições
                    y_pred = regression_analyzer.predict(X_test)

                    # Calcula métricas
                    metrics = regression_analyzer.calculate_metrics(y_test, y_pred)

                    # Exibe a equação do modelo
                    st.subheader("Equação do Modelo")
                    intercept = regression_analyzer.model.intercept_
                    coefficients = regression_analyzer.model.coef_
                    equation = f"{target_column} = {intercept:.4f}"
                    for coef, feature in zip(coefficients, feature_columns):
                        equation += f" + ({coef:.4f} × {feature})"
                    st.code(equation)

                    # Exibe métricas com ícones de ajuda
                    display_metrics(metrics)

                    # Seção de visualização
                    display_visualizations(y_test, y_pred, feature_columns, regression_analyzer)

                # Exporta resultados
                export_results(y_test, y_pred, "regressao_linear_resultados.csv")

            # Aba de XGBoost
            with tabs[1]:
                st.markdown("""
                    ### Regressão XGBoost
                    XGBoost (eXtreme Gradient Boosting) é uma implementação poderosa e eficiente de árvores de decisão com gradiente impulsionado.
                    Principais vantagens:

                    - Seleção automática de variáveis e classificação de importância
                    - Lida com relacionamentos não lineares
                    - Robusto a outliers e valores ausentes
                    - Alta precisão nas previsões
                """)

                with st.spinner("Treinando modelo XGBoost..."):
                    # Inicializa o RegressionAnalyzer com o modelo XGBoost
                    xgb_model = xgb.XGBRegressor(
                        objective='reg:squarederror',
                        n_estimators=100,
                        learning_rate=0.1,
                        random_state=42
                    )
                    regression_analyzer = RegressionAnalyzer(model=xgb_model)
                    regression_analyzer.train_model(X_train, y_train)
                    st.session_state.models['XGBoost'] = regression_analyzer.model

                    # Faz predições
                    y_pred = regression_analyzer.predict(X_test)

                    # Calcula métricas
                    metrics = regression_analyzer.calculate_metrics(y_test, y_pred)

                    # Exibe métricas com ícones de ajuda
                    display_metrics(metrics)

                    # Seção de visualização
                    display_visualizations(y_test, y_pred, feature_columns, regression_analyzer)

                # Exporta resultados
                export_results(y_test, y_pred, "xgboost_resultados.csv")

            # Aba de Comparação de Modelos
            with tabs[6]:
                st.markdown("""
                    ### Comparação de Modelos
                    Compare a performance de diferentes modelos usando validação cruzada.
                    Isso ajuda a entender:
                    - Qual modelo tem melhor performance em média
                    - Quão estáveis são as previsões de cada modelo
                    - Se as diferenças de performance são estatisticamente significativas
                """)

                if len(st.session_state.models) > 0:
                    with st.spinner("Realizando validação cruzada..."):
                        # Prepara o conjunto de dados completo
                        X_full, _, y_full, _, _, _ = data_processor.prepare_data(df, target_column, feature_columns, test_size=0)

                        # Inicializa o ModelEvaluator
                        evaluator = ModelEvaluator(st.session_state.models)

                        # Realiza validação cruzada
                        evaluator.perform_cross_validation(X_full, y_full, cv=5)

                        # Exibe estatísticas resumidas
                        st.subheader("Sumário da Validação Cruzada")
                        summary_stats = evaluator.get_summary_stats()
                        st.dataframe(summary_stats.style.format("{:.4f}"))

                        # Plota comparação
                        st.subheader("Distribuição de Performance")
                        cv_plot = evaluator.plot_cv_comparison()
                        st.plotly_chart(cv_plot, use_container_width=True)

                        # Comparação estatística
                        st.subheader("Comparação Estatística")
                        st.markdown("""
                            Valores-p para comparações pareadas de modelos.
                            Valores < 0.05 indicam diferenças estatisticamente significativas.
                        """)
                        p_values = evaluator.perform_statistical_test()
                        st.dataframe(p_values.style.format("{:.4f}"))
                else:
                    st.info("Treine pelo menos um modelo para ver a comparação.")

            # Espaços reservados para outros modelos
            model_names = [
                "CatBoost",
                "LightGBM",
                "Market Mix Modeling",
                "Deep Learning (LSTM)"
            ]
            for i, model_name in enumerate(model_names, 2):
                with tabs[i]:
                    st.info(f"🚧 Implementação de {model_name} em breve!")
                    st.markdown(f"""
                        ### {model_name}
                        Este modelo está atualmente sendo implementado. Ele irá incluir:
                        - Treinamento e predições do modelo
                        - Métricas de performance
                        - Visualização de importância das features
                        - Equação do modelo ou análise de impacto das features
                        - Exportação de resultados
                    """)

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {str(e)}")
# Rodapé
st.markdown("""
    ---
    Feito com ❤️ FCamara | Data Science
""")