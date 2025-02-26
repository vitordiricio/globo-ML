import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, root_mean_squared_error
from matplotlib.colors import LinearSegmentedColormap

def plot_previsoes_vs_reais(y_test, y_pred):
    """
    Plota um gráfico de dispersão comparando valores reais e previstos.
    """
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

def plot_feature_importance(features, importance):
    """
    Plota um gráfico de barras horizontal com cores condicionais:
    - Verde para valores positivos
    - Vermelho para valores negativos
    """
    df_importancia = pd.DataFrame({
        'Variável': features,
        'Importância': importance
    }).sort_values('Importância', ascending=True)
    
    # Create color column based on importance values
    df_importancia['Cor'] = df_importancia['Importância'].apply(
        lambda x: 'green' if x >= 0 else 'red'
    )
    
    fig = px.bar(
        df_importancia,
        x='Importância',
        y='Variável',
        orientation='h',
        color='Cor',
        color_discrete_map={
            'green': '#D9EAD3',  # Green for positive
            'red': '#EA9999'     # Red for negative
        }
    )
    
    # Remove legend if desired
    fig.update_layout(showlegend=False)
    
    st.plotly_chart(fig, use_container_width=True)

def plot_heatmap_correlation_total(df):
    """
    Plota um heatmap da matriz de correlação usando Plotly considerando todas as colunas iniciais.
    
    Args:
        corr_matrix (DataFrame): Matriz de correlação.
        title (str): Título do gráfico.
    """

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



def mostrar_metricas(y_test, y_pred):
    """
    Calcula e exibe as métricas R², MSE e RMSE usando Streamlit.
    """
    ajuda = {
        'R²': "Explica a porcentagem de variação da variável alvo que o modelo consegue prever",
        'MSE': "Média dos erros quadrados entre valores reais e previstos",
        'RMSE': "Raiz quadrada do MSE, na mesma unidade da variável original"
    }
    
    metricas = {
        'R²': r2_score(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred)
    }
    
    cols = st.columns(3)
    for (nome, valor), col in zip(metricas.items(), cols):
        col.metric(
            label=str(nome),
            value=round(valor, 3),
            help=ajuda[nome],
            label_visibility="visible"
        )

def cria_dataframe_correlacao_com_target(df_model, target):

    # Calcular a correlação de Y com as demais colunas (convertendo para numérico se necessário)
    df_corr_model = df_model.copy()
    for col in df_corr_model.columns:
        if not pd.api.types.is_numeric_dtype(df_corr_model[col]):
            df_corr_model[col] = df_corr_model[col].astype('category').cat.codes
    corr_matrix_model = df_corr_model.corr()
    corr_series = corr_matrix_model[target].drop(target).fillna(0)
    corr_series = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    df_corr_y = corr_series.to_frame(name="Correlação")
    df_corr_y.index.name = "Colunas"

    custom_cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", 
        [(0.0, "lightcoral"), (0.5, "yellow"), (1.0, "lightgreen")]
    )
    
    st.markdown(f"**Correlação das variáveis com `{target}`:**")
    st.dataframe(
        df_corr_y.style.format("{:.5f}").background_gradient(cmap=custom_cmap),
        height=400
    )

