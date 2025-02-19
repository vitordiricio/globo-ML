# utils/visualization.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

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
    Plota um gráfico de barras horizontal mostrando a importância das variáveis.
    """
    df_importancia = pd.DataFrame({
        'Variável': features,
        'Importância': importance
    }).sort_values('Importância', ascending=True)
    fig = px.bar(
        df_importancia,
        x='Importância',
        y='Variável',
        orientation='h'
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(corr_matrix, title):
    """
    Plota um heatmap da matriz de correlação usando Plotly.
    
    Args:
        corr_matrix (DataFrame): Matriz de correlação.
        title (str): Título do gráfico.
    """
    fig = px.imshow(
        corr_matrix, 
        text_auto=True, 
        aspect="auto", 
        color_continuous_scale=[
            (0.0, "lightcoral"),
            (0.5, "yellow"),
            (1.0, "lightgreen")
        ]
    )
    fig.update_layout(title=title, template='plotly_white', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
