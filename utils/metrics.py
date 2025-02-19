# utils/metrics.py
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score, mean_squared_error

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
