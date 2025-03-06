# utils/analise_globoplay.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from scipy import stats

def analise_globoplay(df):
    """
    Performs in-depth analysis of Globoplay data with multiple granularities,
    showing user metrics, comparisons between user types, devices, and consumption patterns.
    Also analyzes correlation with TV Linear metrics.
    
    Args:
        df (DataFrame): Processed dataframe with GP_ and LINEAR_ prefixed columns
    """
    
    st.header("🎬 Globoplay - Consumo e Usuários")
    
    # 1. Header section with last update date
    if 'data_hora' in df.columns:
        last_date = df['data_hora'].max()
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        st.caption(f"Última atualização: {last_date.strftime('%d/%m/%Y')}")
    
    # Ensure data_hora is datetime type
    if 'data_hora' in df.columns and not pd.api.types.is_datetime64_dtype(df['data_hora']):
        df['data_hora'] = pd.to_datetime(df['data_hora'])
    
    # Create copies of the dataframe for each granularity
    df_hourly = df.copy()
    
    # Create daily aggregation
    df_daily = df.copy()
    df_daily['data'] = df_daily['data_hora'].dt.date
    df_daily = df_daily.groupby('data').mean().reset_index()
    df_daily['data_hora'] = pd.to_datetime(df_daily['data'])
    
    # Create weekly aggregation
    df_weekly = df.copy()
    df_weekly['semana'] = df_weekly['data_hora'].dt.to_period('W').astype(str)
    df_weekly = df_weekly.groupby('semana').mean().reset_index()
    df_weekly['data_hora'] = pd.to_datetime(df_weekly['semana'].str.split('/').str[0])
    
    # 2. Granularity Selection Dropdown
    granularity_options = {
        "Semanal": df_weekly,
        "Diário": df_daily,
        "Horário": df_hourly
    }
    
    granularity = st.selectbox(
        "Selecione a granularidade:",
        options=list(granularity_options.keys())
    )
    
    # Get the selected dataframe
    selected_df = granularity_options[granularity]
    
    # 3. Metrics Tables
    st.subheader("Métricas Resumidas")
    
    # Check if we have the required columns for each table
    required_cols_table1 = [
        'GP_usuários_assinantes_', 'GP_horas_consumidas_assinantes',
        'GP_usuários_de_vídeo_logados_free', 'GP_horas_consumidas_de_logados_free',
        'GP_usuários_anonimos', 'GP_horas_consumidas_de_anonimos'
    ]
    
    required_cols_table2 = [
        'GP_usuários_em_mobile', 'GP_horas_consumidas_mobile',
        'GP_usuários_em_demais_devices', 'GP_horas_consumidas_em_demais_devices'
    ]
    
    required_cols_table3 = [
        'GP_usuários_em_tvg_ao_vivo', 'GP_horas_consumidas_em_tvg_ao_vivo',
        'GP_qtd_de_integras_publicadas', 'GP_qtd_de_horas_disponíveis_integras'
    ]
    
    # Function to create metrics table
    def create_metrics_table(data_df, metrics_dict):
        metrics_data = []
        for label, col_name in metrics_dict.items():
            metrics_data.append({
                "Métrica": label,
                "Valor Médio": f"{data_df[col_name].mean():.2f}",
                "Desvio Padrão": f"{data_df[col_name].std():.2f}",
                "Número de Linhas": f"{len(data_df)}"
            })
        
        return pd.DataFrame(metrics_data)
    
    # Table 1: Assinantes vs Logados Free vs Anônimos
    st.markdown("### Assinantes vs Logados Free vs Anônimos")
    
    if all(col in selected_df.columns for col in required_cols_table1):
        table1_metrics = {
            "Horas Consumidas Assinantes": "GP_horas_consumidas_assinantes",
            "Usuários Assinantes": "GP_usuários_assinantes_",
            "Horas Consumidas Logados Free": "GP_horas_consumidas_de_logados_free",
            "Usuários Logados Free": "GP_usuários_de_vídeo_logados_free",
            "Horas Consumidas Anônimos": "GP_horas_consumidas_de_anonimos",
            "Usuários Anônimos": "GP_usuários_anonimos"
        }
        
        table1_df = create_metrics_table(selected_df, table1_metrics)
        st.table(table1_df)
        
        # Create correlation chart with TV Linear cov%
        if 'LINEAR_GLOBO_cov%' in selected_df.columns:
            st.markdown("#### Correlação com TV Linear")
            
            # Prepare data for the chart
            plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
            plot_data['Horas Consumidas Assinantes'] = selected_df['GP_horas_consumidas_assinantes']
            plot_data['Horas Consumidas Logados Free'] = selected_df['GP_horas_consumidas_de_logados_free']
            plot_data['Horas Consumidas Anônimos'] = selected_df['GP_horas_consumidas_de_anonimos']
            plot_data['cov% TV Linear'] = selected_df['LINEAR_GLOBO_cov%']
            
            # Create line chart
            fig = px.line(
                plot_data, 
                x='Data', 
                y=['Horas Consumidas Assinantes', 'Horas Consumidas Logados Free', 
                   'Horas Consumidas Anônimos', 'cov% TV Linear'],
                title=f"Engajamento por Tipo de Usuário vs cov% TV Linear - {granularity}",
                labels={'value': 'Valor', 'variable': 'Métrica'}
            )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor='#F5F5F5',
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlations
            corr_assinantes = selected_df['GP_horas_consumidas_assinantes'].corr(selected_df['LINEAR_GLOBO_cov%'])
            corr_logados = selected_df['GP_horas_consumidas_de_logados_free'].corr(selected_df['LINEAR_GLOBO_cov%'])
            corr_anonimos = selected_df['GP_horas_consumidas_de_anonimos'].corr(selected_df['LINEAR_GLOBO_cov%'])
            
            st.markdown("**Correlações com cov% TV Linear:**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Assinantes",
                    f"{corr_assinantes:.2f}",
                    delta=None,
                    delta_color="normal"
                )
                
            with col2:
                st.metric(
                    "Logados Free",
                    f"{corr_logados:.2f}",
                    delta=None,
                    delta_color="normal"
                )
                
            with col3:
                st.metric(
                    "Anônimos",
                    f"{corr_anonimos:.2f}",
                    delta=None,
                    delta_color="normal"
                )
            
            # Interpretation of correlations
            st.markdown("**Interpretação das correlações:**")
            
            for label, corr in [("Assinantes", corr_assinantes), 
                              ("Logados Free", corr_logados), 
                              ("Anônimos", corr_anonimos)]:
                if abs(corr) > 0.7:
                    st.success(f"**{label}**: Correlação forte ({corr:.2f}) com cov% TV Linear")
                elif abs(corr) > 0.3:
                    st.info(f"**{label}**: Correlação moderada ({corr:.2f}) com cov% TV Linear")
                else:
                    st.warning(f"**{label}**: Correlação fraca ({corr:.2f}) com cov% TV Linear")
    else:
        st.warning("Dados insuficientes para exibir métricas de Assinantes vs Logados Free vs Anônimos.")
    
    # Table 2: Mobile vs Outros Devices
    st.markdown("### Mobile vs Outros Devices")
    
    if all(col in selected_df.columns for col in required_cols_table2):
        table2_metrics = {
            "Horas Consumidas Mobile": "GP_horas_consumidas_mobile",
            "Usuários Mobile": "GP_usuários_em_mobile",
            "Horas Consumidas Outros Devices": "GP_horas_consumidas_em_demais_devices",
            "Usuários Outros Devices": "GP_usuários_em_demais_devices"
        }
        
        table2_df = create_metrics_table(selected_df, table2_metrics)
        st.table(table2_df)
        
        # Create correlation chart with TV Linear cov%
        if 'LINEAR_GLOBO_cov%' in selected_df.columns:
            st.markdown("#### Correlação com TV Linear")
            
            # Prepare data for the chart
            plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
            plot_data['Horas Consumidas Mobile'] = selected_df['GP_horas_consumidas_mobile']
            plot_data['Horas Consumidas Outros Devices'] = selected_df['GP_horas_consumidas_em_demais_devices']
            plot_data['cov% TV Linear'] = selected_df['LINEAR_GLOBO_cov%']
            
            # Create line chart
            fig = px.line(
                plot_data, 
                x='Data', 
                y=['Horas Consumidas Mobile', 'Horas Consumidas Outros Devices', 'cov% TV Linear'],
                title=f"Engajamento por Tipo de Device vs cov% TV Linear - {granularity}",
                labels={'value': 'Valor', 'variable': 'Métrica'}
            )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor='#F5F5F5',
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlations
            corr_mobile = selected_df['GP_horas_consumidas_mobile'].corr(selected_df['LINEAR_GLOBO_cov%'])
            corr_outros = selected_df['GP_horas_consumidas_em_demais_devices'].corr(selected_df['LINEAR_GLOBO_cov%'])
            
            st.markdown("**Correlações com cov% TV Linear:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "Mobile",
                    f"{corr_mobile:.2f}",
                    delta=None,
                    delta_color="normal"
                )
                
            with col2:
                st.metric(
                    "Outros Devices",
                    f"{corr_outros:.2f}",
                    delta=None,
                    delta_color="normal"
                )
            
            # Interpretation of correlations
            st.markdown("**Interpretação das correlações:**")
            
            for label, corr in [("Mobile", corr_mobile), ("Outros Devices", corr_outros)]:
                if abs(corr) > 0.7:
                    st.success(f"**{label}**: Correlação forte ({corr:.2f}) com cov% TV Linear")
                elif abs(corr) > 0.3:
                    st.info(f"**{label}**: Correlação moderada ({corr:.2f}) com cov% TV Linear")
                else:
                    st.warning(f"**{label}**: Correlação fraca ({corr:.2f}) com cov% TV Linear")
    else:
        st.warning("Dados insuficientes para exibir métricas de Mobile vs Outros Devices.")
    
    # Table 3: Simulcasting vs VOD
    st.markdown("### Simulcasting (TVG ao Vivo) vs VOD")
    
    if all(col in selected_df.columns for col in required_cols_table3):
        table3_metrics = {
            "Horas Consumidas TVG ao Vivo": "GP_horas_consumidas_em_tvg_ao_vivo",
            "Usuários TVG ao Vivo": "GP_usuários_em_tvg_ao_vivo",
            "Qtd Íntegras Publicadas": "GP_qtd_de_integras_publicadas",
            "Horas Disponíveis Íntegras": "GP_qtd_de_horas_disponíveis_integras"
        }
        
        table3_df = create_metrics_table(selected_df, table3_metrics)
        st.table(table3_df)
        
        # Create correlation chart with TV Linear cov%
        if 'LINEAR_GLOBO_cov%' in selected_df.columns:
            st.markdown("#### Correlação com TV Linear")
            
            # Prepare data for the chart
            plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
            plot_data['Horas Consumidas TVG ao Vivo'] = selected_df['GP_horas_consumidas_em_tvg_ao_vivo']
            
            # Calculate VOD hours consumed (not directly available, approximating with íntegras)
            plot_data['Horas Disponíveis VOD'] = selected_df['GP_qtd_de_horas_disponíveis_integras']
            plot_data['cov% TV Linear'] = selected_df['LINEAR_GLOBO_cov%']
            
            # Create line chart
            fig = px.line(
                plot_data, 
                x='Data', 
                y=['Horas Consumidas TVG ao Vivo', 'Horas Disponíveis VOD', 'cov% TV Linear'],
                title=f"Simulcasting vs VOD vs cov% TV Linear - {granularity}",
                labels={'value': 'Valor', 'variable': 'Métrica'}
            )
            
            # Update layout
            fig.update_layout(
                plot_bgcolor='#F5F5F5',
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlations
            corr_tvg = selected_df['GP_horas_consumidas_em_tvg_ao_vivo'].corr(selected_df['LINEAR_GLOBO_cov%'])
            corr_vod = selected_df['GP_qtd_de_horas_disponíveis_integras'].corr(selected_df['LINEAR_GLOBO_cov%'])
            
            st.markdown("**Correlações com cov% TV Linear:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    "TVG ao Vivo",
                    f"{corr_tvg:.2f}",
                    delta=None,
                    delta_color="normal"
                )
                
            with col2:
                st.metric(
                    "VOD (Íntegras)",
                    f"{corr_vod:.2f}",
                    delta=None,
                    delta_color="normal"
                )
            
            # Interpretation of correlations
            st.markdown("**Interpretação das correlações:**")
            
            for label, corr in [("TVG ao Vivo", corr_tvg), ("VOD (Íntegras)", corr_vod)]:
                if abs(corr) > 0.7:
                    st.success(f"**{label}**: Correlação forte ({corr:.2f}) com cov% TV Linear")
                elif abs(corr) > 0.3:
                    st.info(f"**{label}**: Correlação moderada ({corr:.2f}) com cov% TV Linear")
                else:
                    st.warning(f"**{label}**: Correlação fraca ({corr:.2f}) com cov% TV Linear")
    else:
        st.warning("Dados insuficientes para exibir métricas de Simulcasting vs VOD.")
    
    # 4. Notes and Documentation
    st.subheader("Notas e Documentação")
    
    with st.expander("Informações sobre os dados"):
        st.markdown("""
        ### Fonte dos Dados
        
        Os dados de consumo do Globoplay são coletados internamente através do sistema de 
        tracking da plataforma. Estes dados refletem a atividade real dos usuários, incluindo 
        tempo de visualização, dispositivos utilizados e tipo de conteúdo consumido.
        
        ### Métricas Disponíveis
        
        - **Usuários**: Contagem única de usuários que acessaram a plataforma.
        - **Horas Consumidas**: Total de horas de conteúdo assistidas pelos usuários.
        - **Quantidade de Íntegras**: Número total de vídeos completos disponibilizados.
        - **TVG ao Vivo**: Métricas relacionadas ao consumo de TV Globo ao vivo (simulcasting).
        - **VOD (Video On Demand)**: Métricas relacionadas ao consumo de conteúdo sob demanda.
        
        ### Tipos de Usuários
        
        - **Assinantes**: Usuários com assinatura paga do Globoplay.
        - **Logados Free**: Usuários com cadastro na plataforma, mas sem assinatura paga.
        - **Anônimos**: Usuários que acessam conteúdos gratuitos sem fazer login.
    
        """)