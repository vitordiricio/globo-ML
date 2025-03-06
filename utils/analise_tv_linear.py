# utils/analise_tv_linear.py
import streamlit as st
import pandas as pd
import plotly.express as px

def analise_tv_linear(df):
    """
    Performs in-depth analysis of TV Linear data with multiple granularities,
    showing audience metrics, comparisons with competitors, and evolution over time.
    
    Args:
        df (DataFrame): Processed dataframe with LINEAR_ prefixed columns
    """
    
    st.header("📺 TV Linear - Audiência")
    
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
    
    # 2. Metrics Tables for each granularity
    st.subheader("Métricas Resumidas")

    st.markdown("""
    Nesta seção, você encontra um resumo das principais métricas da TV Linear, organizadas por diferentes granularidades de tempo: semana, dia ou hora. Use o filtro de tempo no topo da tela para alternar entre essas visões. As métricas apresentadas incluem cov% (cobertura), shr% (share de audiência) e TVR% (telespectadores por ponto de audiência), cada uma com seu valor médio, desvio padrão e quantidade de registros.
    """)
    
    # Check if we have the required columns
    required_cols = [
        'LINEAR_GLOBO_cov%', 'LINEAR_GLOBO_shr%', 'LINEAR_GLOBO_tvr%'
    ]
    
    if all(col in df.columns for col in required_cols):
        # Create three columns for layout
        col1, col2 = st.columns(2)
        
        # Function to create metrics table
        def create_metrics_table(data_df, granularity_name):
            metrics = {
                "📻 cov% (cobertura)": "LINEAR_GLOBO_cov%",
                "% shr% (share)": "LINEAR_GLOBO_shr%",
                "🕒 TVR% (rating)": "LINEAR_GLOBO_tvr%"
            }
            
            metrics_data = []
            for label, col_name in metrics.items():
                metrics_data.append({
                    "Métrica": label,
                    "Valor Médio": f"{data_df[col_name].mean():.2f}",
                    "Desvio Padrão": f"{data_df[col_name].std():.2f}",
                    "Número de Linhas": f"{len(data_df)}"
                })
            
            return pd.DataFrame(metrics_data)
        

        # Display tables in each column
        with col1:
            st.markdown("### Semanal")
            st.dataframe(create_metrics_table(df_weekly, "Semanal"), hide_index=True)
        
        with col2:
            st.markdown("### Diário")
            st.dataframe(create_metrics_table(df_daily, "Diário"), hide_index=True)
        
    else:
        st.warning("Dados insuficientes para exibir métricas. Alguns dados podem estar faltando.")
    
    # 3. Evolution Chart with Metric Selection
    st.subheader("Evolução da Audiência vs Concorrentes")
    st.markdown("""

    Esta análise apresenta dois níveis de comparação de audiência:

    **1. Visão Geral:** Mostra **a performance consolidada da Globo comparada à média dos concorrentes** em cada uma das métricas principais: **cov%**, **shr%** e **TVR%**. Use o filtro de métrica para alternar entre essas visões. O gráfico mostra a evolução da Globo e da concorrência ao longo do tempo, ajudando a identificar se há tendência de ganho ou perda de audiência relativa.

    **2. Comparação Específica:** Aqui você pode comparar diretamente a Globo com **qualquer concorrente específico** (SBT, Record, Band, etc). Escolha o concorrente desejado no filtro e selecione qual métrica quer visualizar (cov%, shr%, TVR%). O gráfico mostra a evolução lado a lado, enquanto a tabela abaixo traz **indicadores estatísticos** da relação entre Globo e esse concorrente.
    """)
    
    # Get competitor channels (those with LINEAR_ prefix but not GLOBO)
    competitors = [col.split('_')[1] for col in df.columns 
                   if col.startswith('LINEAR_') and 
                   col.endswith('_rat%') and 
                   'GLOBO' not in col]
    
    # Filter out any entries that might contain apostrophes or special characters
    competitors = [comp for comp in competitors if "'" not in comp]
    
    # Create 3 columns for dropdowns
    col1, col2, col3 = st.columns(3)
    
    # Granularity selection in first column
    granularity_options = {
        "Diário": df_daily,
        "Semanal": df_weekly,
    }
    
    with col1:
        granularity = st.selectbox(
            "Selecione a granularidade:",
            options=list(granularity_options.keys())
        )
    
    # Get the selected dataframe
    selected_df = granularity_options[granularity]
    
    # Metric selection in second column
    metrics_options = {
        "cov% (cobertura)": "cov%",
        "shr% (share)": "shr%",
        "TVR% (rating)": "tvr%"
    }
    
    with col2:
        selected_metric = st.selectbox(
            "Selecione a métrica:",
            options=list(metrics_options.keys())
        )
    
    metric_suffix = metrics_options[selected_metric]
    
    # Competitor selection in third column
    with col3:
        visualization_options = ["Concorrentes Agregados"] + competitors
        selected_viz = st.selectbox(
            "Selecione a visualização:",
            options=visualization_options
        )
    
    # Create the visualization
    if 'data_hora' in selected_df.columns:
        globo_col = f"LINEAR_GLOBO_{metric_suffix}"
        
        # Prepare plot data
        if selected_viz == "Concorrentes Agregados":
            # Create a dataframe for the plot
            plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
            plot_data['Globo'] = selected_df[globo_col]
            
            # Calculate average of competitors
            comp_cols = [f"LINEAR_{comp}_{metric_suffix}" for comp in competitors 
                         if f"LINEAR_{comp}_{metric_suffix}" in selected_df.columns]
            
            if comp_cols:
                plot_data['Concorrentes (Média)'] = selected_df[comp_cols].mean(axis=1)
                
                # Create line chart
                fig = px.line(
                    plot_data, 
                    x='Data', 
                    y=['Globo', 'Concorrentes (Média)'],
                    title=f"Evolução de {selected_metric} ao Longo do Tempo - {granularity}",
                    labels={'value': selected_metric, 'variable': 'Canal'}
                )
                
                # Customize lines
                for trace in fig.data:
                    if trace.name == 'Globo':
                        trace.line.color = '#0D47A1'  # Azul Globo
                        trace.line.width = 3
                    else:
                        trace.line.color = 'gray'
                        trace.line.width = 1.5
            else:
                # Create line chart with just Globo data
                fig = px.line(
                    plot_data, 
                    x='Data', 
                    y=['Globo'],
                    title=f"Evolução de {selected_metric} ao Longo do Tempo - {granularity}",
                    labels={'value': selected_metric, 'variable': 'Canal'}
                )
                
                # Customize line
                fig.data[0].line.color = '#0D47A1'  # Azul Globo
                fig.data[0].line.width = 3
                
                st.warning("Não foram encontrados dados de concorrentes para comparação agregada.")
        else:  # Specific competitor
            selected_competitor = selected_viz
            
            # Create a dataframe for the plot
            plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
            plot_data['Globo'] = selected_df[globo_col]
            
            comp_col = f"LINEAR_{selected_competitor}_{metric_suffix}"
            if comp_col in selected_df.columns:
                plot_data[selected_competitor] = selected_df[comp_col]
                
                # Create line chart
                fig = px.line(
                    plot_data, 
                    x='Data', 
                    y=['Globo', selected_competitor],
                    title=f"Globo vs {selected_competitor} - {selected_metric} ({granularity})",
                    labels={'value': selected_metric, 'variable': 'Canal'}
                )
                
                # Customize lines
                for trace in fig.data:
                    if trace.name == 'Globo':
                        trace.line.color = '#0D47A1'  # Azul Globo
                        trace.line.width = 3
                    else:
                        trace.line.color = '#757575'  # Gray
                        trace.line.width = 2
            else:
                # Create line chart with just Globo data
                fig = px.line(
                    plot_data, 
                    x='Data', 
                    y=['Globo'],
                    title=f"Evolução de {selected_metric} ao Longo do Tempo - {granularity}",
                    labels={'value': selected_metric, 'variable': 'Canal'}
                )
                
                # Customize line
                fig.data[0].line.color = '#0D47A1'  # Azul Globo
                fig.data[0].line.width = 3
                
                st.warning(f"Não foram encontrados dados para o concorrente {selected_competitor}.")
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='#F5F5F5',
            font=dict(family="Roboto, Arial", color="#212121"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=50, l=50, r=20, b=50),
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dados insuficientes para gerar o gráfico de evolução.")
    
    # 5. Notes and Documentation
    st.subheader("Notas e Documentação")
    
    with st.expander("Informações sobre os dados"):
        st.markdown("""
        ### Fonte dos Dados
        
        Os dados de audiência de TV Linear são coletados pela Kantar IBOPE Media, 
        utilizando o serviço de audiência de televisão. O painel representa 
        estatisticamente a população com televisores no Brasil.
        
        ### Métricas Disponíveis
        
        - **cov% (Cobertura)**: Percentual de telespectadores que assistiram ao conteúdo 
          por pelo menos um minuto em relação ao universo total.
          
        - **shr% (Share)**: Percentual de telespectadores que estavam assistindo a um 
          determinado canal em relação ao total de televisores ligados no mesmo período.
          
        - **TVR% (Rating)**: Percentual de telespectadores que assistiram a um programa 
          ou canal em relação ao universo total, independentemente de quantos televisores 
          estavam ligados.

        """)