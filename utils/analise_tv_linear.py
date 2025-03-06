# utils/analise_tv_linear.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

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

    # MODIFIED: Moved granularity selector to its own row with explanation
    st.markdown("### Configuração de Granularidade")
    
    # Full width for granularity selector with explanation
    granularity_options = {
        "Diário": df_daily,
        "Semanal": df_weekly,
    }
    
    granularity = st.selectbox(
        "Selecione a granularidade:",
        options=list(granularity_options.keys()),
        help="Recomenda-se usar o valor 'Diário' pois a aplicação atualmente contempla somente os dados de 2024 e 53 semanas é muito pouco para validar dados."
    )
    
    st.markdown("""
    **Nota:** Recomenda-se utilizar a granularidade **Diária** para análises, pois a aplicação atualmente contempla 
    somente os dados de 2024, e 53 semanas é uma quantidade muito pequena para validações estatísticas robustas.
    """)
    
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
    
    
    # Get the selected dataframe
    selected_df = granularity_options[granularity]
    
    # MODIFIED: Keep metric and visualization selectors in two columns
    col1, col2 = st.columns(2)
    
    # Metric selection in first column
    metrics_options = {
        "cov% (cobertura)": "cov%",
        "shr% (share)": "shr%",
        "TVR% (rating)": "tvr%"
    }
    
    with col1:
        selected_metric = st.selectbox(
            "Selecione a métrica:",
            options=list(metrics_options.keys())
        )
    
    metric_suffix = metrics_options[selected_metric]
    
    # Competitor selection in second column
    with col2:
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ADDED: Correlation cards for aggregated competitors - Using the same detailed approach as for specific competitors
                corr_value = plot_data['Globo'].corr(plot_data['Concorrentes (Média)'])
                
                st.subheader(f"Correlação: Globo vs Concorrentes ({selected_metric})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Correlação Globo vs Concorrentes",
                        f"{corr_value:.2f}",
                        delta=None,
                        delta_color="normal",
                        help="Uma correlação próxima de 1 indica que Globo e concorrentes tendem a variar juntos, enquanto valores próximos de -1 indicam variação em direções opostas."
                    )
                
                with col2:
                    # Calculate market share
                    globo_avg = plot_data['Globo'].mean()
                    competitors_avg = plot_data['Concorrentes (Média)'].mean()
                    total_avg = globo_avg + competitors_avg
                    
                    if total_avg > 0:
                        globo_share = (globo_avg / total_avg) * 100
                        st.metric(
                            "Participação Média vs Concorrentes",
                            f"{globo_share:.1f}%",
                            delta=None,
                            delta_color="normal",
                            help="Percentual médio da Globo em relação ao total (Globo + Concorrentes)."
                        )
                    else:
                        st.warning("Dados insuficientes para calcular participação média.")
                
                # Add additional correlation analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    corr_interpretation = ""
                    if corr_value > 0.7:
                        corr_interpretation = "**Correlação forte positiva**: Globo e concorrentes tendem a ter variações muito semelhantes na audiência, sugerindo que todos são afetados pelos mesmos fatores externos (como feriados, eventos especiais, etc)."
                    elif corr_value > 0.3:
                        corr_interpretation = "**Correlação moderada positiva**: Existe alguma tendência de variação similar entre Globo e concorrentes, mas cada um também tem seus próprios padrões distintos."
                    elif corr_value > -0.3:
                        corr_interpretation = "**Correlação fraca**: Globo e concorrentes variam independentemente, sugerindo que atendem a públicos diferentes ou que suas estratégias de programação têm efeitos distintos."
                    elif corr_value > -0.7:
                        corr_interpretation = "**Correlação moderada negativa**: Quando a audiência da Globo sobe, a dos concorrentes tende a cair moderadamente, e vice-versa, sugerindo algum nível de competição direta pela mesma audiência."
                    else:
                        corr_interpretation = "**Correlação forte negativa**: Forte competição direta entre Globo e concorrentes. Quando Globo ganha audiência, os concorrentes perdem, sugerindo grande sobreposição de público-alvo."
                    
                    st.markdown(corr_interpretation)
                
                with col2:
                    # Calculate additional statistics
                    ratio = globo_avg / competitors_avg if competitors_avg > 0 else float('inf')
                    
                    if not np.isinf(ratio):
                        st.markdown(f"""
                        **Proporção média Globo/Concorrentes**: {ratio:.2f}x
                        
                        Isso significa que, em média, a métrica {selected_metric} da Globo 
                        é {ratio:.2f} vezes maior que a média dos concorrentes.
                        """)
                    else:
                        st.warning("Não foi possível calcular a proporção com os concorrentes.")
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
                
                st.plotly_chart(fig, use_container_width=True)
                
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                # ADDED: Correlation cards for specific competitor
                corr_value = plot_data['Globo'].corr(plot_data[selected_competitor])
                
                st.subheader(f"Correlação: Globo vs {selected_competitor} ({selected_metric})")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        f"Correlação Globo vs {selected_competitor}",
                        f"{corr_value:.2f}",
                        delta=None,
                        delta_color="normal",
                        help="Uma correlação próxima de 1 indica que Globo e o concorrente tendem a variar juntos, enquanto valores próximos de -1 indicam variação em direções opostas."
                    )
                
                with col2:
                    # Calculate market share
                    globo_avg = plot_data['Globo'].mean()
                    competitor_avg = plot_data[selected_competitor].mean()
                    total_avg = globo_avg + competitor_avg
                    
                    if total_avg > 0:
                        globo_share = (globo_avg / total_avg) * 100
                        st.metric(
                            f"Participação Média vs {selected_competitor}",
                            f"{globo_share:.1f}%",
                            delta=None,
                            delta_color="normal",
                            help=f"Percentual médio da Globo em relação ao total (Globo + {selected_competitor})."
                        )
                    else:
                        st.warning("Dados insuficientes para calcular participação média.")
                
                # Add additional correlation analysis
                col1, col2 = st.columns(2)
                
                with col1:
                    corr_interpretation = ""
                    if corr_value > 0.7:
                        corr_interpretation = f"**Correlação forte positiva**: Globo e {selected_competitor} tendem a ter variações muito semelhantes na audiência, sugerindo que ambos são afetados pelos mesmos fatores externos (como feriados, eventos especiais, etc)."
                    elif corr_value > 0.3:
                        corr_interpretation = f"**Correlação moderada positiva**: Existe alguma tendência de variação similar entre Globo e {selected_competitor}, mas cada um também tem seus próprios padrões distintos."
                    elif corr_value > -0.3:
                        corr_interpretation = f"**Correlação fraca**: Globo e {selected_competitor} variam independentemente, sugerindo que atendem a públicos diferentes ou que suas estratégias de programação têm efeitos distintos."
                    elif corr_value > -0.7:
                        corr_interpretation = f"**Correlação moderada negativa**: Quando a audiência da Globo sobe, a de {selected_competitor} tende a cair moderadamente, e vice-versa, sugerindo algum nível de competição direta pela mesma audiência."
                    else:
                        corr_interpretation = f"**Correlação forte negativa**: Forte competição direta entre Globo e {selected_competitor}. Quando um ganha audiência, o outro perde, sugerindo grande sobreposição de público-alvo."
                    
                    st.markdown(corr_interpretation)
                
                with col2:
                    # Calculate additional statistics
                    ratio = globo_avg / competitor_avg if competitor_avg > 0 else float('inf')
                    
                    if not np.isinf(ratio):
                        st.markdown(f"""
                        **Proporção média Globo/{selected_competitor}**: {ratio:.2f}x
                        
                        Isso significa que, em média, a métrica {selected_metric} da Globo 
                        é {ratio:.2f} vezes maior que a do {selected_competitor}.
                        """)
                    else:
                        st.warning(f"Não foi possível calcular a proporção com {selected_competitor}.")
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
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.warning(f"Não foram encontrados dados para o concorrente {selected_competitor}.")
        
        # Update layout
        fig.update_layout(
            plot_bgcolor='#F5F5F5',
            font=dict(family="Roboto, Arial", color="#212121"),
            legend=dict(orientation="h", y=1.1),
            margin=dict(t=50, l=50, r=20, b=50),
            hovermode="x unified"
        )
        
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