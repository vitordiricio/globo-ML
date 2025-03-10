import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

def analise_externos(df):
    """
        Analyzes external factors' impact on TV Linear data, including economic indicators,
        specific events, and social metrics."
        Args:
        df (DataFrame): Processed dataframe with EXTERNO_ and LINEAR_ prefixed columns
    """

    st.header("📊 Fatores Externos - Impacto na Audiência")

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

    # Use original dataframe for hourly analysis
    df_hourly = df.copy()

    # Create daily aggregation
    df_daily = df.copy()
    df_daily['data'] = df_daily['data_hora'].dt.date

    # Filter numeric columns for aggregation (to avoid categorical type error)
    numeric_cols = df_daily.select_dtypes(include=['number']).columns.tolist()
    if 'data' in numeric_cols:
        numeric_cols.remove('data')

    # Group by date for numeric columns only
    df_daily = df_daily.groupby('data')[numeric_cols].mean().reset_index()
    df_daily['data_hora'] = pd.to_datetime(df_daily['data'])

    # Create weekly aggregation
    df_weekly = df.copy()
    df_weekly['semana'] = df_weekly['data_hora'].dt.to_period('W').astype(str)

    # Group by week for numeric columns only
    df_weekly = df_weekly.groupby('semana')[numeric_cols].mean().reset_index()
    df_weekly['data_hora'] = pd.to_datetime(df_weekly['semana'].str.split('/').str[0])

    # 2. Granularity Selection
    st.markdown("""
    ### Análise de Fatores Externos

    Esta análise explora como fatores externos impactam a audiência da TV Linear da Globo. 
    Investigamos três categorias principais:

    1. **Indicadores Econômicos**: Como inflação, desemprego e outros índices econômicos se correlacionam com o comportamento da audiência
    2. **Eventos Específicos**: O impacto de eventos como competições esportivas, lançamentos de programas concorrentes, etc.
    3. **Volume Social**: Como o volume de conversas nas redes sociais se relaciona com a audiência TV

    Estas análises ajudam a contextualizar o desempenho da TV Linear dentro do ambiente mais amplo em que opera.

    **Dica**: A análise por hora é especialmente útil para entender o impacto de eventos específicos que ocorrem em horários determinados, como jogos de futebol ou programas de TV concorrentes.
    """)

    granularity = st.selectbox(
        "Selecione a granularidade:",
        options=["Horário", "Diário", "Semanal"],
        index=0  # Default to "Horário"
    )

    # Get the selected dataframe
    if granularity == "Diário":
        selected_df = df_daily
    elif granularity == "Semanal":
        selected_df = df_weekly
    else:  # "Horário"
        selected_df = df_hourly

    # Check which external data columns are available
    economic_cols = [col for col in selected_df.columns if col in [
        'EXTERNO_dolar', 'EXTERNO_unemployment_rate', 'EXTERNO_inflation_ipca', 
        'EXTERNO_selic_rate', 'EXTERNO_indice_cond_economicas'
    ]]

    event_cols = [col for col in selected_df.columns if col in [
        'EXTERNO_FUTEBOL_CONCORRENTE_ON', 'EXTERNO_OLIMPIADAS_24',
        'EXTERNO_SBT_LANCA_PROGRAMA_VIRGINIA', 'EXTERNO_NOTICIA_MORTE_SILVIO_SANTOS',
        'EXTERNO_NETFLIX_LUTA_LOGAN_MIKE'
    ]]

    social_cols = [col for col in selected_df.columns if col in [
        'EXTERNO_quantidade_tweets'
    ]]

    tv_col = 'LINEAR_GLOBO_cov%' if 'LINEAR_GLOBO_cov%' in selected_df.columns else None

    # Calculate overall correlation metrics for each category
    st.subheader("Índice de Correlação por Categoria")

    st.markdown("""
    Os índices abaixo mostram a correlação média de cada categoria de fatores externos com a audiência da TV Linear (cov%).
    Valores mais próximos de 1.0 indicam maior poder explicativo sobre a audiência.
    """)

    col1, col2, col3 = st.columns(3)

    # Economic indicators correlation
    eco_corr = 0
    if economic_cols and tv_col:
        # Filter data where both economic indicators and tv_col have valid values
        valid_eco_data = selected_df.dropna(subset=economic_cols + [tv_col])
        
        if not valid_eco_data.empty:
            eco_corrs = []
            for col in economic_cols:
                # Check if column has any non-missing values
                if not valid_eco_data[col].isna().all():
                    corr = abs(valid_eco_data[col].corr(valid_eco_data[tv_col]))
                    if not pd.isna(corr):  # Only include valid correlations
                        eco_corrs.append(corr)
            
            if eco_corrs:
                eco_corr = sum(eco_corrs) / len(eco_corrs)

    with col1:
        st.metric(
            "Indicadores Econômicos",
            f"{eco_corr:.2f}",
            help="Correlação média dos indicadores econômicos com a audiência (cov%)"
        )

    # Events correlation
    event_corr = 0
    if event_cols and tv_col:
        # Filter data where both events and tv_col have valid values
        valid_events_data = selected_df.dropna(subset=event_cols + [tv_col])
        
        if not valid_events_data.empty:
            event_corrs = []
            for col in event_cols:
                # Check if column has any variation (not all zeros or ones)
                if valid_events_data[col].std() > 0:
                    corr = abs(valid_events_data[col].corr(valid_events_data[tv_col]))
                    if not pd.isna(corr):  # Only include valid correlations
                        event_corrs.append(corr)
            
            if event_corrs:
                event_corr = sum(event_corrs) / len(event_corrs)

    with col2:
        st.metric(
            "Eventos",
            f"{event_corr:.2f}",
            help="Correlação média dos eventos externos com a audiência (cov%)"
        )

    # Social volume correlation
    social_corr = 0
    if 'EXTERNO_quantidade_tweets' in selected_df.columns and tv_col:
        # Filter data where both tweets and tv_col have valid values
        valid_social_data = selected_df.dropna(subset=['EXTERNO_quantidade_tweets', tv_col])
        
        if not valid_social_data.empty and valid_social_data['EXTERNO_quantidade_tweets'].std() > 0:
            social_corr = abs(valid_social_data['EXTERNO_quantidade_tweets'].corr(valid_social_data[tv_col]))
            if pd.isna(social_corr):  # Handle NaN correlation
                social_corr = 0

    with col3:
        st.metric(
            "Volume Social",
            f"{social_corr:.2f}",
            help="Correlação do volume de tweets com a audiência (cov%)"
        )

    # Create tabs for different categories of analysis
    tabs = st.tabs(["Indicadores Econômicos", "Eventos", "Volume Social"])

    # 3. Economic Indicators Analysis Tab
    with tabs[0]:
        st.subheader("Análise de Indicadores Econômicos")
        
        st.markdown("""
        Os indicadores econômicos podem influenciar significativamente os hábitos de consumo de mídia.
        Por exemplo, em períodos de maior desemprego, as pessoas podem passar mais tempo em casa assistindo TV,
        enquanto inflação alta pode levar à redução de gastos com entretenimento pago.
        
        O gráfico abaixo mostra a evolução dos principais indicadores econômicos normalizados comparados
        com a evolução da audiência TV Linear da Globo.
        """)
        
        if economic_cols and tv_col:
            # Filter data where both economic indicators and tv_col have valid values
            valid_eco_data = selected_df.dropna(subset=economic_cols + [tv_col])
            
            if not valid_eco_data.empty:
                # Create time series chart comparing economic indicators with TV audience
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add TV audience line
                fig.add_trace(
                    go.Scatter(
                        x=valid_eco_data['data_hora'],
                        y=valid_eco_data[tv_col],
                        name='Audiência TV (cov%)',
                        line=dict(color='rgb(31, 119, 180)', width=3)
                    ),
                    secondary_y=False
                )
                
                # Add economic indicators (normalized)
                colors = ['rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 
                        'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
                
                for i, col in enumerate(economic_cols):
                    # Only include columns with valid data
                    if not valid_eco_data[col].isna().all():
                        # Normalize the indicator for better visualization
                        min_val = valid_eco_data[col].min()
                        max_val = valid_eco_data[col].max()
                        if max_val > min_val:  # Avoid division by zero
                            normalized = (valid_eco_data[col] - min_val) / (max_val - min_val) * valid_eco_data[tv_col].max()
                            
                            # Clean up column name for display
                            display_name = col.replace('EXTERNO_', '').replace('_', ' ').title()
                            
                            fig.add_trace(
                                go.Scatter(
                                    x=valid_eco_data['data_hora'],
                                    y=normalized,
                                    name=display_name,
                                    line=dict(color=colors[i % len(colors)], dash='dash')
                                ),
                                secondary_y=True
                            )
                
                # Update layout
                fig.update_layout(
                    title=f'Evolução da Audiência TV vs. Indicadores Econômicos ({granularity})',
                    xaxis_title='Data',
                    yaxis_title='Audiência TV (cov%)',
                    yaxis2_title='Indicadores (Normalizados)',
                    legend_title='Métricas',
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create correlation matrix between economic indicators and audience metrics
                st.markdown("### Correlação entre Indicadores Econômicos e Métricas de Audiência")
                
                st.markdown("""
                A tabela abaixo mostra a correlação entre diferentes indicadores econômicos e métricas de audiência TV.
                Uma correlação positiva (próxima de 1) indica que as variáveis tendem a aumentar juntas,
                enquanto uma correlação negativa (próxima de -1) indica uma relação inversa.
                Valores próximos de zero indicam pouca ou nenhuma relação linear.
                """)
                
                # Select audience metrics
                audience_metrics = [col for col in valid_eco_data.columns if col.startswith('LINEAR_GLOBO_') 
                                and col in ['LINEAR_GLOBO_cov%', 'LINEAR_GLOBO_shr%', 'LINEAR_GLOBO_tvr%']]
                
                if audience_metrics:
                    # Calculate correlation matrix
                    corr_matrix = valid_eco_data[economic_cols + audience_metrics].corr()
                    
                    # Extract correlations between economic indicators and audience metrics
                    eco_audience_corr = corr_matrix.loc[economic_cols, audience_metrics]
                    
                    # Create a clean correlation matrix for display
                    clean_rows = [col.replace('EXTERNO_', '').replace('_', ' ').title() for col in eco_audience_corr.index]
                    clean_cols = [col.replace('LINEAR_GLOBO_', '').replace('%', ' %') for col in eco_audience_corr.columns]
                    
                    clean_corr = pd.DataFrame(
                        eco_audience_corr.values,
                        index=clean_rows,
                        columns=clean_cols
                    )
                    
                    # Create heatmap
                    fig_heatmap = px.imshow(
                        clean_corr,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale=["red", "white", "green"],
                        labels=dict(x="Métrica de Audiência", y="Indicador Econômico", color="Correlação"),
                        title="Correlação entre Indicadores Econômicos e Métricas de Audiência"
                    )
                    
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Generate insights based on correlations
                    st.markdown("### Insights Principais")
                    
                    # Find strongest positive and negative correlations
                    eco_audience_flat = eco_audience_corr.unstack()
                    strongest_positive = eco_audience_flat.nlargest(1)
                    strongest_negative = eco_audience_flat.nsmallest(1)
                    
                    # Display insights
                    if not strongest_positive.empty:
                        pos_val = strongest_positive.values[0]
                        pos_idx = strongest_positive.index[0]
                        pos_eco = pos_idx[0].replace('EXTERNO_', '').replace('_', ' ').title()
                        pos_aud = pos_idx[1].replace('LINEAR_GLOBO_', '').replace('%', ' %')
                        
                        if pos_val > 0.3:
                            st.success(f"**Relação Positiva Forte:** {pos_eco} tem correlação positiva de {pos_val:.2f} com {pos_aud}, "
                                    f"sugerindo que aumentos neste indicador estão associados a maiores níveis de audiência.")
                    
                    if not strongest_negative.empty:
                        neg_val = strongest_negative.values[0]
                        neg_idx = strongest_negative.index[0]
                        neg_eco = neg_idx[0].replace('EXTERNO_', '').replace('_', ' ').title()
                        neg_aud = neg_idx[1].replace('LINEAR_GLOBO_', '').replace('%', ' %')
                        
                        if neg_val < -0.3:
                            st.error(f"**Relação Negativa Forte:** {neg_eco} tem correlação negativa de {neg_val:.2f} com {neg_aud}, "
                                f"sugerindo que aumentos neste indicador estão associados a menores níveis de audiência.")
                    
                    # Additional insights based on specific economic indicators
                    if 'EXTERNO_unemployment_rate' in economic_cols and 'LINEAR_GLOBO_cov%' in audience_metrics:
                        unemp_corr = corr_matrix.loc['EXTERNO_unemployment_rate', 'LINEAR_GLOBO_cov%']
                        
                        if unemp_corr > 0.3:
                            st.info(f"**Desemprego e Audiência:** A correlação de {unemp_corr:.2f} entre desemprego e audiência "
                                f"sugere que em períodos de maior desemprego há maior consumo de TV Linear, possivelmente "
                                f"devido a mais pessoas em casa e/ou a busca por entretenimento de menor custo.")
                        elif unemp_corr < -0.3:
                            st.info(f"**Desemprego e Audiência:** A correlação de {unemp_corr:.2f} entre desemprego e audiência "
                                f"sugere que em períodos de maior desemprego há menor consumo de TV Linear, possivelmente "
                                f"indicando mudança para alternativas de entretenimento mais econômicas.")
                    
                    if 'EXTERNO_inflation_ipca' in economic_cols and 'LINEAR_GLOBO_cov%' in audience_metrics:
                        infl_corr = corr_matrix.loc['EXTERNO_inflation_ipca', 'LINEAR_GLOBO_cov%']
                        
                        if abs(infl_corr) > 0.3:
                            dir_word = "maior" if infl_corr > 0 else "menor"
                            st.info(f"**Inflação e Audiência:** A correlação de {infl_corr:.2f} entre inflação (IPCA) e audiência "
                                f"sugere que períodos de inflação mais alta estão associados a {dir_word} consumo de TV Linear.")
            else:
                st.warning("Dados insuficientes para análise de indicadores econômicos.")
        else:
            st.warning("Dados econômicos ou de audiência não estão disponíveis.")

    # 4. Events Analysis Tab
    with tabs[1]:
        st.subheader("Análise de Eventos")
        
        # Different explanations based on granularity
        if granularity == "Horário":
            st.markdown("""
            Eventos específicos como competições esportivas, programas concorrentes, ou acontecimentos relevantes 
            podem impactar significativamente a audiência de TV em horários específicos. Esta análise quantifica 
            o impacto desses eventos hora a hora, permitindo entender como afetam a audiência precisamente nos
            momentos em que ocorrem.
            
            O gráfico abaixo mostra a evolução horária da audiência com marcadores nos momentos em que ocorreram eventos específicos.
            """)
        else:
            st.markdown("""
            Eventos específicos como competições esportivas, programas concorrentes, ou acontecimentos relevantes 
            podem impactar significativamente a audiência de TV. Esta análise quantifica o impacto desses eventos,
            permitindo entender melhor como o contexto externo influencia os resultados de audiência.
            
            O gráfico abaixo mostra a evolução da audiência com marcadores nos dias em que ocorreram eventos específicos.
            """)
        
        if event_cols and tv_col:
            # Filter data where at least one event and tv_col have valid values
            valid_events = []
            for col in event_cols:
                if selected_df[col].std() > 0:  # Check if column has variation
                    valid_events.append(col)
            
            if valid_events:
                # Create a dataframe for plotting with valid events only
                plot_df = selected_df[['data_hora', tv_col] + valid_events].dropna().copy()
                
                if not plot_df.empty:
                    # Convert event columns to descriptive names
                    for event_col in valid_events:
                        event_name = event_col.replace('EXTERNO_', '').replace('_', ' ').title()
                        plot_df[event_name] = plot_df[event_col]
                    
                    # Create timeline with event markers
                    fig = go.Figure()
                    
                    # Add TV audience line
                    fig.add_trace(
                        go.Scatter(
                            x=plot_df['data_hora'],
                            y=plot_df[tv_col],
                            name='Audiência TV (cov%)',
                            line=dict(color='rgb(31, 119, 180)', width=2)
                        )
                    )
                    
                    # Add markers for each event
                    colors = ['rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 
                            'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
                    
                    for i, event_col in enumerate(valid_events):
                        event_name = event_col.replace('EXTERNO_', '').replace('_', ' ').title()
                        
                        # Filter for days when the event occurred
                        event_days = plot_df[plot_df[event_col] > 0]
                        
                        if not event_days.empty:
                            fig.add_trace(
                                go.Scatter(
                                    x=event_days['data_hora'],
                                    y=event_days[tv_col],
                                    mode='markers',
                                    name=event_name,
                                    marker=dict(
                                        size=10,
                                        color=colors[i % len(colors)],
                                        symbol='diamond'
                                    )
                                )
                            )
                    
                    # Update layout
                    time_unit = "horário" if granularity == "Horário" else "diário" if granularity == "Diário" else "semanal"
                    fig.update_layout(
                        title=f'Evolução {time_unit} da Audiência TV com Marcadores de Eventos ({granularity})',
                        xaxis_title='Data e Hora' if granularity == "Horário" else 'Data',
                        yaxis_title='Audiência TV (cov%)',
                        legend_title='Eventos',
                        hovermode="closest"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Create impact analysis table
                    st.markdown("### Impacto dos Eventos na Audiência TV")
                    
                    if granularity == "Horário":
                        st.markdown("""
                        A tabela abaixo quantifica o impacto de diferentes tipos de eventos na audiência de TV por hora.
                        Para cada evento, calculamos a diferença percentual média na audiência durante as horas em que o evento
                        ocorreu comparado com horas sem o evento. Isso permite entender o impacto preciso no momento exato de ocorrência.
                        """)
                    else:
                        st.markdown("""
                        A tabela abaixo quantifica o impacto de diferentes tipos de eventos na audiência de TV.
                        Para cada evento, calculamos a diferença percentual média na audiência durante o evento
                        comparado com períodos sem o evento.
                        """)
                    
                    # Calculate impact for each event
                    impact_data = []
                    
                    for event_col in valid_events:
                        event_name = event_col.replace('EXTERNO_', '').replace('_', ' ').title()
                        
                        # Calculate audience with and without event
                        event_on = plot_df[plot_df[event_col] > 0][tv_col].mean()
                        event_off = plot_df[plot_df[event_col] == 0][tv_col].mean()
                        
                        # Calculate impact percentage
                        if event_off > 0:  # Avoid division by zero
                            impact_pct = ((event_on / event_off) - 1) * 100
                            
                            # Count occurrences
                            occurrences = plot_df[plot_df[event_col] > 0].shape[0]
                            
                            impact_data.append({
                                "Evento": event_name,
                                "Impacto na Audiência (%)": f"{impact_pct:.2f}%",
                                "Direção": "Positivo" if impact_pct > 0 else "Negativo",
                                "Ocorrências": occurrences
                            })
                    
                    if impact_data:
                        # Convert to DataFrame
                        impact_df = pd.DataFrame(impact_data)
                        
                        # Display as table
                        st.dataframe(impact_df, hide_index=True, use_container_width=True)
                        
                        # Generate insights based on impact
                        st.markdown("### Insights sobre Eventos")
                        
                        # Find events with strongest positive and negative impacts
                        pos_events = [event for event in impact_data if "Positivo" in event["Direção"]]
                        neg_events = [event for event in impact_data if "Negativo" in event["Direção"]]
                        
                        # Sort by impact magnitude
                        pos_events.sort(key=lambda x: float(x["Impacto na Audiência (%)"].replace("%", "")), reverse=True)
                        neg_events.sort(key=lambda x: float(x["Impacto na Audiência (%)"].replace("%", "")))
                        
                        if pos_events:
                            top_pos = pos_events[0]
                            pos_impact = float(top_pos["Impacto na Audiência (%)"].replace("%", ""))
                            
                            if pos_impact > 5:  # Only show if impact is significant
                                time_text = "horas" if granularity == "Horário" else "dias" if granularity == "Diário" else "semanas"
                                st.success(f"**Evento com Maior Impacto Positivo:** {top_pos['Evento']} aumenta a audiência "
                                        f"em {top_pos['Impacto na Audiência (%)']} em média, baseado em {top_pos['Ocorrências']} {time_text}.")
                        
                        if neg_events:
                            top_neg = neg_events[0]
                            neg_impact = float(top_neg["Impacto na Audiência (%)"].replace("%", ""))
                            
                            if neg_impact < -5:  # Only show if impact is significant
                                time_text = "horas" if granularity == "Horário" else "dias" if granularity == "Diário" else "semanas"
                                st.error(f"**Evento com Maior Impacto Negativo:** {top_neg['Evento']} reduz a audiência "
                                    f"em {top_neg['Impacto na Audiência (%)']} em média, baseado em {top_neg['Ocorrências']} {time_text}.")
                        
                        # Additional analysis for specific event types
                        football_events = [event for event in impact_data if "Futebol" in event["Evento"]]
                        if football_events:
                            football = football_events[0]
                            impact = float(football["Impacto na Audiência (%)"].replace("%", ""))
                            
                            if abs(impact) > 5:
                                direction = "aumenta" if impact > 0 else "reduz"
                                
                                # Add hourly specific insight
                                if granularity == "Horário":
                                    st.info(f"**Impacto de Eventos Esportivos:** Futebol {direction} a audiência em "
                                        f"{abs(impact):.2f}% durante as horas em que ocorre, sugerindo que as transmissões esportivas "
                                        f"{'atraem' if impact > 0 else 'competem pela'} audiência da TV Linear no momento exato "
                                        f"em que acontecem.")
                                else:
                                    st.info(f"**Impacto de Eventos Esportivos:** Futebol {direction} a audiência em "
                                        f"{abs(impact):.2f}%, sugerindo que a programação esportiva {'atrai' if impact > 0 else 'compete pela'} "
                                        f"audiência da TV Linear.")
                    else:
                        st.warning("Não foi possível calcular o impacto dos eventos na audiência.")
                else:
                    st.warning("Dados insuficientes para análise de eventos.")
            else:
                st.warning("Não foram encontrados eventos válidos no período analisado.")
        else:
            st.warning("Dados de eventos ou de audiência não estão disponíveis.")

    # 5. Social Volume Analysis Tab
    with tabs[2]:
        st.subheader("Análise de Volume Social")
        
        if granularity == "Horário":
            st.markdown("""
            O volume de conversas nas redes sociais pode variar significativamente ao longo do dia e impactar a audiência TV hora a hora.
            Programas que geram conversação intensa em tempo real podem atrair espectadores adicionais, enquanto assuntos
            virais podem desviar a atenção da TV em horários específicos.
            
            Esta análise examina a relação entre o volume horário de tweets e métricas de audiência da TV Linear.
            """)
        else:
            st.markdown("""
            O volume de conversas nas redes sociais pode ser tanto um indicador como um driver da audiência de TV.
            Programas que geram mais conversação podem atrair novos espectadores, enquanto assuntos muito discutidos
            nas redes sociais podem desviar a atenção da TV.
            
            Esta análise examina a relação entre o volume de tweets e métricas de audiência da TV Linear.
            """)
        
        # Check if we have tweet data
        if 'EXTERNO_quantidade_tweets' in selected_df.columns and tv_col:
            # Filter data where both tweets and tv_col have valid values
            valid_social_data = selected_df.dropna(subset=['EXTERNO_quantidade_tweets', tv_col])
            
            # Also filter for non-zero tweet values
            valid_social_data = valid_social_data[valid_social_data['EXTERNO_quantidade_tweets'] > 0]
            
            if not valid_social_data.empty:
                # Create time series chart comparing tweet volume with TV audience
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add TV audience line
                fig.add_trace(
                    go.Scatter(
                        x=valid_social_data['data_hora'],
                        y=valid_social_data[tv_col],
                        name='Audiência TV (cov%)',
                        line=dict(color='rgb(31, 119, 180)', width=3)
                    ),
                    secondary_y=False
                )
                
                # Add tweet volume
                fig.add_trace(
                    go.Scatter(
                        x=valid_social_data['data_hora'],
                        y=valid_social_data['EXTERNO_quantidade_tweets'],
                        name='Volume de Tweets',
                        line=dict(color='rgb(255, 127, 14)', width=2)
                    ),
                    secondary_y=True
                )
                
                # Update layout
                time_unit = "" if granularity == "Horário" else "Diária" if granularity == "Diário" else "Semanal"
                fig.update_layout(
                    title=f'Evolução {time_unit} da Audiência TV vs. Volume de Tweets ({granularity})',
                    xaxis_title='Data e Hora' if granularity == "Horário" else 'Data',
                    yaxis_title='Audiência TV (cov%)',
                    yaxis2_title='Quantidade de Tweets',
                    legend_title='Métricas',
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create scatter plot for correlation analysis
                st.markdown("### Correlação entre Volume de Tweets e Audiência TV")
                
                fig_scatter = px.scatter(
                    valid_social_data,
                    x='EXTERNO_quantidade_tweets',
                    y=tv_col,
                    trendline="ols",
                    labels={
                        'EXTERNO_quantidade_tweets': 'Quantidade de Tweets',
                        tv_col: 'Audiência TV (cov%)'
                    },
                    title='Relação entre Volume de Tweets e Audiência TV'
                )
                
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Calculate correlation and regression
                tweets_corr = valid_social_data['EXTERNO_quantidade_tweets'].corr(valid_social_data[tv_col])
                
                # Create lag analysis if not hourly (no lag analysis for hourly makes less sense)
                if granularity != "Horário":
                    st.markdown("### Análise de Lag: Tweets vs. Audiência")
                    
                    st.markdown("""
                    Esta análise verifica se um aumento no volume de tweets precede ou sucede 
                    um aumento na audiência TV. Um lag positivo significativo sugere que a conversação
                    social pode funcionar como um preditor da audiência futura.
                    """)
                    
                    # Create lag columns
                    lags = [1, 2, 3, 7]  # Look at 1, 2, 3, and 7 days lag
                    lag_corrs = []
                    
                    for lag in lags:
                        lag_col = f'tweets_lag_{lag}'
                        valid_social_data[lag_col] = valid_social_data['EXTERNO_quantidade_tweets'].shift(lag)
                        
                        # Calculate correlation with audience, excluding NaN values
                        lag_data = valid_social_data.dropna(subset=[lag_col, tv_col])
                        
                        if not lag_data.empty:
                            corr = lag_data[lag_col].corr(lag_data[tv_col])
                            if not pd.isna(corr):  # Only include valid correlations
                                lag_corrs.append({
                                    'Lag (dias)': lag,
                                    'Correlação': corr
                                })
                    
                    if lag_corrs:
                        # Convert to DataFrame
                        lag_df = pd.DataFrame(lag_corrs)
                        
                        # Create bar chart
                        fig_lag = px.bar(
                            lag_df,
                            x='Lag (dias)',
                            y='Correlação',
                            title='Correlação entre Volume de Tweets (com lag) e Audiência TV',
                            labels={
                                'Lag (dias)': 'Tweets Defasados (dias)',
                                'Correlação': 'Correlação com Audiência TV'
                            },
                            color='Correlação',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        
                        st.plotly_chart(fig_lag, use_container_width=True)
                else:
                    # For hourly analysis, provide insight about real-time social engagement
                    st.markdown("### Análise de Engajamento Social em Tempo Real")
                    
                    st.markdown("""
                    Para dados por hora, a análise de correlação em tempo real é mais relevante que a análise de lag.
                    A correlação atual mostra se as pessoas estão engajadas nas redes sociais durante o mesmo período
                    em que estão assistindo TV, o que pode indicar o uso de "segunda tela" ou conversação social
                    sobre os programas em andamento.
                    """)
                
                # Generate insights based on correlation
                st.markdown("### Insights sobre Volume Social")
                
                # Direct correlation insight
                if not pd.isna(tweets_corr):
                    if abs(tweets_corr) > 0.3:
                        direction = "positiva" if tweets_corr > 0 else "negativa"
                        relation = "aumentam juntos" if tweets_corr > 0 else "têm relação inversa"
                        
                        if granularity == "Horário":
                            st.info(f"**Correlação {direction}:** A correlação de {tweets_corr:.2f} entre volume de tweets e audiência por hora "
                                f"sugere que estas métricas {relation}. Isso pode indicar que {'os espectadores estão usando as redes sociais como segunda tela durante os programas' if tweets_corr > 0 else 'maior atividade nas redes sociais pode desviar atenção da TV em horários específicos'}.")
                        else:
                            st.info(f"**Correlação {direction}:** A correlação de {tweets_corr:.2f} entre volume de tweets e audiência "
                                f"sugere que estas métricas {relation}. Isso pode indicar que {'programas mais assistidos geram mais conversação social' if tweets_corr > 0 else 'maior atividade nas redes sociais pode desviar atenção da TV'}.")
                
                # If we have lag analysis, show those insights
                if granularity != "Horário" and 'lag_corrs' in locals() and lag_corrs:
                    max_lag_corr = max(lag_corrs, key=lambda x: abs(x['Correlação']))
                    
                    if abs(max_lag_corr['Correlação']) > 0.3:
                        direction = "positiva" if max_lag_corr['Correlação'] > 0 else "negativa"
                        effect = "preditor" if max_lag_corr['Correlação'] > 0 else "indicador de possível queda"
                        
                        st.success(f"**Efeito Temporal:** A correlação {direction} mais forte ({max_lag_corr['Correlação']:.2f}) "
                                f"ocorre com {max_lag_corr['Lag (dias)']} dia(s) de defasagem, sugerindo que o volume de tweets "
                                f"pode ser um {effect} da audiência TV futura.")
                
                # Hourly specific insight
                if granularity == "Horário":
                    # Analyze correlation during prime time vs other times
                    prime_time_hours = list(range(19, 23))  # 7pm to 10pm
                    
                    prime_df = valid_social_data[valid_social_data['data_hora'].dt.hour.isin(prime_time_hours)]
                    non_prime_df = valid_social_data[~valid_social_data['data_hora'].dt.hour.isin(prime_time_hours)]
                    
                    if not prime_df.empty and not non_prime_df.empty:
                        prime_corr = prime_df['EXTERNO_quantidade_tweets'].corr(prime_df[tv_col])
                        non_prime_corr = non_prime_df['EXTERNO_quantidade_tweets'].corr(non_prime_df[tv_col])
                        
                        if not pd.isna(prime_corr) and not pd.isna(non_prime_corr) and abs(prime_corr - non_prime_corr) > 0.2:
                            stronger = "horário nobre" if abs(prime_corr) > abs(non_prime_corr) else "fora do horário nobre"
                            st.success(f"**Diferença por Horário:** A correlação entre tweets e audiência é mais forte durante o {stronger} "
                                    f"({prime_corr:.2f} vs {non_prime_corr:.2f}), sugerindo que "
                                    f"{'o comportamento de segunda tela é mais prevalente no horário nobre' if stronger == 'horário nobre' else 'o uso de redes sociais durante o dia tem impacto maior na audiência fora do horário nobre'}.")
                
                # Quantification insight
                if not pd.isna(tweets_corr) and abs(tweets_corr) > 0.3:
                    # Fit a simple linear regression model
                    X = sm.add_constant(valid_social_data['EXTERNO_quantidade_tweets'])
                    y = valid_social_data[tv_col]
                    model = sm.OLS(y, X).fit()
                    
                    # Get coefficient to quantify impact
                    coef = model.params[1]
                    
                    if abs(coef) > 0.001:  # Only show if effect size is meaningful
                        direction = "aumento" if coef > 0 else "redução"
                        
                        qty_text = "por hora" if granularity == "Horário" else "por dia" if granularity == "Diário" else "por semana"
                        st.info(f"**Quantificação do Impacto:** A cada 1000 tweets adicionais {qty_text}, observa-se {direction} de "
                            f"{abs(coef*1000):.3f} pontos percentuais na audiência TV.")
            else:
                st.warning("Dados insuficientes para análise de volume social no período selecionado.")
        else:
            st.warning("Dados sobre volume de tweets (EXTERNO_quantidade_tweets) não estão disponíveis.")

    # 8. Final notes - always show
    with st.expander("Informações sobre a análise de fatores externos"):
        st.markdown("""
        ### Fonte dos Dados

        **Indicadores Econômicos**: Os dados econômicos são obtidos de fontes oficiais como o Banco Central do Brasil e o IBGE.

        **Eventos**: Os eventos são identificados e categorizados manualmente com base em um calendário de eventos relevantes.

        **Volume Social**: Os dados de volume de tweets são obtidos via API do Twitter/X, com foco em termos relacionados à mídia e entretenimento.

        ### Considerações Metodológicas

        1. **Correlação não implica causalidade**: Embora identifiquemos correlações entre fatores externos e audiência, isso não necessariamente indica uma relação causal. Outros fatores não observados podem estar influenciando ambas as variáveis.

        2. **Limitações temporais**: A análise considera apenas o período coberto pelos dados disponíveis, que pode não representar todos os ciclos econômicos ou sazonalidades.

        3. **Simplificação de eventos**: Eventos são tratados como variáveis binárias (ocorreram ou não), sem considerar sua intensidade ou duração específica.

        4. **Modelo linear**: O modelo explicativo integrado assume relações lineares entre fatores externos e audiência, o que pode não capturar completamente relações mais complexas.
        
        5. **Granularidade dos dados**: A análise horária permite maior precisão ao examinar o impacto imediato de eventos, enquanto a análise diária ou semanal captura tendências mais amplas.
        
        6. **Comparações válidas**: Para cada análise, consideramos apenas os períodos em que existem dados válidos para todas as variáveis envolvidas, garantindo que as comparações sejam consistentes e representativas.
        """)