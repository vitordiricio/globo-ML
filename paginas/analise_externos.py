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
    specific events, and social metrics.
    
    Args:
        df (DataFrame): Processed dataframe with EXTERNO_* prefixed columns
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
    Investigamos cinco categorias principais:

    1. **Indicadores Econômicos**: Como inflação, desemprego e outros índices econômicos se correlacionam com o comportamento da audiência
    2. **Programas Recorrentes**: Analisamos como programas recorrentes (novelas, reality shows, futebol) influenciam a audiência
    3. **Gêneros de Programação**: Como diferentes gêneros televisivos impactam a audiência
    4. **Eventos Isolados**: O impacto de eventos pontuais como notícias importantes, lançamentos, etc.
    5. **Volume Social**: Como o volume de conversas nas redes sociais se relaciona com a audiência TV

    Estas análises ajudam a contextualizar o desempenho da TV Linear dentro do ambiente competitivo.

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

    # Identify the TV column to use (audience metric)
    tv_col = 'LINEAR_GLOBO_cov%' if 'LINEAR_GLOBO_cov%' in selected_df.columns else None

    if tv_col is None:
        st.error("Não foi possível encontrar a coluna 'LINEAR_GLOBO_cov%' no conjunto de dados.")
        return

    # Create tabs for different categories of analysis
    tabs = st.tabs(["Indicadores Econômicos", "Programas Recorrentes", "Gêneros de Programação", "Eventos Isolados", "Volume Social", "Resumo"])

    #######################################
    # 3. Economic Indicators Analysis Tab #
    #######################################
    with tabs[0]:
        st.subheader("Análise de Indicadores Econômicos")
        
        st.markdown("""
        Os indicadores econômicos podem influenciar significativamente os hábitos de consumo de mídia.
        Por exemplo, em períodos de maior desemprego, as pessoas podem passar mais tempo em casa assistindo TV,
        enquanto inflação alta pode levar à redução de gastos com entretenimento pago.
        
        O gráfico abaixo mostra a evolução dos principais indicadores econômicos normalizados comparados
        com a evolução da audiência TV Linear da Globo.
        """)
        
        # Find economic indicator columns based on new naming pattern
        economic_cols = [col for col in selected_df.columns if col.startswith('EXTERNO_ECONOMICO_')]
        
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
                        name='Audiência TV Globo (cov%)',
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
                            display_name = col.replace('EXTERNO_ECONOMICO_', '').replace('_', ' ').title()
                            
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
                    yaxis_title='Audiência TV Globo (cov%)',
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
                    clean_rows = [col.replace('EXTERNO_ECONOMICO_', '').replace('_', ' ').title() for col in eco_audience_corr.index]
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
                        labels=dict(x="Métrica de Audiência Globo", y="Indicador Econômico", color="Correlação"),
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
                        pos_eco = pos_idx[0].replace('EXTERNO_ECONOMICO_', '').replace('_', ' ').title()
                        pos_aud = pos_idx[1].replace('LINEAR_GLOBO_', '').replace('%', ' %')
                        
                        if pos_val > 0.3:
                            st.success(f"**Relação Positiva Forte:** {pos_eco} tem correlação positiva de {pos_val:.2f} com {pos_aud}, "
                                    f"sugerindo que aumentos neste indicador estão associados a maiores níveis de audiência da Globo.")
                    
                    if not strongest_negative.empty:
                        neg_val = strongest_negative.values[0]
                        neg_idx = strongest_negative.index[0]
                        neg_eco = neg_idx[0].replace('EXTERNO_ECONOMICO_', '').replace('_', ' ').title()
                        neg_aud = neg_idx[1].replace('LINEAR_GLOBO_', '').replace('%', ' %')
                        
                        if neg_val < -0.3:
                            st.error(f"**Relação Negativa Forte:** {neg_eco} tem correlação negativa de {neg_val:.2f} com {pos_aud}, "
                                f"sugerindo que aumentos neste indicador estão associados a menores níveis de audiência da Globo.")
                    
                    # Additional insights based on specific economic indicators
                    unemployment_col = [col for col in economic_cols if 'unemployment' in col.lower()]
                    if unemployment_col and audience_metrics:
                        unemp_col = unemployment_col[0]
                        unemp_corr = corr_matrix.loc[unemp_col, tv_col]
                        
                        if abs(unemp_corr) > 0.3:
                            direction = "positiva" if unemp_corr > 0 else "negativa"
                            impact = "maior" if unemp_corr > 0 else "menor"
                            reason = "mais pessoas em casa e/ou a busca por entretenimento de menor custo" if unemp_corr > 0 else "mudança para alternativas de entretenimento mais econômicas"
                            
                            st.info(f"**Desemprego e Audiência:** A correlação {direction} de {unemp_corr:.2f} entre desemprego e audiência "
                                f"sugere que em períodos de maior desemprego há {impact} consumo de TV Linear Globo, possivelmente "
                                f"devido a {reason}.")
                    
                    inflation_col = [col for col in economic_cols if 'inflation' in col.lower()]
                    if inflation_col and audience_metrics:
                        infl_col = inflation_col[0]
                        infl_corr = corr_matrix.loc[infl_col, tv_col]
                        
                        if abs(infl_corr) > 0.3:
                            dir_word = "maior" if infl_corr > 0 else "menor"
                            st.info(f"**Inflação e Audiência:** A correlação de {infl_corr:.2f} entre inflação (IPCA) e audiência "
                                f"sugere que períodos de inflação mais alta estão associados a {dir_word} consumo de TV Linear Globo.")
            else:
                st.warning("Dados insuficientes para análise de indicadores econômicos.")
        else:
            st.warning("Dados econômicos ou de audiência não estão disponíveis.")

    ########################################
    # 4. Recurring Programs Analysis Tab   #
    ########################################
    with tabs[1]:
        st.subheader("Análise de Programas Recorrentes")
        
        st.markdown("""
        Programas recorrentes como novelas, reality shows e jogos de futebol podem ter impacto significativo
        na audiência da TV Globo, tanto os transmitidos pela própria Globo quanto os exibidos por emissoras concorrentes.
        
        Esta análise permite observar como programas recorrentes de emissoras concorrentes podem "roubar" audiência da TV Globo,
        ou como programas da própria Globo tendem a aumentar sua audiência.
        """)
        
        # Find columns for recurring programs
        recorrente_cols = [col for col in selected_df.columns if col.startswith('EXTERNO_GRADE_RECORRENTE_')]
        
        if recorrente_cols and tv_col:
            # Extract unique program types from column names
            # The program type is in the 4th position when splitting by '_'
            program_types = set()
            for col in recorrente_cols:
                parts = col.split('_')
                if len(parts) >= 5:  # Ensure we have enough parts
                    program_types.add(parts[4])  # Extract program type (4th element)
            
            program_types = sorted(list(program_types))
            
            if program_types:
                # Let user select a program type to analyze
                selected_program = st.selectbox(
                    "Selecione um programa para análise:",
                    options=program_types
                )
                
                # Get all columns related to the selected program
                program_cols = [col for col in recorrente_cols if f"_{selected_program}_" in col]
                
                if program_cols:
                    # Extract broadcaster from each column
                    broadcasters = []
                    for col in program_cols:
                        parts = col.split('_')
                        if len(parts) >= 6:  # Ensure we have enough parts
                            broadcasters.append(parts[3])  # Extract broadcaster (5th element)
                    
                    st.markdown(f"### Impacto do Programa {selected_program} na Audiência da TV Globo")
                    
                    # Calculate impact for each broadcaster
                    impact_data = []
                    
                    for col in program_cols:
                        # Extract broadcaster name
                        parts = col.split('_')
                        broadcaster = parts[3] if len(parts) >= 6 else "Unknown"
                        
                        # Calculate audience with and without program
                        valid_data = selected_df.dropna(subset=[col, tv_col])
                        if valid_data.empty or valid_data[col].sum() == 0:
                            continue
                            
                        program_on = valid_data[valid_data[col] > 0][tv_col].mean()
                        program_off = valid_data[valid_data[col] == 0][tv_col].mean()
                        
                        # Calculate impact percentage
                        if program_off > 0:  # Avoid division by zero
                            impact_pct = ((program_on / program_off) - 1) * 100
                            
                            # Count occurrences
                            occurrences = valid_data[valid_data[col] > 0].shape[0]
                            
                            relation_type = "própria" if broadcaster == "GLOBO" else "concorrente"
                            
                            impact_data.append({
                                "Programa": selected_program,
                                "Emissora": broadcaster,
                                "Tipo": relation_type,
                                "Impacto (%)": impact_pct,
                                "Direção": "Positivo" if impact_pct > 0 else "Negativo",
                                "Ocorrências": occurrences
                            })
                    
                    if impact_data:
                        # Convert to DataFrame and format
                        impact_df = pd.DataFrame(impact_data)
                        
                        # Create a key insight about this program's impact
                        if "GLOBO" in impact_df["Emissora"].values:
                            globo_impact = impact_df[impact_df["Emissora"] == "GLOBO"]["Impacto (%)"].values[0]
                            direction = "aumenta" if globo_impact > 0 else "reduz"
                            
                            st.info(f"**{selected_program} na Globo:** Quando exibido na própria Globo, este programa {direction} "
                                  f"a audiência da emissora em {abs(globo_impact):.2f}% em média.")
                        
                        # Find competitors with biggest impact
                        competitors = impact_df[impact_df["Emissora"] != "GLOBO"]
                        if not competitors.empty:
                            strongest_competitor = competitors.loc[competitors["Impacto (%)"].abs().idxmax()]
                            comp_direction = "aumenta" if strongest_competitor["Impacto (%)"] > 0 else "reduz"
                            
                            st.warning(f"**{selected_program} na {strongest_competitor['Emissora']}:** Quando exibido nesta emissora concorrente, "
                                     f"{comp_direction} a audiência da Globo em {abs(strongest_competitor['Impacto (%)']):.2f}% em média.")
                            
                            if strongest_competitor["Impacto (%)"] < 0:
                                st.error(f"👉 Isso sugere que o programa {selected_program} na {strongest_competitor['Emissora']} está "
                                       f"'roubando' audiência da TV Globo, potencialmente atraindo seus telespectadores.")
                        
                        # Create a bar chart comparing impact across broadcasters
                        impact_df["Impacto Formatado"] = impact_df["Impacto (%)"].apply(lambda x: f"{x:.2f}%")
                        
                        # Add a special color for Globo vs competitors
                        color_map = {"própria": "green", "concorrente": "red"}
                        
                        fig = px.bar(
                            impact_df,
                            x="Emissora",
                            y="Impacto (%)",
                            color="Tipo",
                            text="Impacto Formatado",
                            title=f"Impacto do {selected_program} na Audiência da TV Globo por Emissora",
                            color_discrete_map=color_map,
                            hover_data=["Ocorrências"]
                        )
                        
                        fig.update_layout(
                            xaxis_title="Emissora",
                            yaxis_title="Impacto na Audiência da Globo (%)",
                            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1)
                        )
                        
                        fig.add_annotation(
                            text="Acima de 0: Aumenta audiência da Globo | Abaixo de 0: Reduz audiência da Globo",
                            xref="paper", yref="paper",
                            x=0.5, y=1.05,
                            showarrow=False,
                            font=dict(size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Create timeline visualization
                        st.markdown("### Evolução Temporal")
                        
                        st.markdown(f"""
                        O gráfico abaixo mostra a evolução da audiência da TV Globo ao longo do tempo, 
                        com marcadores destacando quando o programa {selected_program} foi exibido em cada emissora.
                        Isso permite visualizar o impacto em tempo real do programa na audiência da Globo.
                        """)
                        
                        # Create figure for temporal analysis
                        fig_time = go.Figure()
                        
                        # Add base audience line
                        fig_time.add_trace(
                            go.Scatter(
                                x=selected_df["data_hora"],
                                y=selected_df[tv_col],
                                mode="lines",
                                name="Audiência TV Globo",
                                line=dict(color="lightgray", width=1)
                            )
                        )
                        
                        # Colors for different broadcasters
                        broadcaster_colors = {
                            "GLOBO": "green",
                            "SBT": "blue",
                            "RECORD": "red",
                            "BAND": "purple",
                            "TV BAND": "purple"
                        }
                        
                        # Add markers for program occurrences by broadcaster
                        for col in program_cols:
                            parts = col.split('_')
                            broadcaster = parts[3] if len(parts) >= 6 else "Unknown"
                            
                            # Filter data for when this program is on
                            program_data = selected_df[selected_df[col] > 0]
                            
                            if not program_data.empty:
                                color = broadcaster_colors.get(broadcaster, "gray")
                                
                                fig_time.add_trace(
                                    go.Scatter(
                                        x=program_data["data_hora"],
                                        y=program_data[tv_col],
                                        mode="markers",
                                        name=f"{selected_program} na {broadcaster}",
                                        marker=dict(
                                            color=color,
                                            size=8,
                                            symbol="circle"
                                        )
                                    )
                                )
                        
                        # Update layout
                        fig_time.update_layout(
                            title=f"Audiência da TV Globo Durante Exibição de {selected_program} por Emissora",
                            xaxis_title="Data/Hora",
                            yaxis_title="Audiência TV Globo (cov%)",
                            legend_title="Eventos",
                            hovermode="closest"
                        )
                        
                        st.plotly_chart(fig_time, use_container_width=True)
                        
                        # Display a table with the impact data
                        st.markdown("### Impacto Detalhado por Emissora")
                        
                        impact_table = impact_df[["Emissora", "Impacto Formatado", "Direção", "Ocorrências"]]
                        impact_table = impact_table.rename(columns={"Impacto Formatado": "Impacto na Audiência da Globo"})
                        
                        st.dataframe(impact_table, hide_index=True, use_container_width=True)
                    else:
                        st.warning(f"Não há dados suficientes para analisar o impacto do programa {selected_program}.")
                else:
                    st.warning(f"Não foram encontradas informações para o programa {selected_program}.")
            else:
                st.warning("Não foi possível identificar tipos de programas recorrentes.")
                
            # Show overall ranking of program impacts
            st.markdown("### Ranking de Impacto de Programas na Audiência da TV Globo")
            
            all_program_impacts = []
            
            for col in recorrente_cols:
                parts = col.split('_')
                if len(parts) >= 6:
                    program = parts[4]
                    broadcaster = parts[3]
                    
                    valid_data = selected_df.dropna(subset=[col, tv_col])
                    if valid_data.empty or valid_data[col].sum() == 0:
                        continue
                    
                    program_on = valid_data[valid_data[col] > 0][tv_col].mean()
                    program_off = valid_data[valid_data[col] == 0][tv_col].mean()
                    
                    if program_off > 0:
                        impact_pct = ((program_on / program_off) - 1) * 100
                        relation_type = "própria" if broadcaster == "GLOBO" else "concorrente"
                        
                        all_program_impacts.append({
                            "Programa": program,
                            "Emissora": broadcaster,
                            "Tipo": relation_type,
                            "Impacto (%)": impact_pct,
                            "Impacto Abs (%)": abs(impact_pct),
                            "Direção": "Positivo" if impact_pct > 0 else "Negativo"
                        })
            
            if all_program_impacts:
                # Convert to DataFrame
                all_impacts_df = pd.DataFrame(all_program_impacts)
                
                # Create two tables: top positive and top negative impacts
                if not all_impacts_df[all_impacts_df["Impacto (%)"] > 0].empty:
                    st.markdown("#### Programas com Maior Impacto Positivo na Audiência da Globo")
                    positive_df = all_impacts_df[all_impacts_df["Impacto (%)"] > 0].sort_values("Impacto (%)", ascending=False).head(5)
                    positive_df["Impacto (%)"] = positive_df["Impacto (%)"].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(
                        positive_df[["Programa", "Emissora", "Tipo", "Impacto (%)"]],
                        hide_index=True,
                        use_container_width=True
                    )
                
                if not all_impacts_df[all_impacts_df["Impacto (%)"] < 0].empty:
                    st.markdown("#### Programas com Maior Impacto Negativo na Audiência da Globo")
                    negative_df = all_impacts_df[all_impacts_df["Impacto (%)"] < 0].sort_values("Impacto (%)", ascending=True).head(5)
                    negative_df["Impacto (%)"] = negative_df["Impacto (%)"].apply(lambda x: f"{x:.2f}%")
                    
                    st.dataframe(
                        negative_df[["Programa", "Emissora", "Tipo", "Impacto (%)"]],
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # Create insight about competitors stealing audience
                    competitors_negative = negative_df[negative_df["Emissora"] != "GLOBO"]
                    if not competitors_negative.empty:
                        top_competitor = competitors_negative.iloc[0]
                        
                        st.error(f"""
                        **Maior 'Ladrão' de Audiência:** O programa {top_competitor['Programa']} na {top_competitor['Emissora']} 
                        causa a maior redução na audiência da TV Globo ({top_competitor['Impacto (%)']}).
                        
                        Isso indica uma forte competição direta por audiência, onde os telespectadores estão 
                        escolhendo este programa em vez da programação da Globo no mesmo horário.
                        """)
        else:
            st.warning("Não foram encontrados dados sobre programas recorrentes.")

    ########################################
    # 5. Genre Analysis Tab                #
    ########################################
    with tabs[2]:
        st.subheader("Análise de Gêneros de Programação")
        
        st.markdown("""
        Diferentes gêneros de programação têm impactos distintos na audiência da TV Globo.
        Esta análise mostra como cada gênero, tanto na própria Globo quanto em emissoras concorrentes,
        afeta os níveis de audiência da Globo, permitindo identificar:
        
        1. Quais gêneros da própria Globo atraem mais audiência
        2. Quais gêneros em emissoras concorrentes mais "roubam" audiência da Globo
        """)
        
        # Find columns for program genres
        genero_cols = [col for col in selected_df.columns if col.startswith('EXTERNO_GRADE_GENERO_')]
        
        if genero_cols and tv_col:
            # Extract unique genres and broadcasters
            genres = set()
            broadcasters = set()
            
            for col in genero_cols:
                parts = col.split('_')
                if len(parts) >= 6:  # Ensure we have enough parts
                    broadcaster = parts[3]  # Extract broadcaster
                    genre = parts[4]  # Extract genre
                    
                    genres.add(genre)
                    broadcasters.add(broadcaster)
            
            genres = sorted(list(genres))
            broadcasters = sorted(list(broadcasters))
            
            if genres:
                # Let user select a genre to analyze
                selected_genre = st.selectbox(
                    "Selecione um gênero para análise:",
                    options=genres
                )
                
                # Get all columns related to the selected genre
                genre_cols = [col for col in genero_cols if f"_GENERO_{selected_genre}_" in col or f"_GENERO_{selected_genre}" in col]
                
                if genre_cols:
                    st.markdown(f"### Impacto do Gênero {selected_genre} na Audiência da TV Globo")
                    
                    # Calculate impact for each broadcaster
                    impact_data = []
                    
                    for col in genre_cols:
                        # Extract broadcaster name
                        parts = col.split('_')
                        if len(parts) >= 4:
                            broadcaster = parts[3]
                            
                            # Calculate audience with and without genre
                            valid_data = selected_df.dropna(subset=[col, tv_col])
                            if valid_data.empty or valid_data[col].sum() == 0:
                                continue
                                
                            genre_on = valid_data[valid_data[col] > 0][tv_col].mean()
                            genre_off = valid_data[valid_data[col] == 0][tv_col].mean()
                            
                            # Calculate impact percentage
                            if genre_off > 0:  # Avoid division by zero
                                impact_pct = ((genre_on / genre_off) - 1) * 100
                                
                                # Count occurrences
                                occurrences = valid_data[valid_data[col] > 0].shape[0]
                                
                                relation_type = "própria" if broadcaster == "GLOBO" else "concorrente"
                                
                                impact_data.append({
                                    "Gênero": selected_genre,
                                    "Emissora": broadcaster,
                                    "Tipo": relation_type,
                                    "Impacto (%)": impact_pct,
                                    "Direção": "Positivo" if impact_pct > 0 else "Negativo",
                                    "Ocorrências": occurrences
                                })
                    
                    if impact_data:
                        # Convert to DataFrame and format
                        impact_df = pd.DataFrame(impact_data)
                        impact_df["Impacto Formatado"] = impact_df["Impacto (%)"].apply(lambda x: f"{x:.2f}%")
                        
                        # Create a key insight about this genre's impact
                        if "GLOBO" in impact_df["Emissora"].values:
                            globo_impact = impact_df[impact_df["Emissora"] == "GLOBO"]["Impacto (%)"].values[0]
                            direction = "aumenta" if globo_impact > 0 else "reduz"
                            
                            st.info(f"**{selected_genre} na Globo:** Quando este gênero é exibido na própria Globo, {direction} "
                                  f"a audiência da emissora em {abs(globo_impact):.2f}% em média.")
                        
                        # Find competitors with biggest impact
                        competitors = impact_df[impact_df["Emissora"] != "GLOBO"]
                        if not competitors.empty:
                            strongest_competitor = competitors.loc[competitors["Impacto (%)"].abs().idxmax()]
                            comp_direction = "aumenta" if strongest_competitor["Impacto (%)"] > 0 else "reduz"
                            
                            st.warning(f"**{selected_genre} na {strongest_competitor['Emissora']}:** Quando este gênero é exibido nesta emissora concorrente, "
                                     f"{comp_direction} a audiência da Globo em {abs(strongest_competitor['Impacto (%)']):.2f}% em média.")
                            
                            if strongest_competitor["Impacto (%)"] < 0:
                                st.error(f"👉 Isso sugere que programas do gênero {selected_genre} na {strongest_competitor['Emissora']} estão "
                                       f"'roubando' audiência da TV Globo, potencialmente atraindo seus telespectadores.")
                        
                        # Add a special color for Globo vs competitors
                        color_map = {"própria": "green", "concorrente": "red"}
                        
                        # Create a bar chart comparing impact across broadcasters
                        fig = px.bar(
                            impact_df,
                            x="Emissora",
                            y="Impacto (%)",
                            color="Tipo",
                            text="Impacto Formatado",
                            title=f"Impacto do Gênero {selected_genre} na Audiência da TV Globo por Emissora",
                            color_discrete_map=color_map,
                            hover_data=["Ocorrências"]
                        )
                        
                        fig.update_layout(
                            xaxis_title="Emissora",
                            yaxis_title="Impacto na Audiência da Globo (%)",
                            yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1)
                        )
                        
                        fig.add_annotation(
                            text="Acima de 0: Aumenta audiência da Globo | Abaixo de 0: Reduz audiência da Globo",
                            xref="paper", yref="paper",
                            x=0.5, y=1.05,
                            showarrow=False,
                            font=dict(size=10)
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display a table with the impact data
                        st.markdown("### Impacto Detalhado por Emissora")
                        
                        impact_table = impact_df[["Emissora", "Impacto Formatado", "Direção", "Ocorrências"]]
                        impact_table = impact_table.rename(columns={"Impacto Formatado": "Impacto na Audiência da Globo"})
                        
                        st.dataframe(impact_table, hide_index=True, use_container_width=True)
                    else:
                        st.warning(f"Não há dados suficientes para analisar o impacto do gênero {selected_genre}.")
                else:
                    st.warning(f"Não foram encontradas informações para o gênero {selected_genre}.")
                    
                # Show overall ranking of genre impacts
                st.markdown("### Ranking de Impacto de Gêneros na Audiência da TV Globo")
                
                all_genre_impacts = []
                
                for col in genero_cols:
                    parts = col.split('_')
                    if len(parts) >= 5:
                        broadcaster = parts[3]
                        genre = parts[4]
                        
                        valid_data = selected_df.dropna(subset=[col, tv_col])
                        if valid_data.empty or valid_data[col].sum() == 0:
                            continue
                        
                        genre_on = valid_data[valid_data[col] > 0][tv_col].mean()
                        genre_off = valid_data[valid_data[col] == 0][tv_col].mean()
                        
                        if genre_off > 0:
                            impact_pct = ((genre_on / genre_off) - 1) * 100
                            relation_type = "própria" if broadcaster == "GLOBO" else "concorrente"
                            
                            all_genre_impacts.append({
                                "Gênero": genre,
                                "Emissora": broadcaster,
                                "Tipo": relation_type,
                                "Impacto (%)": impact_pct,
                                "Impacto Abs (%)": abs(impact_pct),
                                "Direção": "Positivo" if impact_pct > 0 else "Negativo"
                            })
                
                if all_genre_impacts:
                    # Convert to DataFrame
                    all_impacts_df = pd.DataFrame(all_genre_impacts)
                    
                    # Create two tabs for viewing by genre or by broadcaster
                    genre_view_tabs = st.tabs(["Ver por Gênero", "Ver por Emissora"])
                    
                    with genre_view_tabs[0]:
                        # Calculate average impact by genre across broadcasters
                        genre_avg = all_impacts_df.groupby("Gênero")["Impacto (%)"].mean().reset_index()
                        genre_avg = genre_avg.sort_values("Impacto (%)", ascending=False)
                        
                        # Top and bottom 5 genres
                        top_genres = genre_avg.head(5)
                        bottom_genres = genre_avg.tail(5)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### Top 5 Gêneros que Mais Aumentam Audiência da Globo")
                            fig_top = px.bar(
                                top_genres,
                                x="Gênero",
                                y="Impacto (%)",
                                text=top_genres["Impacto (%)"].apply(lambda x: f"{x:.2f}%"),
                                color="Impacto (%)",
                                color_continuous_scale=["yellow", "green"]
                            )
                            
                            fig_top.update_layout(
                                xaxis_title="Gênero",
                                yaxis_title="Impacto Médio (%)",
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig_top, use_container_width=True)
                        
                        with col2:
                            st.markdown("#### Top 5 Gêneros que Mais Reduzem Audiência da Globo")
                            fig_bottom = px.bar(
                                bottom_genres,
                                x="Gênero",
                                y="Impacto (%)",
                                text=bottom_genres["Impacto (%)"].apply(lambda x: f"{x:.2f}%"),
                                color="Impacto (%)",
                                color_continuous_scale=["red", "yellow"]
                            )
                            
                            fig_bottom.update_layout(
                                xaxis_title="Gênero",
                                yaxis_title="Impacto Médio (%)",
                                xaxis_tickangle=-45
                            )
                            
                            st.plotly_chart(fig_bottom, use_container_width=True)
                    
                    with genre_view_tabs[1]:
                        # Group by broadcaster and display top genres for each
                        for broadcaster in broadcasters:
                            broadcaster_data = all_impacts_df[all_impacts_df["Emissora"] == broadcaster]
                            
                            if not broadcaster_data.empty:
                                st.markdown(f"#### Top Gêneros na {broadcaster}")
                                
                                # Sort by impact
                                broadcaster_data = broadcaster_data.sort_values("Impacto (%)", ascending=False)
                                broadcaster_data["Impacto (%)"] = broadcaster_data["Impacto (%)"].apply(lambda x: f"{x:.2f}%")
                                
                                # Show the table
                                st.dataframe(
                                    broadcaster_data[["Gênero", "Impacto (%)", "Direção"]],
                                    hide_index=True,
                                    use_container_width=True
                                )
                                
                                # Add an insight about competition
                                if broadcaster != "GLOBO":
                                    negative_impacts = all_impacts_df[(all_impacts_df["Emissora"] == broadcaster) & 
                                                                    (all_impacts_df["Impacto (%)"] < 0)]
                                    
                                    if not negative_impacts.empty:
                                        worst_genre = negative_impacts.sort_values("Impacto (%)").iloc[0]
                                        
                                        st.warning(f"""
                                        **Competição Direta:** O gênero {worst_genre['Gênero']} na {broadcaster} 
                                        causa uma redução de {abs(worst_genre['Impacto (%)']):.2f}% na audiência da Globo,
                                        o que sugere forte competição neste segmento de programação.
                                        """)
                else:
                    st.warning("Não há dados suficientes para criar um ranking de gêneros.")
            else:
                st.warning("Não foi possível identificar gêneros de programação.")
        else:
            st.warning("Não foram encontrados dados sobre gêneros de programação.")

    ########################################
    # 6. Isolated Events Analysis Tab      #
    ########################################
    with tabs[3]:
        st.subheader("Análise de Eventos Isolados")
        
        st.markdown("""
        Eventos isolados representam acontecimentos pontuais como notícias importantes, eventos especiais, 
        ou lançamentos que podem ter um impacto significativo na audiência da TV Globo.
        
        Diferentemente dos programas recorrentes, estes eventos são únicos ou raros, tornando sua análise 
        particularmente valiosa para entender o impacto de acontecimentos específicos.
        """)
        
        # Find columns for isolated events
        isolado_cols = [col for col in selected_df.columns if col.startswith('EXTERNO_ISOLADO_')]
        
        if isolado_cols and tv_col:
            # Create a timeline showing isolated events
            st.markdown("### Linha do Tempo de Eventos Isolados")
            
            st.markdown("""
            O gráfico abaixo mostra a audiência da TV Globo ao longo do tempo, com marcadores nos momentos 
            em que ocorreram eventos isolados específicos. Isso permite visualizar o impacto imediato destes 
            eventos na audiência.
            """)
            
            # Create timeline figure
            fig_timeline = go.Figure()
            
            # Add base audience line
            fig_timeline.add_trace(
                go.Scatter(
                    x=selected_df["data_hora"],
                    y=selected_df[tv_col],
                    mode="lines",
                    name="Audiência TV Globo",
                    line=dict(color="lightgray", width=1)
                )
            )
            
            # Generate colors for different events
            event_colors = px.colors.qualitative.Plotly
            
            # Add markers for each isolated event
            for i, col in enumerate(isolado_cols):
                event_name = col.replace('EXTERNO_ISOLADO_', '').replace('_', ' ')
                color = event_colors[i % len(event_colors)]
                
                # Get data points where the event occurred
                event_data = selected_df[selected_df[col] > 0]
                
                if not event_data.empty:
                    fig_timeline.add_trace(
                        go.Scatter(
                            x=event_data["data_hora"],
                            y=event_data[tv_col],
                            mode="markers",
                            name=event_name,
                            marker=dict(
                                color=color,
                                size=10,
                                symbol="star"
                            )
                        )
                    )
            
            # Update layout
            fig_timeline.update_layout(
                title="Audiência da TV Globo Durante Eventos Isolados",
                xaxis_title="Data/Hora",
                yaxis_title="Audiência TV Globo (cov%)",
                legend_title="Eventos",
                hovermode="closest"
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Calculate impact of each isolated event
            st.markdown("### Impacto dos Eventos Isolados na Audiência")
            
            st.markdown("""
            A tabela abaixo quantifica o impacto de cada evento isolado na audiência da TV Globo.
            Para uma comparação justa, comparamos a audiência durante o evento com a audiência média
            em períodos similares (mesmo dia da semana e horário) sem eventos.
            """)
            
            # Calculate baseline for fair comparison
            event_impacts = []
            
            for col in isolado_cols:
                event_name = col.replace('EXTERNO_ISOLADO_', '').replace('_', ' ')
                
                # Get data points where the event occurred
                event_data = selected_df[selected_df[col] > 0]
                
                if not event_data.empty:
                    # Extract day of week and hour information for comparison
                    event_data["day_of_week"] = event_data["data_hora"].dt.dayofweek
                    event_data["hour"] = event_data["data_hora"].dt.hour
                    
                    # Calculate average audience during the event
                    event_audience = event_data[tv_col].mean()
                    
                    # Create baseline for comparison - same days of week and hours without the event
                    similar_periods = []
                    
                    for _, row in event_data.iterrows():
                        day = row["day_of_week"]
                        hour = row["hour"]
                        
                        # Find similar periods (same day & hour) without the event
                        similar = selected_df[
                            (selected_df["data_hora"].dt.dayofweek == day) & 
                            (selected_df["data_hora"].dt.hour == hour) & 
                            (selected_df[col] == 0)
                        ]
                        
                        similar_periods.append(similar)
                    
                    # Combine all similar periods
                    if similar_periods:
                        baseline_data = pd.concat(similar_periods, ignore_index=True)
                        
                        if not baseline_data.empty:
                            baseline_audience = baseline_data[tv_col].mean()
                            
                            # Calculate impact
                            if baseline_audience > 0:
                                impact_pct = ((event_audience / baseline_audience) - 1) * 100
                                
                                event_impacts.append({
                                    "Evento": event_name,
                                    "Audiência Durante Evento": event_audience,
                                    "Audiência Típica (Baseline)": baseline_audience,
                                    "Impacto (%)": impact_pct,
                                    "Direção": "Positivo" if impact_pct > 0 else "Negativo",
                                    "Ocorrências": len(event_data)
                                })
            
            if event_impacts:
                # Convert to DataFrame and sort by impact
                impact_df = pd.DataFrame(event_impacts)
                impact_df = impact_df.sort_values("Impacto (%)", ascending=False)
                
                # Format columns for display
                impact_df["Audiência Durante Evento"] = impact_df["Audiência Durante Evento"].apply(lambda x: f"{x:.2f}%")
                impact_df["Audiência Típica (Baseline)"] = impact_df["Audiência Típica (Baseline)"].apply(lambda x: f"{x:.2f}%")
                impact_df["Impacto (%)"] = impact_df["Impacto (%)"].apply(lambda x: f"{x:.2f}%")
                
                # Display as table
                st.dataframe(impact_df, hide_index=True, use_container_width=True)
                
                # Create a bar chart of impacts
                impact_df_plot = pd.DataFrame(event_impacts)  # Create a new copy for plotting
                
                fig_impact = px.bar(
                    impact_df_plot,
                    x="Evento",
                    y="Impacto (%)",
                    color="Direção",
                    text=impact_df_plot["Impacto (%)"].apply(lambda x: f"{x:.2f}%"),
                    title="Impacto de Eventos Isolados na Audiência da TV Globo",
                    color_discrete_map={"Positivo": "green", "Negativo": "red"},
                    hover_data=["Ocorrências"]
                )
                
                fig_impact.update_layout(
                    xaxis_title="Evento",
                    yaxis_title="Impacto na Audiência (%)",
                    xaxis_tickangle=-45,
                    yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1)
                )
                
                st.plotly_chart(fig_impact, use_container_width=True)
                
                # Generate insights
                if not impact_df.empty:
                    top_positive = impact_df[impact_df["Direção"] == "Positivo"]
                    top_negative = impact_df[impact_df["Direção"] == "Negativo"]
                    
                    if not top_positive.empty:
                        top_pos = top_positive.iloc[0]
                        
                        st.success(f"""
                        **Evento com Maior Impacto Positivo:** {top_pos['Evento']} aumentou a audiência da Globo em 
                        {top_pos['Impacto (%)']} comparado com períodos similares sem o evento.
                        
                        Este tipo de evento representa uma oportunidade para aumentar a audiência em momentos estratégicos.
                        """)
                    
                    if not top_negative.empty:
                        top_neg = top_negative.iloc[0]
                        
                        st.error(f"""
                        **Evento com Maior Impacto Negativo:** {top_neg['Evento']} reduziu a audiência da Globo em 
                        {top_neg['Impacto (%)']} comparado com períodos similares.
                        
                        Eventos deste tipo podem estar desviando a atenção dos telespectadores para outras atividades
                        ou canais concorrentes.
                        """)
            else:
                st.warning("Não foi possível calcular o impacto dos eventos isolados devido a dados insuficientes.")
        else:
            st.warning("Não foram encontrados dados sobre eventos isolados.")

    ######################################
    # 7. Social Volume Analysis Tab      #
    ######################################
    with tabs[4]:
        st.subheader("Análise de Volume Social")
        
        st.markdown("""
        O volume de conversas nas redes sociais pode ser tanto um indicador como um impulsionador da audiência de TV.
        Programas que geram mais conversação podem atrair novos espectadores, enquanto assuntos muito discutidos
        nas redes sociais podem desviar a atenção da TV.
        
        Esta análise examina a relação entre o volume de atividade social e métricas de audiência da TV Globo.
        """)
        
        # Find social volume columns based on new naming pattern
        social_cols = [col for col in selected_df.columns if col.startswith('EXTERNO_NPS_')]
        
        if social_cols and tv_col:
            # Filter data where both social and tv_col have valid values
            valid_social_data = selected_df.dropna(subset=social_cols + [tv_col])
            
            # Filter for non-zero social values
            for col in social_cols:
                valid_social_data = valid_social_data[valid_social_data[col] > 0]
            
            if not valid_social_data.empty:
                # Create time series chart comparing social volume with TV audience
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add TV audience line
                fig.add_trace(
                    go.Scatter(
                        x=valid_social_data['data_hora'],
                        y=valid_social_data[tv_col],
                        name='Audiência TV Globo (cov%)',
                        line=dict(color='rgb(31, 119, 180)', width=3)
                    ),
                    secondary_y=False
                )
                
                # Add social volume lines with different colors
                colors = ['rgb(255, 127, 14)', 'rgb(44, 160, 44)', 'rgb(214, 39, 40)', 
                        'rgb(148, 103, 189)', 'rgb(140, 86, 75)']
                
                for i, col in enumerate(social_cols):
                    display_name = col.replace('EXTERNO_NPS_', '').replace('_', ' ').title()
                    
                    fig.add_trace(
                        go.Scatter(
                            x=valid_social_data['data_hora'],
                            y=valid_social_data[col],
                            name=display_name,
                            line=dict(color=colors[i % len(colors)], width=2)
                        ),
                        secondary_y=True
                    )
                
                # Update layout
                time_unit = "" if granularity == "Horário" else "Diária" if granularity == "Diário" else "Semanal"
                fig.update_layout(
                    title=f'Evolução {time_unit} da Audiência TV Globo vs. Volume Social ({granularity})',
                    xaxis_title='Data e Hora' if granularity == "Horário" else 'Data',
                    yaxis_title='Audiência TV Globo (cov%)',
                    yaxis2_title='Volume Social',
                    legend_title='Métricas',
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Create correlation analysis
                st.markdown("### Correlação entre Volume Social e Audiência TV")
                
                # Calculate correlation matrix
                correlation_data = []
                
                for col in social_cols:
                    display_name = col.replace('EXTERNO_NPS_', '').replace('_', ' ').title()
                    corr = valid_social_data[col].corr(valid_social_data[tv_col])
                    
                    correlation_data.append({
                        'Métrica Social': display_name,
                        'Correlação': corr,
                        'Força': abs(corr)
                    })
                
                if correlation_data:
                    corr_df = pd.DataFrame(correlation_data)
                    corr_df = corr_df.sort_values('Força', ascending=False)
                    
                    # Create bar chart of correlations
                    fig_corr = px.bar(
                        corr_df,
                        x='Métrica Social',
                        y='Correlação',
                        color='Correlação',
                        text=corr_df['Correlação'].apply(lambda x: f"{x:.2f}"),
                        title='Correlação entre Métricas Sociais e Audiência TV Globo',
                        color_continuous_scale=['red', 'white', 'green']
                    )
                    
                    st.plotly_chart(fig_corr, use_container_width=True)
                    
                    # Show scatter plot for the most correlated metric
                    if not corr_df.empty:
                        strongest_metric = corr_df.iloc[0]['Métrica Social']
                        strongest_col = [col for col in social_cols if strongest_metric.lower().replace(' ', '_') in col.lower()][0]
                        
                        st.markdown(f"### Relação Detalhada: {strongest_metric} vs. Audiência")
                        
                        fig_scatter = px.scatter(
                            valid_social_data,
                            x=strongest_col,
                            y=tv_col,
                            trendline="ols",
                            labels={
                                strongest_col: strongest_metric,
                                tv_col: 'Audiência TV Globo (cov%)'
                            },
                            title=f'Relação entre {strongest_metric} e Audiência TV Globo'
                        )
                        
                        st.plotly_chart(fig_scatter, use_container_width=True)
                        
                        # Add insight about relationship
                        strongest_corr = corr_df.iloc[0]['Correlação']
                        
                        if abs(strongest_corr) > 0.3:
                            if strongest_corr > 0:
                                st.success(f"""
                                **Correlação Positiva Significativa:** O aumento no {strongest_metric} está associado a um 
                                aumento na audiência da TV Globo (correlação: {strongest_corr:.2f}).
                                
                                Isso sugere que este tipo de atividade social pode estar impulsionando telespectadores 
                                para assistir à programação da Globo, possivelmente indicando engajamento em "segunda tela".
                                """)
                            else:
                                st.error(f"""
                                **Correlação Negativa Significativa:** O aumento no {strongest_metric} está associado a uma 
                                redução na audiência da TV Globo (correlação: {strongest_corr:.2f}).
                                
                                Isso sugere que este tipo de atividade social pode estar competindo pela atenção dos 
                                telespectadores, potencialmente desviando-os da programação da Globo.
                                """)
                
                # Create lag analysis if not hourly
                if granularity != "Horário" and len(social_cols) > 0:
                    primary_social_col = social_cols[0]
                    display_name = primary_social_col.replace('EXTERNO_NPS_', '').replace('_', ' ').title()
                    
                    st.markdown("### Análise de Lag: Volume Social vs. Audiência")
                    
                    st.markdown("""
                    Esta análise verifica se um aumento no volume social precede ou sucede 
                    um aumento na audiência TV. Um lag positivo significativo sugere que a conversação
                    social pode funcionar como um preditor da audiência futura.
                    """)
                    
                    # Create lag columns
                    lags = [1, 2, 3, 7]  # Look at 1, 2, 3, and 7 days/weeks lag
                    lag_corrs = []
                    
                    for lag in lags:
                        lag_col = f'social_lag_{lag}'
                        valid_social_data[lag_col] = valid_social_data[primary_social_col].shift(lag)
                        
                        # Calculate correlation with audience, excluding NaN values
                        lag_data = valid_social_data.dropna(subset=[lag_col, tv_col])
                        
                        if not lag_data.empty:
                            corr = lag_data[lag_col].corr(lag_data[tv_col])
                            if not pd.isna(corr):  # Only include valid correlations
                                unit = "dias" if granularity == "Diário" else "semanas"
                                lag_corrs.append({
                                    f'Lag ({unit})': lag,
                                    'Correlação': corr
                                })
                    
                    if lag_corrs:
                        # Convert to DataFrame
                        lag_df = pd.DataFrame(lag_corrs)
                        
                        # Create bar chart
                        unit = "dias" if granularity == "Diário" else "semanas"
                        fig_lag = px.bar(
                            lag_df,
                            x=f'Lag ({unit})',
                            y='Correlação',
                            title=f'Correlação entre Volume Social (com lag) e Audiência TV Globo',
                            labels={
                                f'Lag ({unit})': f'Volume Social Defasado ({unit})',
                                'Correlação': 'Correlação com Audiência TV Globo'
                            },
                            color='Correlação',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        
                        st.plotly_chart(fig_lag, use_container_width=True)
                        
                        # Generate insights based on lag analysis
                        max_lag_corr = max(lag_df['Correlação'])
                        max_lag = lag_df.loc[lag_df['Correlação'].idxmax(), f'Lag ({unit})']
                        
                        if abs(max_lag_corr) > 0.3:
                            direction = "positiva" if max_lag_corr > 0 else "negativa"
                            effect = "preditor" if max_lag_corr > 0 else "indicador de possível queda"
                            
                            st.success(f"**Efeito Temporal:** A correlação {direction} mais forte ({max_lag_corr:.2f}) "
                                    f"ocorre com {max_lag} {unit} de defasagem, sugerindo que o volume social "
                                    f"pode ser um {effect} da audiência TV Globo futura.")
                else:
                    # For hourly analysis, provide insight about real-time social engagement
                    st.markdown("### Análise de Engajamento Social em Tempo Real")
                    
                    st.markdown("""
                    Para dados por hora, a análise de correlação em tempo real é mais relevante que a análise de lag.
                    A correlação atual mostra se as pessoas estão engajadas nas redes sociais durante o mesmo período
                    em que estão assistindo TV, o que pode indicar o uso de "segunda tela" ou conversação social
                    sobre os programas em andamento.
                    """)
                    
                    # Analyze correlation during prime time vs other times for hourly data
                    if granularity == "Horário" and len(social_cols) > 0:
                        primary_social_col = social_cols[0]
                        
                        prime_time_hours = list(range(19, 23))  # 7pm to 10pm
                        
                        prime_df = valid_social_data[valid_social_data['data_hora'].dt.hour.isin(prime_time_hours)]
                        non_prime_df = valid_social_data[~valid_social_data['data_hora'].dt.hour.isin(prime_time_hours)]
                        
                        if not prime_df.empty and not non_prime_df.empty:
                            prime_corr = prime_df[primary_social_col].corr(prime_df[tv_col])
                            non_prime_corr = non_prime_df[primary_social_col].corr(non_prime_df[tv_col])
                            
                            if not pd.isna(prime_corr) and not pd.isna(non_prime_corr):
                                stronger = "horário nobre" if abs(prime_corr) > abs(non_prime_corr) else "fora do horário nobre"
                                
                                st.success(f"**Diferença por Horário:** A correlação entre volume social e audiência da Globo é mais forte durante o {stronger} "
                                        f"({prime_corr:.2f} vs {non_prime_corr:.2f}), sugerindo que "
                                        f"{'o comportamento de segunda tela é mais prevalente no horário nobre' if stronger == 'horário nobre' else 'o uso de redes sociais durante o dia tem impacto maior na audiência fora do horário nobre'}.")
            else:
                st.warning("Dados insuficientes para análise de volume social no período selecionado.")
        else:
            st.warning("Dados sobre volume social não estão disponíveis.")

    ######################################
    # 8. Summary Analysis Tab            #
    ######################################
    with tabs[5]:
        st.subheader("Resumo dos Insights")
        
        st.markdown("""
        Esta seção consolida os principais insights de todas as categorias analisadas,
        destacando os fatores externos que mais impactam a audiência da TV Globo.
        """)
        
        if tv_col:
            # Collect all relevant columns for correlation analysis
            relevant_cols = []
            relevant_cols.extend([col for col in selected_df.columns if col.startswith('EXTERNO_ECONOMICO_')])
            relevant_cols.extend([col for col in selected_df.columns if col.startswith('EXTERNO_GRADE_')])
            relevant_cols.extend([col for col in selected_df.columns if col.startswith('EXTERNO_ISOLADO_')])
            relevant_cols.extend([col for col in selected_df.columns if col.startswith('EXTERNO_NPS_')])
            
            if relevant_cols:
                # Calculate correlation with audience metric
                correlations = {}
                
                for col in relevant_cols:
                    valid_data = selected_df.dropna(subset=[col, tv_col])
                    if not valid_data.empty and valid_data[col].std() > 0:  # Ensure there's variation
                        corr = valid_data[col].corr(valid_data[tv_col])
                        if not pd.isna(corr):
                            correlations[col] = corr
                
                if correlations:
                    # Convert to DataFrame
                    corr_df = pd.DataFrame({
                        'Fator': list(correlations.keys()),
                        'Correlação': list(correlations.values()),
                        'Força': [abs(c) for c in correlations.values()]
                    })
                    
                    # Add category
                    def categorize(factor):
                        if factor.startswith('EXTERNO_ECONOMICO_'):
                            return 'Indicador Econômico'
                        elif factor.startswith('EXTERNO_GRADE_RECORRENTE_'):
                            parts = factor.split('_')
                            if len(parts) >= 5:
                                broadcaster = parts[4]
                                if broadcaster == "GLOBO":
                                    return 'Programa Globo'
                                else:
                                    return 'Programa Concorrente'
                            return 'Programa Recorrente'
                        elif factor.startswith('EXTERNO_GRADE_GENERO_'):
                            parts = factor.split('_')
                            if len(parts) >= 4:
                                broadcaster = parts[3]
                                if broadcaster == "GLOBO":
                                    return 'Gênero Globo'
                                else:
                                    return 'Gênero Concorrente'
                            return 'Gênero de Programação'
                        elif factor.startswith('EXTERNO_ISOLADO_'):
                            return 'Evento Isolado'
                        elif factor.startswith('EXTERNO_NPS_'):
                            return 'Volume Social'
                        else:
                            return 'Outros'
                    
                    corr_df['Categoria'] = corr_df['Fator'].apply(categorize)
                    
                    # Clean up factor names for display
                    def clean_factor_name(factor):
                        if factor.startswith('EXTERNO_ECONOMICO_'):
                            return factor.replace('EXTERNO_ECONOMICO_', '').replace('_', ' ').title()
                        elif factor.startswith('EXTERNO_GRADE_RECORRENTE_'):
                            parts = factor.split('_')
                            if len(parts) >= 6:
                                program = parts[4]
                                broadcaster = parts[3]
                                return f"{program} ({broadcaster})"
                            return factor.replace('EXTERNO_GRADE_RECORRENTE_', '')
                        elif factor.startswith('EXTERNO_GRADE_GENERO_'):
                            parts = factor.split('_')
                            if len(parts) >= 5:
                                broadcaster = parts[3]
                                genre = parts[4]
                                return f"{genre} ({broadcaster})"
                            return factor.replace('EXTERNO_GRADE_GENERO_', '')
                        elif factor.startswith('EXTERNO_ISOLADO_'):
                            return factor.replace('EXTERNO_ISOLADO_', '').replace('_', ' ')
                        elif factor.startswith('EXTERNO_NPS_'):
                            return factor.replace('EXTERNO_NPS_', '').replace('_', ' ').title()
                        else:
                            return factor
                    
                    corr_df['Fator Formatado'] = corr_df['Fator'].apply(clean_factor_name)
                    
                    # Add relationship to Globo
                    def determine_relationship(row):
                        if row['Categoria'] in ['Programa Concorrente', 'Gênero Concorrente']:
                            if row['Correlação'] < 0:
                                return "Rouba audiência da Globo"
                            else:
                                return "Não compete com Globo"
                        elif row['Categoria'] in ['Programa Globo', 'Gênero Globo']:
                            if row['Correlação'] > 0:
                                return "Aumenta audiência da Globo"
                            else:
                                return "Reduz audiência da Globo"
                        else:
                            return "Correlação Neutra"
                    
                    corr_df['Relação com Globo'] = corr_df.apply(determine_relationship, axis=1)
                    
                    # Sort by absolute correlation strength
                    corr_df = corr_df.sort_values('Força', ascending=False)
                    
                    # Display top factors overall
                    st.markdown("### Fatores Externos com Maior Impacto na Audiência da Globo")
                    
                    # Take top 10 overall
                    top_overall = corr_df.head(10)
                    
                    # Create a color map based on relationship to Globo
                    relationship_colors = {
                        "Rouba audiência da Globo": "red",
                        "Não compete com Globo": "lightblue",
                        "Aumenta audiência da Globo": "green",
                        "Reduz audiência da Globo": "orange",
                        "Correlação Neutra": "gray"
                    }
                    
                    fig_top = px.bar(
                        top_overall,
                        x='Fator Formatado',
                        y='Correlação',
                        color='Relação com Globo',
                        text=top_overall['Correlação'].apply(lambda x: f"{x:.2f}"),
                        title='Top 10 Fatores Externos por Correlação com Audiência da TV Globo',
                        color_discrete_map=relationship_colors
                    )
                    
                    fig_top.update_layout(
                        xaxis_title='Fator Externo',
                        yaxis_title='Correlação com Audiência TV Globo (cov%)',
                        xaxis_tickangle=-45,
                        yaxis=dict(zeroline=True, zerolinecolor='black', zerolinewidth=1)
                    )
                    
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Create special section for competition analysis
                    st.markdown("### Análise de Competição: Quem 'Rouba' Audiência da Globo?")
                    
                    competition_df = corr_df[
                        (corr_df['Categoria'].isin(['Programa Concorrente', 'Gênero Concorrente'])) & 
                        (corr_df['Correlação'] < 0)
                    ].sort_values('Correlação', ascending=True)
                    
                    if not competition_df.empty:
                        top_competitors = competition_df.head(5)
                        
                        fig_comp = px.bar(
                            top_competitors,
                            x='Fator Formatado',
                            y='Correlação',
                            text=top_competitors['Correlação'].apply(lambda x: f"{x:.2f}"),
                            title='Top 5 Programações Concorrentes que Mais "Roubam" Audiência da Globo',
                            color='Categoria',
                            color_discrete_map={"Programa Concorrente": "red", "Gênero Concorrente": "darkred"}
                        )
                        
                        fig_comp.update_layout(
                            xaxis_title='Programação Concorrente',
                            yaxis_title='Correlação com Audiência Globo',
                            xaxis_tickangle=-45
                        )
                        
                        st.plotly_chart(fig_comp, use_container_width=True)
                        
                        # Create insight about competition
                        top_competitor = top_competitors.iloc[0]
                        
                        st.error(f"""
                        **Competidor Mais Forte:** {top_competitor['Fator Formatado']} tem a correlação negativa mais 
                        forte ({top_competitor['Correlação']:.2f}) com a audiência da Globo.
                        
                        Isso indica que este conteúdo concorrente está efetivamente atraindo telespectadores 
                        que poderiam estar assistindo à Globo, representando uma ameaça competitiva significativa.
                        """)
                    else:
                        st.info("Não foram identificados competidores significativos com correlação negativa.")
                    
                    # Display top factors by category
                    st.markdown("### Análise por Categoria de Fator Externo")
                    
                    # Create columns for the top factor in each category
                    categories = corr_df['Categoria'].unique()
                    cols = st.columns(min(4, len(categories)))
                    
                    for i, category in enumerate(categories):
                        # Get top factor in this category
                        top_in_category = corr_df[corr_df['Categoria'] == category].iloc[0] if not corr_df[corr_df['Categoria'] == category].empty else None
                        
                        if top_in_category is not None:
                            with cols[i % 4]:
                                direction = "positiva" if top_in_category['Correlação'] > 0 else "negativa"
                                
                                st.metric(
                                    label=f"Top em {category}",
                                    value=top_in_category['Fator Formatado'],
                                    delta=f"{top_in_category['Correlação']:.2f} ({direction})"
                                )
                    
                    # Generate key insights
                    st.markdown("### Insights Estratégicos")
                    
                    # Economic insight
                    eco_df = corr_df[corr_df['Categoria'] == 'Indicador Econômico']
                    if not eco_df.empty:
                        top_eco = eco_df.iloc[0]
                        direction = "positiva" if top_eco['Correlação'] > 0 else "negativa"
                        impact = "aumento" if top_eco['Correlação'] > 0 else "redução"
                        
                        st.info(f"""
                        **Economia:** O indicador {top_eco['Fator Formatado']} tem a correlação {direction} mais forte ({top_eco['Correlação']:.2f}) 
                        com a audiência da Globo, sugerindo que seu {impact} está associado a {'maior' if top_eco['Correlação'] > 0 else 'menor'} 
                        consumo de TV Linear Globo.
                        
                        **Implicação:** Monitorar este indicador pode ajudar a prever flutuações na audiência e adaptar estratégias comerciais 
                        e de programação de acordo.
                        """)
                    
                    # Globo programs
                    globo_progs = corr_df[corr_df['Categoria'] == 'Programa Globo']
                    if not globo_progs.empty:
                        top_prog = globo_progs.iloc[0]
                        
                        st.success(f"""
                        **Programação Própria:** O programa {top_prog['Fator Formatado']} tem o maior impacto positivo 
                        na audiência geral da Globo (correlação: {top_prog['Correlação']:.2f}).
                        
                        **Implicação:** Este tipo de conteúdo representa um ponto forte da emissora e deve ser 
                        potencializado em termos de investimento e marketing.
                        """)
                    
                    # Competition threat
                    competitors = corr_df[(corr_df['Categoria'].isin(['Programa Concorrente', 'Gênero Concorrente'])) & (corr_df['Correlação'] < 0)]
                    if not competitors.empty:
                        top_threat = competitors.iloc[0]
                        
                        st.error(f"""
                        **Ameaça Competitiva:** {top_threat['Fator Formatado']} é o conteúdo concorrente que mais 
                        reduz a audiência da Globo (correlação: {top_threat['Correlação']:.2f}).
                        
                        **Implicação:** Este é um ponto de vulnerabilidade da Globo. Considerar estratégias como 
                        contraprogramação ou fortalecimento de conteúdos similares no portfólio próprio pode ser necessário.
                        """)
                    
                    # Generate an overall business recommendation
                    st.markdown("### Recomendação para o Negócio")
                    
                    # Organize factors by correlation type
                    pos_factors = corr_df[corr_df['Correlação'] > 0.3].head(3)
                    neg_factors = corr_df[(corr_df['Correlação'] < -0.3) & (corr_df['Categoria'].isin(['Programa Concorrente', 'Gênero Concorrente']))].head(3)
                    
                    pos_factors_list = ", ".join([f"{row['Fator Formatado']} ({row['Correlação']:.2f})" for _, row in pos_factors.iterrows()])
                    neg_factors_list = ", ".join([f"{row['Fator Formatado']} ({row['Correlação']:.2f})" for _, row in neg_factors.iterrows()])
                    
                    st.success(f"""
                    **Estratégia Baseada nos Dados:**
                    
                    Com base na análise de correlação entre fatores externos e audiência da TV Globo, recomendamos:
                    
                    1. **Potencializar fatores positivos:** Aproveitar e amplificar a presença de {pos_factors_list}, que demonstraram forte correlação positiva com a audiência da Globo.
                    
                    2. **Mitigar ameaças competitivas:** Desenvolver estratégias de contraprogramação para {neg_factors_list}, que estão efetivamente "roubando" audiência da Globo.
                    
                    3. **Monitorar continuamente:** Estabelecer um sistema de monitoramento contínuo destes fatores para antecipar flutuações na audiência e adaptar a programação e estratégias de marketing de acordo.
                    
                    4. **Adaptar ao contexto econômico:** Considerando os indicadores econômicos com maior correlação, ajustar a grade de programação e esforços comerciais para maximizar a audiência conforme as condições do mercado.
                    """)
                else:
                    st.warning("Não foi possível calcular correlações significativas com os dados disponíveis.")
            else:
                st.warning("Não foram encontrados fatores externos para análise.")
        else:
            st.warning("Dados de audiência TV não estão disponíveis para análise.")

    # 9. Final notes - always show
    with st.expander("Informações sobre a análise de fatores externos"):
        st.markdown("""
        ### Fonte dos Dados

        **Indicadores Econômicos**: Os dados econômicos são obtidos de fontes oficiais como o Banco Central do Brasil e o IBGE.

        **Grade de Programação**: Os dados da grade são obtidos a partir das informações oficiais de programação de cada emissora.

        **Eventos Isolados**: Eventos pontuais identificados e categorizados manualmente com base em calendários e notícias relevantes.

        **Volume Social**: Os dados de volume social são obtidos via APIs de plataformas de redes sociais, com foco em termos relacionados à mídia e entretenimento.

        ### Considerações Metodológicas

        1. **Correlação não implica causalidade**: Embora identifiquemos correlações entre fatores externos e audiência, isso não necessariamente indica uma relação causal. Outros fatores não observados podem estar influenciando ambas as variáveis.

        2. **Limitações temporais**: A análise considera apenas o período coberto pelos dados disponíveis, que pode não representar todos os ciclos econômicos ou sazonalidades.

        3. **Competição por audiência**: Correlações negativas entre programas de emissoras concorrentes e a audiência da Globo indicam potencial "roubo" de audiência, mas é importante considerar que o universo total de telespectadores não é fixo.

        4. **Modelo linear**: O modelo explicativo integrado assume relações lineares entre fatores externos e audiência, o que pode não capturar completamente relações mais complexas.
        
        5. **Granularidade dos dados**: A análise horária permite maior precisão ao examinar o impacto imediato de eventos, enquanto a análise diária ou semanal captura tendências mais amplas.
        
        6. **Comparações válidas**: Para cada análise, consideramos apenas os períodos em que existem dados válidos para todas as variáveis envolvidas, garantindo que as comparações sejam consistentes e representativas.
        """)