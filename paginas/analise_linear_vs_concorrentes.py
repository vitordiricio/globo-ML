# utils/analise_linear_vs_concorrentes.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm

def analise_linear_vs_concorrentes(df):
    """
    Performs comparative analysis of Globo versus its competitors (Record, SBT, Band, etc.),
    showing side-by-side metrics, correlation analysis, and evolution over time.
    
    Args:
        df (DataFrame): Processed dataframe with LINEAR_ prefixed columns for all channels
    """
    
    st.header("📊 TV Linear - Comparativo Globo vs Concorrentes")
    
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
    numeric_cols = df_daily.select_dtypes(include=['number']).columns.tolist()
    df_daily = df_daily.groupby('data')[numeric_cols].mean().reset_index()
    df_daily['data_hora'] = pd.to_datetime(df_daily['data'])
    
    # Create weekly aggregation
    df_weekly = df.copy()
    df_weekly['semana'] = df_weekly['data_hora'].dt.to_period('W').astype(str)
    numeric_cols = df_weekly.select_dtypes(include=['number']).columns.tolist()
    df_weekly = df_weekly.groupby('semana')[numeric_cols].mean().reset_index()
    df_weekly['data_hora'] = pd.to_datetime(df_weekly['semana'].str.split('/').str[0])
    
    st.markdown("""
    Esta análise compara o desempenho da Globo com seus principais concorrentes, permitindo visualizar:
    
    - Métricas médias e desvios padrão de cada emissora
    - Evolução histórica da Globo versus a média dos concorrentes
    - Correlação direta entre a Globo e concorrentes específicos
    - Sugestões de métricas relevantes para modelagem multivariada
    """)
    
    # Sidebar for selecting granularity and metric
    col1, col2 = st.columns(2)
    
    with col1:
        # Granularity Selection Dropdown
        granularity_options = {
            "Diário": df_daily,
            "Semanal": df_weekly,
        }
        
        granularity = st.selectbox(
            "Selecione a granularidade:",
            options=list(granularity_options.keys())
        )
    
    with col2:
        # Metric Selection Dropdown
        metric_options = {
            "cov% (cobertura)": "cov%",
            "shr% (share)": "shr%",
            "TVR% (rating)": "tvr%"
        }
        
        selected_metric_type = st.selectbox(
            "Selecione a métrica para comparação:",
            options=list(metric_options.keys())
        )
    
    # Get the selected dataframe and metric suffix
    selected_df = granularity_options[granularity]
    metric_suffix = metric_options[selected_metric_type]
    
    # Identify competitors (those with LINEAR_ prefix but not GLOBO)
    competitors = []
    for col in selected_df.columns:
        if col.startswith('LINEAR_') and metric_suffix in col and 'GLOBO' not in col:
            # Extract channel name from column name
            channel = col.split('_')[1]
            if channel not in competitors:
                competitors.append(channel)
    
    if not competitors:
        st.warning("Não foram encontrados dados de concorrentes para comparação.")
        return
    
    # 2. Metrics Table - Globo vs Competitors
    st.subheader("Métricas Resumidas - Globo vs Concorrentes")
    
    # Create comparison dataframe
    comparison_data = []
    
    # Add Globo data
    globo_col = f"LINEAR_GLOBO_{metric_suffix}"
    if globo_col in selected_df.columns:
        comparison_data.append({
            "Emissora": "Globo",
            "Métrica": selected_metric_type,
            "Média": selected_df[globo_col].mean(),
            "Desvio Padrão": selected_df[globo_col].std()
        })
    
    # Add competitors data
    for competitor in competitors:
        comp_col = f"LINEAR_{competitor}_{metric_suffix}"
        if comp_col in selected_df.columns:
            comparison_data.append({
                "Emissora": competitor,
                "Métrica": selected_metric_type,
                "Média": selected_df[comp_col].mean(),
                "Desvio Padrão": selected_df[comp_col].std()
            })
    
    # Create dataframe and format
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('Média', ascending=False)
    
    # Format for display
    display_df = comparison_df.copy()
    display_df['Média'] = display_df['Média'].map(lambda x: f"{x:.2f}")
    display_df['Desvio Padrão'] = display_df['Desvio Padrão'].map(lambda x: f"{x:.2f}")
    
    st.dataframe(display_df, hide_index=True, use_container_width=True)
    
    # 3. Aggregated Comparison - Globo vs Average Competitors
    st.subheader("Comparação Agregada - Globo vs Média Concorrentes")
    
    st.markdown("""
    Este gráfico compara a evolução da Globo com a média dos concorrentes ao longo do tempo.
    Quando a linha da Globo está acima, indica superioridade em audiência; quando está abaixo,
    sugere que os concorrentes estão, em média, performando melhor.
    """)
    
    if globo_col in selected_df.columns:
        # Calculate average of competitors for each timestamp
        comp_cols = [f"LINEAR_{comp}_{metric_suffix}" for comp in competitors if f"LINEAR_{comp}_{metric_suffix}" in selected_df.columns]
        
        if comp_cols:
            selected_df['Concorrentes (Média)'] = selected_df[comp_cols].mean(axis=1)
            
            # Create the comparison plot
            fig = go.Figure()
            
            # Add Globo trace
            fig.add_trace(go.Scatter(
                x=selected_df['data_hora'],
                y=selected_df[globo_col],
                mode='lines',
                name='Globo',
                line=dict(color='#0D47A1', width=3)  # Thick blue line for Globo
            ))
            
            # Add competitors average trace
            fig.add_trace(go.Scatter(
                x=selected_df['data_hora'],
                y=selected_df['Concorrentes (Média)'],
                mode='lines',
                name='Concorrentes (Média)',
                line=dict(color='#757575', width=2)  # Gray line for competitors
            ))
            
            fig.update_layout(
                title=f'Evolução do {selected_metric_type} - Globo vs Média dos Concorrentes ({granularity})',
                xaxis_title='Data',
                yaxis_title=selected_metric_type,
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation and market share
            corr = selected_df[globo_col].corr(selected_df['Concorrentes (Média)'])
            
            # Calculate correlation on first differences (stationary series)
            # Create differenced series and drop NaN values
            globo_diff = selected_df[globo_col].diff().dropna()
            competitors_diff = selected_df['Concorrentes (Média)'].diff().dropna()
            
            # Ensure indices match after dropping NaN values
            common_index = globo_diff.index.intersection(competitors_diff.index)
            corr_stationary = globo_diff.loc[common_index].corr(competitors_diff.loc[common_index])
            
            globo_avg = selected_df[globo_col].mean()
            competitors_avg = selected_df['Concorrentes (Média)'].mean()
            total_avg = globo_avg + competitors_avg
            
            if total_avg > 0:
                globo_share = (globo_avg / total_avg) * 100
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Correlação Globo vs Concorrentes",
                        f"{corr:.2f}",
                        help="Uma correlação próxima de 1 indica que Globo e concorrentes variam juntos, enquanto valores próximos de -1 indicam variação em direções opostas."
                    )
                
                with col2:
                    st.metric(
                        "Correlação Globo vs Concorrentes (Estacionário)",
                        f"{corr_stationary:.2f}",
                        help="Correlação calculada após aplicar a primeira diferença nas séries temporais, removendo tendências e sazonalidades."
                    )
                
                with col3:
                    st.metric(
                        "Participação Média da Globo",
                        f"{globo_share:.1f}%",
                        help="Percentual médio da Globo em relação ao total (Globo + Concorrentes)."
                    )
                
                # Interpretation based on correlation
                st.markdown("### Análise da Correlação Agregada")
                
                if corr > 0.7:
                    st.info("""
                    **Correlação forte positiva**: Globo e concorrentes tendem a ter variações muito semelhantes na audiência,
                    sugerindo que todos são afetados pelos mesmos fatores externos (como feriados, eventos especiais, padrões sazonais, etc).
                    """)
                elif corr > 0.3:
                    st.info("""
                    **Correlação moderada positiva**: Existe alguma tendência de variação similar entre Globo e concorrentes,
                    mas cada um também tem seus próprios padrões distintos de resposta ao mercado.
                    """)
                elif corr > -0.3:
                    st.info("""
                    **Correlação fraca**: Globo e concorrentes variam independentemente, sugerindo que atendem a públicos
                    diferentes ou que suas estratégias de programação têm efeitos distintos na audiência.
                    """)
                elif corr > -0.7:
                    st.info("""
                    **Correlação moderada negativa**: Quando a audiência da Globo sobe, a dos concorrentes tende a cair moderadamente,
                    e vice-versa, sugerindo algum nível de competição direta pela mesma audiência.
                    """)
                else:
                    st.info("""
                    **Correlação forte negativa**: Existe forte competição direta entre Globo e concorrentes. Quando um ganha audiência,
                    o outro perde, sugerindo grande sobreposição de público-alvo e conteúdo substituto.
                    """)
                
                # Add interpretation for stationary correlation
                st.markdown("### Análise da Correlação Estacionária")
                if abs(corr - corr_stationary) > 0.3:
                    st.info(f"""
                    **Diferença significativa entre correlações**: A correlação das séries estacionárias ({corr_stationary:.2f}) 
                    é bastante diferente da correlação das séries originais ({corr:.2f}), o que sugere que grande parte da 
                    correlação original era influenciada por tendências comuns ou sazonalidade, e não por uma relação causal direta.
                    """)
                else:
                    st.info(f"""
                    **Correlações similares**: A correlação das séries estacionárias ({corr_stationary:.2f}) é semelhante à 
                    correlação das séries originais ({corr:.2f}), o que reforça a robustez da relação identificada, indicando 
                    que as variações de curto prazo entre Globo e concorrentes mantêm padrão similar à tendência geral.
                    """)
        else:
            st.warning("Dados insuficientes de concorrentes para criar a comparação agregada.")
    else:
        st.warning(f"A métrica {selected_metric_type} não está disponível para a Globo.")
    
    # 4. Direct Comparison - Globo vs Specific Competitor
    st.subheader("Comparação Direta - Globo vs Concorrente Específico")
    
    st.markdown("""
    Selecione um concorrente específico para analisar sua relação direta com a Globo. 
    A análise mostra a evolução lado a lado, correlação e, quando aplicável, a equação 
    que relaciona as audiências das duas emissoras.
    """)
    
    # Dropdown to select specific competitor
    selected_competitor = st.selectbox(
        "Selecione um concorrente para comparação detalhada:",
        options=competitors
    )
    
    if selected_competitor:
        comp_col = f"LINEAR_{selected_competitor}_{metric_suffix}"
        
        if globo_col in selected_df.columns and comp_col in selected_df.columns:
            # Create direct comparison plot
            fig = go.Figure()
            
            # Add Globo trace
            fig.add_trace(go.Scatter(
                x=selected_df['data_hora'],
                y=selected_df[globo_col],
                mode='lines',
                name='Globo',
                line=dict(color='#0D47A1', width=3)  # Thick blue line for Globo
            ))
            
            # Add competitor trace
            fig.add_trace(go.Scatter(
                x=selected_df['data_hora'],
                y=selected_df[comp_col],
                mode='lines',
                name=selected_competitor,
                line=dict(color='#D32F2F', width=2)  # Red line for competitor
            ))
            
            fig.update_layout(
                title=f'Evolução do {selected_metric_type} - Globo vs {selected_competitor} ({granularity})',
                xaxis_title='Data',
                yaxis_title=selected_metric_type,
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate correlation between Globo and selected competitor
            corr = selected_df[globo_col].corr(selected_df[comp_col])
            
            # Calculate correlation on first differences (stationary series)
            # Create differenced series and drop NaN values
            globo_diff = selected_df[globo_col].diff().dropna()
            competitor_diff = selected_df[comp_col].diff().dropna()
            
            # Ensure indices match after dropping NaN values
            common_index = globo_diff.index.intersection(competitor_diff.index)
            corr_stationary = globo_diff.loc[common_index].corr(competitor_diff.loc[common_index])
            
            # Calculate average metrics
            globo_avg = selected_df[globo_col].mean()
            comp_avg = selected_df[comp_col].mean()
            total_avg = globo_avg + comp_avg
            
            if total_avg > 0:
                globo_share = (globo_avg / total_avg) * 100
                ratio = globo_avg / comp_avg if comp_avg > 0 else float('inf')
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        f"Correlação Globo vs {selected_competitor}",
                        f"{corr:.2f}",
                        help="Uma correlação próxima de 1 indica que as emissoras variam juntas, enquanto valores próximos de -1 indicam variação em direções opostas."
                    )
                
                with col2:
                    st.metric(
                        f"Correlação Globo vs {selected_competitor} (Estacionário)",
                        f"{corr_stationary:.2f}",
                        help="Correlação calculada após aplicar a primeira diferença nas séries temporais, removendo tendências e sazonalidades."
                    )
                
                with col3:
                    st.metric(
                        f"Participação da Globo vs {selected_competitor}",
                        f"{globo_share:.1f}%",
                        help=f"Percentual da Globo em relação ao total (Globo + {selected_competitor})."
                    )
                
                with col4:
                    st.metric(
                        "Proporção Globo/Concorrente",
                        f"{ratio:.2f}x",
                        help=f"Quantas vezes a audiência da Globo é maior que a do {selected_competitor}."
                    )
                
                # If correlation is strong, show regression equation
                if abs(corr) > 0.6:
                    st.markdown("### Equação de Regressão Linear")
                    
                    # Fit linear regression model
                    X = selected_df[comp_col].values.reshape(-1, 1)
                    X = sm.add_constant(X)  # Add constant term
                    y = selected_df[globo_col].values
                    
                    model = sm.OLS(y, X).fit()
                    
                    # Get coefficients
                    intercept = model.params[0]
                    slope = model.params[1]
                    
                    # Display equation
                    st.markdown(f"""
                    Como a correlação é {abs(corr):.2f}, forte o suficiente, podemos estabelecer uma 
                    relação direta entre a audiência das duas emissoras:
                    
                    **Globo {metric_suffix} = {intercept:.4f} + {slope:.4f} × {selected_competitor} {metric_suffix}**
                    
                    Esta equação indica que:
                    
                    - O valor base da Globo é {intercept:.4f} mesmo quando {selected_competitor} tem audiência zero
                    - Para cada 1 ponto no {metric_suffix} do {selected_competitor}, a Globo {('ganha' if slope > 0 else 'perde')} {abs(slope):.4f} pontos
                    """)
                    
                    # Create scatter plot with regression line
                    fig_scatter = px.scatter(
                        selected_df, 
                        x=comp_col, 
                        y=globo_col,
                        trendline="ols",
                        labels={
                            comp_col: f"{selected_competitor} {metric_suffix}",
                            globo_col: f"Globo {metric_suffix}"
                        },
                        title=f"Relação entre Globo e {selected_competitor}"
                    )
                    
                    st.plotly_chart(fig_scatter, use_container_width=True)
                
                # Interpretation based on correlation
                st.markdown(f"### Análise da Correlação com {selected_competitor}")
                
                if corr > 0.7:
                    st.info(f"""
                    **Correlação forte positiva**: Globo e {selected_competitor} tendem a ter variações muito semelhantes na audiência,
                    sugerindo que ambos são afetados pelos mesmos fatores ou atendem a públicos similares em momentos semelhantes.
                    """)
                elif corr > 0.3:
                    st.info(f"""
                    **Correlação moderada positiva**: Existe algum nível de similaridade entre os padrões de audiência da Globo e do {selected_competitor},
                    mas cada um também tem seus momentos únicos de pico ou queda.
                    """)
                elif corr > -0.3:
                    st.info(f"""
                    **Correlação fraca**: Globo e {selected_competitor} parecem operar com certa independência, atendendo a públicos
                    diferentes ou com estratégias de programação que resultam em padrões distintos de audiência.
                    """)
                elif corr > -0.7:
                    st.info(f"""
                    **Correlação moderada negativa**: Existe uma tendência de que quando a audiência da Globo aumenta, a do {selected_competitor} diminui,
                    e vice-versa, sugerindo competição direta pelo mesmo público em determinados horários.
                    """)
                else:
                    st.info(f"""
                    **Correlação forte negativa**: Há uma clara relação inversa entre as audiências da Globo e do {selected_competitor},
                    indicando forte competição pelo mesmo público e conteúdos altamente substitutos.
                    """)
                
                # Add interpretation for stationary correlation
                st.markdown(f"### Análise da Correlação Estacionária com {selected_competitor}")
                if abs(corr - corr_stationary) > 0.3:
                    st.info(f"""
                    **Diferença significativa entre correlações**: A correlação das séries estacionárias ({corr_stationary:.2f}) 
                    é bastante diferente da correlação das séries originais ({corr:.2f}), o que sugere que grande parte da 
                    correlação original era influenciada por tendências comuns ou sazonalidade, e não por uma relação causal direta 
                    entre Globo e {selected_competitor}.
                    """)
                else:
                    st.info(f"""
                    **Correlações similares**: A correlação das séries estacionárias ({corr_stationary:.2f}) é semelhante à 
                    correlação das séries originais ({corr:.2f}), o que reforça a robustez da relação identificada, indicando 
                    que as variações de curto prazo entre Globo e {selected_competitor} mantêm padrão similar à tendência geral.
                    """)
            else:
                st.warning("Dados insuficientes para calcular métricas comparativas.")
        else:
            st.warning(f"A métrica {selected_metric_type} não está disponível para Globo ou {selected_competitor}.")
    
    # 5. Suggested External Metrics for Multivariate Modeling
    st.subheader("Sugestão de Métricas Externas Relevantes para Modelagem Multivariada")
    
    st.markdown("""
    Com base nas correlações detectadas, estas são as métricas de concorrentes que podem ser mais relevantes 
    para incluir em modelos multivariados que buscam explicar ou prever a audiência da Globo.
    """)
    
    if globo_col in selected_df.columns and competitors:
        # Calculate correlations between Globo and each competitor for different metrics
        correlation_data = []
        
        for competitor in competitors:
            for metric in ['cov%', 'shr%', 'tvr%']:
                comp_col = f"LINEAR_{competitor}_{metric}"
                
                if comp_col in selected_df.columns:
                    corr = selected_df[globo_col].corr(selected_df[comp_col])
                    
                    correlation_data.append({
                        'Concorrente': competitor,
                        'Métrica Relevante': metric,
                        'Correlação com cov% Globo': corr
                    })
        
        # Sort by absolute correlation value
        if correlation_data:
            correlation_df = pd.DataFrame(correlation_data)
            correlation_df['Correlação Abs'] = correlation_df['Correlação com cov% Globo'].abs()
            correlation_df = correlation_df.sort_values('Correlação Abs', ascending=False).drop('Correlação Abs', axis=1)
            
            # Take top correlations (up to 5)
            top_correlations = correlation_df.head(5)
            
            # Format for display
            display_corr_df = top_correlations.copy()
            display_corr_df['Correlação com cov% Globo'] = display_corr_df['Correlação com cov% Globo'].map(lambda x: f"{x:.2f}")
            
            st.dataframe(display_corr_df, hide_index=True, use_container_width=True)
            
            # Visualization of top correlations
            fig_corr = px.bar(
                top_correlations,
                x='Concorrente',
                y='Correlação com cov% Globo',
                color='Correlação com cov% Globo',
                color_continuous_scale=['red', 'white', 'green'],
                title="Correlações mais Significativas com cov% Globo"
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # List top positive and negative correlations
            top_positive = correlation_df[correlation_df['Correlação com cov% Globo'] > 0].head(1)
            top_negative = correlation_df[correlation_df['Correlação com cov% Globo'] < 0].head(1)
            
            if not top_positive.empty:
                positive_corr = top_positive.iloc[0]
                st.success(f"""
                ✅ **Concorrente mais correlacionado positivamente**: {positive_corr['Concorrente']} ({positive_corr['Métrica Relevante']})
                com correlação de {positive_corr['Correlação com cov% Globo']:.2f}
                """)
            
            if not top_negative.empty:
                negative_corr = top_negative.iloc[0]
                st.error(f"""
                ✅ **Concorrente mais correlacionado negativamente**: {negative_corr['Concorrente']} ({negative_corr['Métrica Relevante']})
                com correlação de {negative_corr['Correlação com cov% Globo']:.2f}
                """)
        else:
            st.warning("Dados insuficientes para calcular correlações entre métricas.")
    else:
        st.warning("Dados insuficientes para recomendar métricas externas.")
    
    # 6. Final Conclusions
    st.subheader("Conclusões Finais - Comparação com Concorrentes")
    
    st.markdown("""
    ## Principais Insights
    
    - **Posicionamento relativo:** A análise revela a posição da Globo em relação aos concorrentes,
    tanto em valores médios quanto em tendências temporais.
    
    - **Correlações importantes:** Identificamos quais concorrentes apresentam maior correlação
    (positiva ou negativa) com a audiência da Globo, o que é fundamental para entender dinâmicas de mercado.
    
    - **Correlações estacionárias:** A análise das séries temporais após aplicar a primeira diferença
    permite entender relações mais robustas entre as emissoras, removendo o efeito de tendências comuns.
    
    - **Equações preditivas:** Para correlações fortes, estabelecemos equações lineares que
    permitem estimar o comportamento da audiência da Globo com base em concorrentes específicos.
    
    - **Sugestões para modelagem:** As métricas mais relevantes de concorrentes foram identificadas
    para inclusão em modelos multivariados, possibilitando análises mais robustas.
    
    Estes insights complementam a análise isolada da Globo, permitindo um entendimento mais
    completo do mercado de TV Linear e das dinâmicas competitivas que influenciam a audiência.
    """)
    
    with st.expander("Observação Metodológica"):
        st.markdown("""
        ### Importância da Análise Competitiva
        
        A análise comparativa com concorrentes é especialmente valiosa por:
        
        1. **Contextualização:** Entender se tendências observadas na Globo são específicas da emissora ou 
        refletem movimentos do mercado como um todo
        
        2. **Identificação de padrões de substituição:** Detectar se a audiência migra entre canais ou 
        simplesmente deixa de assistir TV Linear
        
        3. **Modelagem mais precisa:** Incluir métricas de concorrentes pode melhorar significativamente 
        a precisão de modelos preditivos para a audiência da Globo
        
        4. **Análise estacionária:** A correlação em séries estacionárias (após diferenciação) é essencial 
        para eliminar relações espúrias e identificar relações causais mais confiáveis entre variáveis
        
        A separação desta análise em uma aba específica permite um foco metodológico claro, 
        facilitando tanto a interpretação dos dados quanto a aplicação dos insights em estratégias 
        de programação e posicionamento competitivo.
        """)