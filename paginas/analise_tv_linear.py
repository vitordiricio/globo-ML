# utils/analise_tv_linear.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm

def analise_tv_linear(df):
    """
    Performs in-depth analysis of TV Linear data with focus on Globo's performance,
    showing audience metrics, trend analysis, seasonality and baseline measurements.
    
    Args:
        df (DataFrame): Processed dataframe with LINEAR_ prefixed columns
    """
    
    st.header("📺 TV Linear - Desempenho Histórico da Globo")
    
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
    
    # 2. Metrics Tables for each granularity
    st.subheader("Métricas Resumidas - TV Globo")

    st.markdown("""
    Abaixo estão as principais métricas da TV Linear Globo, organizadas pela granularidade 
    selecionada. Você pode escolher entre visualização diária ou semanal para entender diferentes 
    padrões temporais de audiência.
    """)
    
    # Granularity Selection Dropdown
    granularity_options = {
        "Diário": df_daily,
        "Semanal": df_weekly,
    }
    
    granularity = st.selectbox(
        "Selecione a granularidade:",
        options=list(granularity_options.keys())
    )
    
    # Get the selected dataframe
    selected_df = granularity_options[granularity]
    
    # Metric type selection
    metric_options = {
        "cov% (cobertura)": "LINEAR_GLOBO_cov%",
        "shr% (share)": "LINEAR_GLOBO_shr%",
        "TVR% (rating)": "LINEAR_GLOBO_tvr%"
    }
    
    selected_metric_type = st.selectbox(
        "Selecione o tipo de métrica para análise:",
        options=list(metric_options.keys())
    )
    
    selected_metric = metric_options[selected_metric_type]
    
    # Check if we have the required columns
    if selected_metric in selected_df.columns:
        # Create metrics table
        metrics_data = [{
            "Métrica": selected_metric_type,
            "Valor Médio": f"{selected_df[selected_metric].mean():.2f}",
            "Desvio Padrão": f"{selected_df[selected_metric].std():.2f}",
            "Número de Registros": f"{len(selected_df)}"
        }]
        
        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
    else:
        st.warning(f"A métrica {selected_metric_type} não está disponível nos dados.")
    
    # 3. Trend and Seasonality Analysis (Winter-Holt)
    st.subheader("Análise de Tendência e Sazonalidade (Winter-Holt)")
    
    st.markdown("""
    Esta análise utiliza o modelo de suavização exponencial Winter-Holt para 
    decompor a série temporal da audiência em componentes de tendência, sazonalidade e resíduos. 
    Isso permite entender padrões de longo prazo e variações cíclicas na audiência da Globo.
    """)
    
    if "LINEAR_GLOBO_cov%" in selected_df.columns and len(selected_df) >= 14:  # Need sufficient data points
        # Prepare data for time series analysis
        ts_data = selected_df.set_index('data_hora')['LINEAR_GLOBO_cov%']
        
        try:
            # Adjust seasonal_period based on granularity and data length
            if granularity == "Semanal":
                # For weekly data, use 4 (monthly seasonality) if we don't have enough for yearly
                if len(ts_data) >= 104:  # At least 2 years of data (104 weeks)
                    seasonal_period = 52  # Yearly seasonality
                elif len(ts_data) >= 8:  # At least 2 months of data (8 weeks)
                    seasonal_period = 4   # Monthly seasonality
                else:
                    raise ValueError("Insufficient data for seasonal analysis with weekly granularity")
            else:
                # For daily data, use 7 (weekly seasonality)
                seasonal_period = 7
            
            # Make sure index has frequency
            if granularity == "Semanal":
                ts_data = ts_data.asfreq('W-MON')  # Weekly frequency (Mondays)
            else:
                ts_data = ts_data.asfreq('D')  # Daily frequency
            
            # Fill missing values if any
            ts_data = ts_data.ffill().bfill()
            
            # Fit Holt-Winters model
            model = ExponentialSmoothing(
                ts_data,
                seasonal_periods=seasonal_period,
                trend='add',
                seasonal='add',
                use_boxcox=False,
                initialization_method="estimated"
            )
            
            fitted_model = model.fit()
            
            # Calculate R-squared
            predictions = fitted_model.fittedvalues
            r_squared = 1 - (((ts_data - predictions) ** 2).sum() / ((ts_data - ts_data.mean()) ** 2).sum())
            
            # Create decomposition manually
            # Extract the trend direction by looking at the slope of fitted values
            if len(predictions) >= 2:
                trend_start = predictions.iloc[0]
                trend_end = predictions.iloc[-1]
                trend_direction = trend_end - trend_start
                trend_daily_pp = trend_direction / len(predictions)
                
                # Convert to daily if weekly
                if granularity == "Semanal":
                    trend_daily_pp = trend_daily_pp / 7
                
                trend_label = "crescimento" if trend_daily_pp > 0 else "queda"
                
                # Create artificial seasonal pattern based on residuals
                residuals = ts_data - predictions
                
                if granularity == "Diário":
                    # Group residuals by day of week for daily data
                    ts_data_with_day = pd.DataFrame(ts_data)
                    ts_data_with_day['day'] = ts_data_with_day.index.dayofweek
                    ts_data_with_day['residual'] = residuals.values
                    
                    # Calculate average residual by day
                    seasonal_pattern = ts_data_with_day.groupby('day')['residual'].mean()
                    
                    # Map to day names
                    days = ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"]
                    dow_pattern = {days[i]: seasonal_pattern.get(i, 0) for i in range(7)}
                    
                    max_day = max(dow_pattern.items(), key=lambda x: x[1])[0]
                    min_day = min(dow_pattern.items(), key=lambda x: x[1])[0]
                elif granularity == "Semanal" and seasonal_period == 4:
                    # Group by month position (0-3) for monthly seasonality
                    ts_data_with_month = pd.DataFrame(ts_data)
                    ts_data_with_month['month_pos'] = (ts_data_with_month.index.isocalendar().week % 4)
                    ts_data_with_month['residual'] = residuals.values
                    
                    # Calculate average residual by month position
                    seasonal_pattern = ts_data_with_month.groupby('month_pos')['residual'].mean()
                    month_positions = ["Semana 1", "Semana 2", "Semana 3", "Semana 4"]
                    month_pattern = {month_positions[i]: seasonal_pattern.get(i, 0) for i in range(4)}
                    
                    max_month = max(month_pattern.items(), key=lambda x: x[1])[0]
                    min_month = min(month_pattern.items(), key=lambda x: x[1])[0]
                
                # Calculate forecast for next periods
                forecast_periods = 7 if granularity == "Diário" else 4
                forecast = fitted_model.forecast(forecast_periods)
                
                # Plot components
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create plot for original data, fitted values, and forecast
                    fig_data = go.Figure()
                    
                    # Add original data
                    fig_data.add_trace(go.Scatter(
                        x=ts_data.index, 
                        y=ts_data.values,
                        mode='lines',
                        name='Dados Originais',
                        line=dict(color='blue')
                    ))
                    
                    # Add fitted values
                    fig_data.add_trace(go.Scatter(
                        x=predictions.index, 
                        y=predictions.values,
                        mode='lines',
                        name='Modelo Ajustado',
                        line=dict(color='green')
                    ))
                    
                    # Add forecast
                    fig_data.add_trace(go.Scatter(
                        x=forecast.index, 
                        y=forecast.values,
                        mode='lines',
                        name='Previsão',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_data.update_layout(
                        title=f'Série Temporal com Modelo Ajustado (R² = {r_squared:.2f})',
                        xaxis_title='Data',
                        yaxis_title='cov% Globo',
                        legend=dict(orientation="h", y=1.1)
                    )
                    
                    st.plotly_chart(fig_data, use_container_width=True)
                
                with col2:
                    if granularity == "Diário":
                        # Create plot for seasonal component by day of week
                        seasonal_df = pd.DataFrame({
                            'Dia': days,
                            'Efeito Sazonal': [dow_pattern[day] for day in days]
                        })
                        
                        fig_seasonal = px.bar(
                            seasonal_df, 
                            x='Dia', 
                            y='Efeito Sazonal',
                            title='Padrão Sazonal por Dia da Semana',
                            color='Efeito Sazonal',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        
                        fig_seasonal.update_layout(
                            xaxis_title='Dia da Semana',
                            yaxis_title='Efeito na cov% (p.p.)'
                        )
                        
                        st.plotly_chart(fig_seasonal, use_container_width=True)
                    elif granularity == "Semanal" and seasonal_period == 4:
                        # Create plot for seasonal component by month position
                        seasonal_df = pd.DataFrame({
                            'Posição no Mês': list(month_pattern.keys()),
                            'Efeito Sazonal': list(month_pattern.values())
                        })
                        
                        fig_seasonal = px.bar(
                            seasonal_df, 
                            x='Posição no Mês', 
                            y='Efeito Sazonal',
                            title='Padrão Sazonal Mensal',
                            color='Efeito Sazonal',
                            color_continuous_scale=['red', 'yellow', 'green']
                        )
                        
                        fig_seasonal.update_layout(
                            xaxis_title='Posição no Mês',
                            yaxis_title='Efeito na cov% (p.p.)'
                        )
                        
                        st.plotly_chart(fig_seasonal, use_container_width=True)
                    else:
                        # For other cases, just show trend line
                        fig_trend = px.line(
                            pd.DataFrame({
                                'Data': predictions.index,
                                'Tendência': predictions.values
                            }), 
                            x='Data', 
                            y='Tendência',
                            title='Tendência da cov% ao Longo do Tempo',
                            markers=True
                        )
                        
                        fig_trend.update_layout(
                            xaxis_title='Data',
                            yaxis_title='cov% Tendência'
                        )
                        
                        st.plotly_chart(fig_trend, use_container_width=True)
                
                # Display key insights
                st.success(f"""
                **Resultados principais da análise temporal**:
                - **Melhor granularidade**: {granularity} (R² = {r_squared:.2f})
                - **Tendência atual**: {trend_label.title()} ({trend_daily_pp:.4f} p.p./dia)
                """)
                
                if granularity == "Diário" and 'max_day' in locals() and 'min_day' in locals():
                    st.success(f"""
                    **Padrão sazonal diário**:
                    - **Picos**: {max_day} (+{dow_pattern[max_day]:.2f} p.p.)
                    - **Quedas**: {min_day} ({dow_pattern[min_day]:.2f} p.p.)
                    """)
                elif granularity == "Semanal" and seasonal_period == 4 and 'max_month' in locals() and 'min_month' in locals():
                    st.success(f"""
                    **Padrão sazonal mensal**:
                    - **Picos**: {max_month} (+{month_pattern[max_month]:.2f} p.p.)
                    - **Quedas**: {min_month} ({month_pattern[min_month]:.2f} p.p.)
                    """)
            else:
                st.warning("Dados insuficientes para análise de tendência.")
        except Exception as e:
            st.error(f"Erro na análise de tendência e sazonalidade: {str(e)}")
            
            # If the error is due to insufficient data for seasonality, provide simplified analysis
            if "seasonal" in str(e).lower():
                st.info("""
                **Análise simplificada (sem sazonalidade):**
                
                Não há dados suficientes para identificar um padrão sazonal completo. 
                Vamos mostrar apenas a análise de tendência linear.
                """)
                
                try:
                    # Simple linear trend analysis
                    ts_data_reset = ts_data.reset_index()
                    ts_data_reset['time_idx'] = range(len(ts_data_reset))
                    
                    X = sm.add_constant(ts_data_reset['time_idx'])
                    y = ts_data_reset['LINEAR_GLOBO_cov%']
                    
                    trend_model = sm.OLS(y, X).fit()
                    
                    # Get trend coefficient and calculate daily change
                    trend_coef = trend_model.params[1]
                    daily_trend = trend_coef if granularity == "Diário" else trend_coef / 7
                    trend_label = "crescimento" if daily_trend > 0 else "queda"
                    
                    # Calculate fitted values and R-squared
                    trend_values = trend_model.predict(X)
                    r_squared = trend_model.rsquared
                    
                    # Create forecast
                    forecast_periods = 7 if granularity == "Diário" else 4
                    last_idx = ts_data_reset['time_idx'].max()
                    
                    forecast_indices = list(range(last_idx + 1, last_idx + 1 + forecast_periods))
                    forecast_dates = [ts_data_reset['data_hora'].iloc[-1] + pd.Timedelta(days=i) 
                                     for i in range(1, forecast_periods + 1)]
                    
                    forecast_X = sm.add_constant(forecast_indices)
                    forecast_values = trend_model.predict(forecast_X)
                    
                    # Plot trend and forecast
                    fig_trend = go.Figure()
                    
                    # Add original data
                    fig_trend.add_trace(go.Scatter(
                        x=ts_data_reset['data_hora'], 
                        y=ts_data_reset['LINEAR_GLOBO_cov%'],
                        mode='markers+lines',
                        name='Dados Originais',
                        line=dict(color='blue')
                    ))
                    
                    # Add trend line
                    fig_trend.add_trace(go.Scatter(
                        x=ts_data_reset['data_hora'], 
                        y=trend_values,
                        mode='lines',
                        name='Tendência Linear',
                        line=dict(color='green')
                    ))
                    
                    # Add forecast
                    fig_trend.add_trace(go.Scatter(
                        x=forecast_dates, 
                        y=forecast_values,
                        mode='lines',
                        name='Previsão',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_trend.update_layout(
                        title=f'Análise de Tendência Linear (R² = {r_squared:.2f})',
                        xaxis_title='Data',
                        yaxis_title='cov% Globo',
                        legend=dict(orientation="h", y=1.1)
                    )
                    
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    # Display key insights
                    st.success(f"""
                    **Resultados principais da análise de tendência**:
                    - **Granularidade**: {granularity} (R² = {r_squared:.2f})
                    - **Tendência atual**: {trend_label.title()} ({daily_trend:.4f} p.p./dia)
                    """)
                    
                    # Set variables for use in later sections
                    trend_daily_pp = daily_trend
                except Exception as e_inner:
                    st.error(f"Erro na análise de tendência simplificada: {str(e_inner)}")
    else:
        st.warning("Dados insuficientes para análise de tendência e sazonalidade.")
    
    # 4. Elasticity Analysis and Baseline Audience
    st.subheader("Análise de Elasticidade e Basal de Audiência")
    
    st.markdown("""
    Esta análise identifica o "chão" da audiência da Globo - o nível mínimo sustentado mesmo em 
    condições adversas. Este basal é estimado usando Regressão Quantílica em diferentes granularidades.
    """)
    
    if "LINEAR_GLOBO_cov%" in selected_df.columns and len(selected_df) >= 14:
        try:
            # Prepare data for quantile regression
            selected_df_qr = selected_df.copy()
            
            # Add time trend variable
            selected_df_qr['time_index'] = range(len(selected_df_qr))
            
            # Prepare data for statsmodels
            X = sm.add_constant(selected_df_qr['time_index'])
            y = selected_df_qr['LINEAR_GLOBO_cov%']
            
            # Fit quantile regression model at 10% quantile
            qr_model = QuantReg(y, X)
            qr_result = qr_model.fit(q=0.1)
            
            # Get coefficient for time trend (per period change)
            intercept = qr_result.params.iloc[0]  # Constant
            time_coef = qr_result.params.iloc[1]  # time_index coefficient
            
            # Calculate baseline for current period
            current_baseline = intercept + time_coef * selected_df_qr['time_index'].iloc[-1]
            
            # Convert coefficient to daily trend if weekly
            if granularity == "Semanal":
                daily_trend = time_coef / 7
            else:
                daily_trend = time_coef
            
            # Create baseline data for each granularity
            baseline_data = [
                {
                    "Granularidade": granularity,
                    "Basal Atual (cov%)": f"{current_baseline:.2f}%",
                    "Tendência do Basal (p.p./dia)": f"{daily_trend:.4f}"
                }
            ]
            
            baseline_df = pd.DataFrame(baseline_data)
            st.dataframe(baseline_df, hide_index=True, use_container_width=True)
            
            # Plot with baseline
            fig = go.Figure()
            
            # Add original data
            fig.add_trace(go.Scatter(
                x=selected_df_qr['data_hora'], 
                y=selected_df_qr['LINEAR_GLOBO_cov%'],
                mode='lines',
                name='cov% Globo',
                line=dict(color='blue')
            ))
            
            # Add baseline
            baseline_values = intercept + time_coef * selected_df_qr['time_index']
            fig.add_trace(go.Scatter(
                x=selected_df_qr['data_hora'], 
                y=baseline_values,
                mode='lines',
                name='Nível Basal (10%)',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title=f'cov% Globo com Nível Basal Estimado - {granularity}',
                xaxis_title='Data',
                yaxis_title='cov%',
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Check if current value is close to baseline
            current_cov = selected_df_qr['LINEAR_GLOBO_cov%'].iloc[-1]
            current_baseline_value = baseline_values.iloc[-1]
            
            proximity_threshold = 0.2  # 20% above baseline
            proximity_ratio = (current_cov - current_baseline_value) / current_baseline_value
            
            if proximity_ratio < proximity_threshold:
                st.warning(f"""
                ⚠️ **Alerta**: O cov% atual ({current_cov:.2f}%) está muito próximo do basal estimado ({current_baseline_value:.2f}%).
                Isso indica que a audiência está operando próxima do seu nível mínimo histórico.
                """)
            
            if daily_trend < 0:
                st.warning(f"""
                ⚠️ **Alerta**: A tendência do basal é de queda ({daily_trend:.4f} p.p./dia),
                sugerindo uma erosão gradual da audiência mínima ao longo do tempo.
                """)
            
        except Exception as e:
            st.error(f"Erro na análise de elasticidade e basal: {str(e)}")
    else:
        st.warning("Dados insuficientes para análise de elasticidade e basal.")
    
    # 5. Final Conclusions
    st.subheader("Conclusões Finais - TV Globo")
    
    if "LINEAR_GLOBO_cov%" in selected_df.columns:
        # Calculate key metrics
        avg_cov = selected_df['LINEAR_GLOBO_cov%'].mean()
        
        # Get R² from the trend analysis if available
        r_squared_value = r_squared if 'r_squared' in locals() else None
        
        # Get trend direction and value if available
        trend_daily_pp_value = trend_daily_pp if 'trend_daily_pp' in locals() else None
        
        # Get baseline value if available
        baseline_value = current_baseline if 'current_baseline' in locals() else None
        
        # Create metrics columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("cov% Médio Diário", f"{avg_cov:.2f}%")
            if r_squared_value is not None:
                st.metric("Melhor Granularidade", f"{granularity} (R² = {r_squared_value:.2f})")
            
        with col2:
            if trend_daily_pp_value is not None:
                trend_label = "Crescimento" if trend_daily_pp_value > 0 else "Queda"
                st.metric("Tendência", f"{trend_label} ({trend_daily_pp_value:.4f} p.p./dia)")
            
            if baseline_value is not None:
                st.metric("Basal Diário", f"{baseline_value:.2f}%")
    
    st.markdown("""
    ## Principais Insights
    
    - **Tendência de longo prazo:** A análise de Winter-Holt permite identificar a direção geral da audiência,
    separando flutuações de curto prazo de tendências estruturais.
    
    - **Padrão sazonal:** Os dias da semana têm impacto previsível na audiência, com diferenças significativas entre
    fim de semana e dias úteis, o que deve ser considerado no planejamento de conteúdo.
    
    - **Nível basal:** O piso de audiência calculado representa o público fiel da Globo que assiste independentemente
    de fatores externos, servindo como uma medida de resiliência da audiência.
    
    - **Alertas automáticos:** O sistema identifica quando a audiência está operando próxima ao nível basal ou quando
    há tendências consistentes de queda, permitindo intervenções proativas.
    
    Estes insights fornecem uma visão abrangente da performance histórica da Globo, permitindo decisões estratégicas
    melhor fundamentadas para manter e expandir a audiência.
    """)
    
    # 6. Notes and Documentation
    st.subheader("Notas Metodológicas")
    
    with st.expander("Informações sobre metodologia"):
        st.markdown("""
        ### Fonte dos Dados
        
        Os dados de audiência de TV Linear são coletados pela Kantar IBOPE Media.
        
        ### Cálculo do Basal de Audiência
        
        O nível basal é estimado utilizando **Regressão Quantílica** no percentil 10%, o que nos permite 
        identificar o patamar mínimo sustentável de audiência ao longo do tempo, mesmo em condições adversas.
        
        ### Análise de Tendência e Sazonalidade
        
        Utilizamos o modelo de **Winter-Holt** (Suavização Exponencial Tripla) para decompor a série temporal em três componentes:
        
        1. **Nível** - A base da série temporal
        2. **Tendência** - A direção e velocidade da mudança na audiência
        3. **Sazonalidade** - Padrões cíclicos recorrentes (diários ou semanais)
        
        O R² do modelo indica a qualidade do ajuste e, consequentemente, a confiabilidade da previsão.
        
        Para dados semanais com histórico limitado, utilizamos uma abordagem adaptativa que ajusta o período 
        sazonal com base na quantidade de dados disponíveis, ou recorre a uma análise de tendência linear quando 
        não há dados suficientes para detectar sazonalidade.
        """)