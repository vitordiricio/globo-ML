# utils/analise_tv_linear.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, r2_score


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

    # ================= MAIN ANALYSIS SECTIONS =================
    
    # Ensure we have enough data for analysis
    if "LINEAR_GLOBO_cov%" in selected_df.columns and len(selected_df) >= 14:
        # Prepare data for all models
        # Create a copy of the dataframe with data_hora as index
        analysis_df = selected_df.copy().set_index('data_hora')
        ts_data = analysis_df['LINEAR_GLOBO_cov%'].copy()
        
        # Ensure no missing values and proper frequency
        ts_data = ts_data.fillna(method='ffill').fillna(method='bfill')
        
        # Determine seasonal period based on granularity
        if granularity == "Semanal":
            if len(ts_data) >= 104:  # At least 2 years of data
                seasonal_period = 52  # Yearly seasonality
            else:
                seasonal_period = 4   # Monthly seasonality
        else:
            seasonal_period = 7  # Weekly seasonality for daily data

        # ================= SECTION 1: ARIMA MODEL =================
        st.subheader("1️⃣ Previsão de Tendência e Sazonalidade → ARIMA")
        st.markdown("""
        Esta análise utiliza o modelo ARIMA (AutoRegressive Integrated Moving Average) para 
        modelar a série temporal da audiência, permitindo identificar tendências e fazer previsões.
        """)
        
        try:
            # Prepare data for ARIMA
            model_data = pd.DataFrame({'y': ts_data})
            model_data = model_data.apply(pd.to_numeric, errors='coerce').fillna(0)
            
            # Fit ARIMA model with seasonal component
            arima_order = (1, 1, 1)  # p, d, q parameters
            seasonal_order = (1, 0, 1, seasonal_period)  # P, D, Q, s parameters
            
            # Use SARIMAX for seasonal modeling
            mod = sm.tsa.statespace.SARIMAX(
                model_data['y'], 
                order=arima_order,
                seasonal_order=seasonal_order,
                enforce_stationarity=True,
                enforce_invertibility=False,
                robust=True
            )
            
            fit_res = mod.fit(disp=False)
            
            # Make in-sample predictions
            predictions = fit_res.get_prediction().predicted_mean
            
            # Make sure predictions have same index as original data
            predictions = pd.Series(predictions, index=model_data.index)
            
            # Calculate metrics
            arima_r_squared = 1 - ((model_data['y'] - predictions) ** 2).sum() / ((model_data['y'] - model_data['y'].mean()) ** 2).sum()
            arima_mae = mean_absolute_error(model_data['y'], predictions)
            
            # Calculate forecast for next periods
            forecast_periods = 7 if granularity == "Diário" else 4
            
            # Create future date range
            if granularity == "Diário":
                future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=1), 
                                            periods=forecast_periods, freq='D')
            else:
                future_dates = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(days=7), 
                                            periods=forecast_periods, freq='W-MON')
            
            # Generate forecast
            forecast = fit_res.get_forecast(steps=forecast_periods).predicted_mean
            forecast = pd.Series(forecast, index=future_dates)
            
            # Calculate trend direction
            trend_start = predictions.iloc[0]
            trend_end = predictions.iloc[-1]
            trend_direction = trend_end - trend_start
            arima_daily_pp = trend_direction / len(predictions)
            
            # Convert to daily if weekly
            if granularity == "Semanal":
                arima_daily_pp = arima_daily_pp / 7
            
            trend_label = "crescimento" if arima_daily_pp > 0 else "queda"
            
            # Create plots
            # Time series plot with model fit and forecast
            fig_data = go.Figure()
            
            # Original data
            fig_data.add_trace(go.Scatter(
                x=model_data.index, 
                y=model_data['y'].values,
                mode='lines',
                name='Dados Originais',
                line=dict(color='blue')
            ))
            
            # Fitted values
            fig_data.add_trace(go.Scatter(
                x=predictions.index, 
                y=predictions.values,
                mode='lines',
                name='Modelo Ajustado',
                line=dict(color='green')
            ))
            
            # Forecast
            fig_data.add_trace(go.Scatter(
                x=forecast.index, 
                y=forecast.values,
                mode='lines',
                name='Previsão',
                line=dict(color='red', dash='dash')
            ))
            
            fig_data.update_layout(
                title=f'ARIMA (R² = {arima_r_squared:.2f}, MAE = {arima_mae:.2f})',
                xaxis_title='Data',
                yaxis_title='cov% Globo',
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_data, use_container_width=True)
            
            # Display summary
            st.success(f"""
            **Resultados principais da análise ARIMA**:
            - **Performance**: R² = {arima_r_squared:.2f}, MAE = {arima_mae:.2f}
            - **Tendência**: {trend_label.title()} ({arima_daily_pp:.4f} p.p./dia)
            """)
            
            # Save results for benchmarking
            arima_results = {
                'r2': arima_r_squared,
                'mae': arima_mae,
                'trend': arima_daily_pp
            }
            
        except Exception as e:
            st.error(f"Erro na análise ARIMA: {str(e)}")
            
            # Initialize an empty dict for results
            arima_results = {}

        # ================= SECTION 2: BASAL CALCULATION USING STATE-SPACE MODEL =================
        st.subheader("2️⃣ Cálculo do Basal → State-Space Model")
        st.markdown("""
        Esta análise identifica o "chão" da audiência da Globo - o nível mínimo sustentado mesmo em 
        condições adversas. Este basal é estimado usando um modelo de espaço de estados (State-Space Model)
        que permite capturar o nível latente da audiência.
        """)
        
        try:
            # Definimos um modelo local level mais simples para o State-Space Model
            # Este é mais estável e menos propenso a problemas dimensionais
            model_type = 'local level'
            
            # Ajustamos o modelo
            state_space_model = sm.tsa.UnobservedComponents(
                ts_data, 
                model_type,
                irregular=True  # Permite variância nos erros
            )
            
            fit_ssm = state_space_model.fit(disp=False)
            
            # Extrair o estado suavizado (nível da série)
            # Garantindo que acessamos as dimensões corretamente
            smoothed_states = fit_ssm.smoothed_state
            
            # Verificar se smoothed_states é um array 1D ou 2D
            if smoothed_states.ndim == 1:  # É um array 1D
                smoothed_level = smoothed_states
            else:  # É um array 2D
                smoothed_level = smoothed_states[0, :]  # Primeiro componente é o nível
            
            # Obter variância do estado para calcular intervalos
            state_cov = fit_ssm.smoothed_state_cov
            
            # Lidar com diferentes dimensões de state_cov
            if state_cov.ndim == 2:  # Formato simples para Local Level
                state_var = np.diagonal(state_cov)
            else:  # Formato mais complexo com múltiplos estados
                state_var = np.diagonal(state_cov[0, :, :])
            
            # Calcular o percentil 10% para o basal (simulando a distribuição)
            level_percentiles = np.zeros(len(smoothed_level))
            
            for i in range(len(smoothed_level)):
                # Verificar se a variância em i existe
                variance = state_var[i] if i < len(state_var) else state_var[-1]
                
                # Simular a distribuição Gaussiana do estado
                state_dist = np.random.normal(
                    loc=smoothed_level[i],
                    scale=np.sqrt(variance),
                    size=1000
                )
                # Calcular o percentil 10%
                level_percentiles[i] = np.percentile(state_dist, 10)
            
            # Criar DataFrame com resultados
            ssm_results = pd.DataFrame({
                'data_hora': ts_data.index,
                'cov_original': ts_data.values,
                'nivel_latente': smoothed_level,
                'basal_kalman': level_percentiles
            })
            
            # Calcular basal atual e tendência
            current_basal = level_percentiles[-1]
            
            # Calcular tendência (mudança diária)
            basal_trend = (level_percentiles[-1] - level_percentiles[0]) / len(level_percentiles)
            if granularity == "Semanal":
                basal_trend = basal_trend / 7  # Converter para diário se semanal
            
            basal_trend_label = "crescimento" if basal_trend > 0 else "queda"
            
            # Criar tabela de resultados
            basal_data = [
                {
                    "Método": "State-Space Model",
                    "Basal Atual (cov%)": f"{current_basal:.2f}%",
                    "Tendência do Basal (p.p./dia)": f"{basal_trend:.4f}"
                }
            ]
            
            basal_df = pd.DataFrame(basal_data)
            st.dataframe(basal_df, hide_index=True, use_container_width=True)
            
            # Criar gráfico
            fig_ssm = go.Figure()
            
            # Adicionar dados originais
            fig_ssm.add_trace(go.Scatter(
                x=ssm_results['data_hora'],
                y=ssm_results['cov_original'],
                mode='lines',
                name='cov% Original',
                line=dict(color='blue')
            ))
            
            # Adicionar nível latente
            fig_ssm.add_trace(go.Scatter(
                x=ssm_results['data_hora'],
                y=ssm_results['nivel_latente'],
                mode='lines',
                name='Nível Latente',
                line=dict(color='green')
            ))
            
            # Adicionar basal (percentil 10%)
            fig_ssm.add_trace(go.Scatter(
                x=ssm_results['data_hora'],
                y=ssm_results['basal_kalman'],
                mode='lines',
                name='Basal (Percentil 10%)',
                line=dict(color='red', dash='dash')
            ))
            
            fig_ssm.update_layout(
                title='Análise de Basal com Modelo de Espaço de Estados',
                xaxis_title='Data',
                yaxis_title='cov%',
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_ssm, use_container_width=True)
            
            # Verificar se valor atual está próximo do basal
            current_cov = ts_data.iloc[-1]
            proximity_threshold = 0.2  # 20% acima do basal
            proximity_ratio = (current_cov - current_basal) / current_basal
            
            if proximity_ratio < proximity_threshold:
                st.warning(f"""
                ⚠️ **Alerta**: O cov% atual ({current_cov:.2f}%) está muito próximo do basal estimado ({current_basal:.2f}%).
                Isso indica que a audiência está operando próxima do seu nível mínimo histórico.
                """)
            
            if basal_trend < 0:
                st.warning(f"""
                ⚠️ **Alerta**: A tendência do basal é de queda ({basal_trend:.4f} p.p./dia),
                sugerindo uma erosão gradual da audiência mínima ao longo do tempo.
                """)
            
            # Salvar resultados
            basal_results = {
                'basal': current_basal,
                'basal_trend': basal_trend
            }
            
        except Exception as e:
            st.error(f"Erro na análise de basal com State-Space Model: {str(e)}")
            
            # Alternativa com Regressão Quantílica em caso de erro
            st.info("Usando Regressão Quantílica como método alternativo...")
            
            try:
                # Preparar dados para regressão quantílica
                selected_df_qr = selected_df.copy()
                selected_df_qr['time_index'] = range(len(selected_df_qr))
                
                # Prepare data for statsmodels
                X = sm.add_constant(selected_df_qr['time_index'])
                y = selected_df_qr['LINEAR_GLOBO_cov%']
                
                # Fit quantile regression model at 10% quantile
                qr_model = QuantReg(y, X)
                qr_result = qr_model.fit(q=0.1)
                
                # Get coefficient for time trend
                intercept = qr_result.params.iloc[0]
                time_coef = qr_result.params.iloc[1]
                
                # Calculate baseline for current period
                current_baseline = intercept + time_coef * selected_df_qr['time_index'].iloc[-1]
                
                # Convert coefficient to daily trend if weekly
                if granularity == "Semanal":
                    daily_trend = time_coef / 7
                else:
                    daily_trend = time_coef
                
                # Create baseline data
                baseline_data = [
                    {
                        "Método": "Regressão Quantílica (alternativa)",
                        "Basal Atual (cov%)": f"{current_baseline:.2f}%",
                        "Tendência do Basal (p.p./dia)": f"{daily_trend:.4f}"
                    }
                ]
                
                baseline_df = pd.DataFrame(baseline_data)
                st.dataframe(baseline_df, hide_index=True, use_container_width=True)
                
                # Salvar resultados do método alternativo
                basal_results = {
                    'basal': current_baseline,
                    'basal_trend': daily_trend
                }
                
            except Exception as e_inner:
                st.error(f"Erro na análise alternativa: {str(e_inner)}")
                basal_results = {}

        # ================= SECTION 3: BENCHMARKING AND VALIDATION =================
        st.subheader("3️⃣ Benchmarking e Validação")
        st.markdown("""
        Esta seção compara o desempenho do modelo ARIMA com o modelo de referência Holt-Winters,
        permitindo verificar qual abordagem captura melhor os padrões da série temporal.
        """)
        
        try:
            # Try to fit Holt-Winters model
            holt_winters_model = ExponentialSmoothing(
                ts_data,
                seasonal_periods=seasonal_period,
                trend='add',
                seasonal='add',
                initialization_method="estimated"
            )
            
            fit_hw = holt_winters_model.fit()
            
            # Calculate metrics
            hw_predictions = fit_hw.fittedvalues
            hw_r_squared = 1 - ((ts_data - hw_predictions) ** 2).sum() / ((ts_data - ts_data.mean()) ** 2).sum()
            hw_mae = mean_absolute_error(ts_data, hw_predictions)
            
            # Extract trend direction
            hw_trend_start = hw_predictions.iloc[0]
            hw_trend_end = hw_predictions.iloc[-1]
            hw_trend_direction = hw_trend_end - hw_trend_start
            hw_daily_pp = hw_trend_direction / len(hw_predictions)
            
            # Convert to daily if weekly
            if granularity == "Semanal":
                hw_daily_pp = hw_daily_pp / 7
                
            # Generate Holt-Winters forecast
            forecast_periods = 7 if granularity == "Diário" else 4
            hw_forecast = fit_hw.forecast(forecast_periods)
            
            # Create comparison table
            comparison_data = [{
                "Modelo": "Holt-Winters",
                "R²": f"{hw_r_squared:.4f}",
                "MAE": f"{hw_mae:.4f}",
                "Tendência (p.p./dia)": f"{hw_daily_pp:.4f}"
            }]
            
            # Add ARIMA if available
            if 'r2' in arima_results:
                comparison_data.append({
                    "Modelo": "ARIMA",
                    "R²": f"{arima_results['r2']:.4f}",
                    "MAE": f"{arima_results['mae']:.4f}",
                    "Tendência (p.p./dia)": f"{arima_results['trend']:.4f}"
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, hide_index=True, use_container_width=True)
            
            # Plot comparison
            fig_comp = go.Figure()
            
            # Original data
            fig_comp.add_trace(go.Scatter(
                x=ts_data.index,
                y=ts_data.values,
                mode='lines',
                name='Dados Originais',
                line=dict(color='black')
            ))
            
            # Holt-Winters predictions
            fig_comp.add_trace(go.Scatter(
                x=hw_predictions.index,
                y=hw_predictions.values,
                mode='lines',
                name='Holt-Winters',
                line=dict(color='blue')
            ))
            
            # Holt-Winters forecast
            fig_comp.add_trace(go.Scatter(
                x=hw_forecast.index,
                y=hw_forecast.values,
                mode='lines',
                name='Forecast HW',
                line=dict(color='blue', dash='dash')
            ))
            
            # Add ARIMA if available
            if 'r2' in arima_results and 'predictions' in locals() and 'forecast' in locals():
                fig_comp.add_trace(go.Scatter(
                    x=predictions.index,
                    y=predictions.values,
                    mode='lines',
                    name='ARIMA',
                    line=dict(color='red')
                ))
                
                fig_comp.add_trace(go.Scatter(
                    x=forecast.index,
                    y=forecast.values,
                    mode='lines',
                    name='Forecast ARIMA',
                    line=dict(color='red', dash='dash')
                ))
            
            fig_comp.update_layout(
                title='Comparação entre Modelos',
                xaxis_title='Data',
                yaxis_title='cov%',
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_comp, use_container_width=True)
            
            # Determine winner
            if 'r2' in arima_results:
                if arima_results['r2'] > hw_r_squared:
                    winner_r2 = "ARIMA"
                    r2_diff = arima_results['r2'] - hw_r_squared
                else:
                    winner_r2 = "Holt-Winters"
                    r2_diff = hw_r_squared - arima_results['r2']
                
                if arima_results['mae'] < hw_mae:
                    winner_mae = "ARIMA"
                    mae_diff = hw_mae - arima_results['mae']
                else:
                    winner_mae = "Holt-Winters"
                    mae_diff = arima_results['mae'] - hw_mae
                
                st.success(f"""
                **Resultado da comparação**:
                - **R²**: {winner_r2} vence com diferença de {r2_diff:.4f}
                - **MAE**: {winner_mae} vence com diferença de {mae_diff:.4f}
                """)
            
            # Save Holt-Winters results
            holt_winters_results = {
                'r2': hw_r_squared,
                'mae': hw_mae,
                'trend': hw_daily_pp
            }
            
        except Exception as e:
            st.error(f"Erro na análise de benchmarking: {str(e)}")
            holt_winters_results = {}

        # ================= FINAL CONCLUSION =================
        st.subheader("Conclusões")
        
        # Calculate key metrics
        avg_cov = selected_df['LINEAR_GLOBO_cov%'].mean()
        
        # Determine best model based on benchmarking
        best_model = None
        if 'r2' in arima_results and 'r2' in holt_winters_results:
            if arima_results['r2'] > holt_winters_results['r2']:
                best_model = "ARIMA"
                best_r2 = arima_results['r2']
                best_trend = arima_results['trend']
            else:
                best_model = "Holt-Winters"
                best_r2 = holt_winters_results['r2']
                best_trend = holt_winters_results['trend']
        elif 'r2' in arima_results:
            best_model = "ARIMA"
            best_r2 = arima_results['r2']
            best_trend = arima_results['trend']
        elif 'r2' in holt_winters_results:
            best_model = "Holt-Winters"
            best_r2 = holt_winters_results['r2']
            best_trend = holt_winters_results['trend']
        
        # Get basal value
        basal_value = basal_results.get('basal', None)
        basal_trend = basal_results.get('basal_trend', None)
        
        # Create summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("cov% Médio", f"{avg_cov:.2f}%")
        
        with col2:
            if best_model:
                trend_label = "Crescimento" if best_trend > 0 else "Queda"
                st.metric("Tendência Geral", f"{trend_label} ({best_trend:.4f} p.p./dia)")
        
        with col3:
            if basal_value:
                basal_trend_label = "Crescimento" if basal_trend > 0 else "Queda"
                st.metric("Basal (10%)", f"{basal_value:.2f}% ({basal_trend_label})")
        
        st.markdown("""
        ## Principais Insights
        
        - **Modelo com melhor desempenho:** O benchmark indica qual abordagem (ARIMA ou Holt-Winters)
        captura melhor os padrões da audiência, permitindo previsões mais precisas.
        
        - **Componentes sazonais:** A análise de séries temporais identifica padrões cíclicos importantes
        que influenciam a audiência, revelando periodicidades diárias, semanais ou mensais.
        
        - **Nível basal:** A regressão quantílica revela o piso de audiência histórico,
        permitindo identificar quando a audiência está operando próxima ao seu mínimo sustentável.
        
        - **Alertas automáticos:** Os indicadores mostram quando há tendências preocupantes,
        como queda consistente do basal ou aproximação ao nível mínimo histórico.
        """)
    else:
        st.warning("Dados insuficientes para análise completa. São necessários pelo menos 14 registros com a métrica LINEAR_GLOBO_cov%.")