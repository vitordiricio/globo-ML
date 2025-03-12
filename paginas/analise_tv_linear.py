# utils/analise_tv_linear.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.regression.quantile_regression import QuantReg
import statsmodels.api as sm
import numpy as np
from scipy.fft import rfft, rfftfreq
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.structural import UnobservedComponents
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
    
    # 3. Trend and Seasonality Analysis (Fourier + ARIMA)
    st.subheader("Análise de Tendência e Sazonalidade (Fourier + ARIMA)")
    
    st.markdown("""
    Esta análise utiliza uma combinação de Transformada de Fourier e modelo ARIMA para 
    decompor a série temporal da audiência em componentes. 
    
    A Transformada de Fourier identifica os padrões cíclicos mais relevantes na audiência, 
    enquanto o ARIMA modela a tendência e os resíduos após remover a sazonalidade. Esta
    abordagem permite capturar padrões complexos na audiência.
    """)
    
    if "LINEAR_GLOBO_cov%" in selected_df.columns and len(selected_df) >= 14:  # Need sufficient data points
        # Prepare data for time series analysis
        ts_data = selected_df.set_index('data_hora')['LINEAR_GLOBO_cov%']
        
        try:
            # Make sure index has frequency
            if granularity == "Semanal":
                ts_data = ts_data.asfreq('W-MON')  # Weekly frequency (Mondays)
            else:
                ts_data = ts_data.asfreq('D')  # Daily frequency
            
            # Fill missing values if any
            ts_data = ts_data.ffill().bfill()
            
            # Define helper function for Fourier features
            def create_fourier_features(y, num_periods, harmonics=3):
                """
                Create Fourier features based on the dominant frequencies
                
                Args:
                    y: Time series data
                    num_periods: List of periods (e.g., [7, 30, 365] for daily, monthly, yearly patterns)
                    harmonics: Number of harmonics to include for each period
                    
                Returns:
                    DataFrame with Fourier features
                """
                n = len(y)
                fourier_df = pd.DataFrame(index=y.index)
                
                for period in num_periods:
                    for harmonic in range(1, harmonics + 1):
                        sin_name = f'sin_{period}_{harmonic}'
                        cos_name = f'cos_{period}_{harmonic}'
                        
                        # Create sine and cosine features for each period/harmonic
                        fourier_df[sin_name] = np.sin(2 * np.pi * harmonic * np.arange(n) / period)
                        fourier_df[cos_name] = np.cos(2 * np.pi * harmonic * np.arange(n) / period)
                        
                return fourier_df
                
            # Use FFT to identify dominant frequencies
            def find_dominant_frequencies(y, top_n=3):
                """
                Use Fast Fourier Transform to identify dominant frequencies
                
                Args:
                    y: Time series data
                    top_n: Number of dominant frequencies to return
                    
                Returns:
                    List of dominant periods
                """
                n = len(y)
                y_fft = rfft(y.values)
                freqs = rfftfreq(n)
                
                # Exclude the first frequency (DC component)
                power = np.abs(y_fft[1:]) ** 2
                idx = np.argsort(power)[-top_n:]
                dominant_freqs = freqs[idx + 1]  # +1 because we excluded DC
                
                # Convert frequencies to periods
                dominant_periods = [int(round(1 / freq)) for freq in dominant_freqs if freq > 0]
                
                return dominant_periods
            
            # Setup for Fourier + ARIMA analysis
            # Determine periods based on granularity
            if granularity == "Diário":
                # For daily data, manually specify periods (daily=1, weekly=7, monthly=30/31)
                # Using predetermined periods instead of FFT for more interpretable results
                periods = [7, 30]  # Weekly and monthly patterns
            else:
                # For weekly data, look for monthly and quarterly patterns
                periods = [4, 13]  # 4 weeks in a month, ~13 weeks in a quarter
            
            # Also detect dominant frequencies from the data using FFT
            fft_periods = find_dominant_frequencies(ts_data)
            
            # Combine manual periods with detected periods, remove duplicates
            all_periods = list(set(periods + fft_periods))
            all_periods.sort()
            
            # Create Fourier features
            fourier_df = create_fourier_features(ts_data, all_periods, harmonics=2)
            
            # Combine time series data with Fourier features
            data_with_features = pd.DataFrame({'y': ts_data})
            data_with_features = pd.concat([data_with_features, fourier_df], axis=1)
            
            # Split data for fitting and validation
            train_size = int(len(data_with_features) * 0.8)
            train_data = data_with_features.iloc[:train_size]
            test_data = data_with_features.iloc[train_size:]
            
            # Build ARIMA model with Fourier features
            # Start with a simple model (p=1, d=1, q=1) and adjust if needed
            exog_train = train_data.drop('y', axis=1)
            exog_test = test_data.drop('y', axis=1)
            
            # Fit ARIMA model
            arima_model = ARIMA(
                train_data['y'],
                exog=exog_train,
                order=(1, 1, 1)  # Simple ARIMA model with Fourier exogenous variables
            )
            
            arima_fit = arima_model.fit()
            
            # Predict on train data
            fitted_values = arima_fit.fittedvalues
            
            # Predict on test data
            forecast = arima_fit.forecast(steps=len(test_data), exog=exog_test)
            
            # Calculate metrics
            train_r2 = r2_score(train_data['y'].iloc[1:], fitted_values)
            test_r2 = r2_score(test_data['y'], forecast)
            train_mae = mean_absolute_error(train_data['y'].iloc[1:], fitted_values)
            test_mae = mean_absolute_error(test_data['y'], forecast)
            
            # Refit on full data for final model
            full_model = ARIMA(
                data_with_features['y'],
                exog=data_with_features.drop('y', axis=1),
                order=(1, 1, 1)
            )
            full_fit = full_model.fit()
            
            # Create future Fourier features for forecasting
            forecast_periods = 14 if granularity == "Diário" else 8  # 2 weeks or 8 weeks
            
            # Create a date range for future periods
            future_dates = pd.date_range(
                start=data_with_features.index[-1] + pd.Timedelta('1 day' if granularity == "Diário" else '1 week'),
                periods=forecast_periods, 
                freq='D' if granularity == "Diário" else 'W-MON'
            )
            
            n_total = len(data_with_features) + forecast_periods
            future_fourier = pd.DataFrame(index=future_dates)
            
            # Generate Fourier features for future periods
            for period in all_periods:
                for harmonic in range(1, 3):
                    idx_offset = len(data_with_features)
                    sin_name = f'sin_{period}_{harmonic}'
                    cos_name = f'cos_{period}_{harmonic}'
                    
                    future_fourier[sin_name] = np.sin(2 * np.pi * harmonic * np.arange(idx_offset, n_total) / period)
                    future_fourier[cos_name] = np.cos(2 * np.pi * harmonic * np.arange(idx_offset, n_total) / period)
            
            # Make forecast
            future_forecast = full_fit.forecast(steps=forecast_periods, exog=future_fourier)
            
            # Combine original data with forecast for plotting
            fitted_with_forecast = pd.Series(
                index=pd.concat([pd.Series(index=data_with_features.index), pd.Series(index=future_dates)]).index,
                dtype=float
            )
            
            # Fill with fitted values and forecast
            fitted_with_forecast.loc[data_with_features.index] = full_fit.fittedvalues
            fitted_with_forecast.loc[future_dates] = future_forecast
            
            # Plot results
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
                    x=data_with_features.index, 
                    y=full_fit.fittedvalues,
                    mode='lines',
                    name='Modelo Ajustado',
                    line=dict(color='green')
                ))
                
                # Add forecast
                fig_data.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_forecast,
                    mode='lines',
                    name='Previsão',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_data.update_layout(
                    title=f'Modelo Fourier+ARIMA (R² = {train_r2:.2f})',
                    xaxis_title='Data',
                    yaxis_title='cov% Globo',
                    legend=dict(orientation="h", y=1.1)
                )
                
                st.plotly_chart(fig_data, use_container_width=True)
            
            with col2:
                # Plot the seasonal components
                # Compute the contribution of each period to the overall seasonality
                seasonal_contributions = {}
                
                # For each period, compute its contribution
                for period in all_periods:
                    # Sum the contribution of all harmonics for this period
                    period_contribution = np.zeros(len(ts_data))
                    
                    for harmonic in range(1, 3):
                        sin_name = f'sin_{period}_{harmonic}'
                        cos_name = f'cos_{period}_{harmonic}'
                        
                        # Get the coefficients from the ARIMA model
                        sin_coef = full_fit.params.get(sin_name, 0)
                        cos_coef = full_fit.params.get(cos_name, 0)
                        
                        # Compute the contribution
                        sin_contrib = sin_coef * fourier_df[sin_name].values
                        cos_contrib = cos_coef * fourier_df[cos_name].values
                        
                        period_contribution += sin_contrib + cos_contrib
                    
                    # Store the mean absolute contribution for this period
                    seasonal_contributions[period] = np.mean(np.abs(period_contribution))
                
                # Create a dataframe for plotting
                periods_df = pd.DataFrame({
                    'Período': [f"{p} dias" if granularity == "Diário" else f"{p} semanas" for p in seasonal_contributions.keys()],
                    'Contribuição': list(seasonal_contributions.values())
                })
                
                # Sort by contribution
                periods_df = periods_df.sort_values('Contribuição', ascending=False)
                
                # Plot
                fig_seasonal = px.bar(
                    periods_df, 
                    x='Período', 
                    y='Contribuição',
                    title='Componentes Sazonais Identificados',
                    color='Contribuição',
                    color_continuous_scale=['yellow', 'orange', 'red']
                )
                
                fig_seasonal.update_layout(
                    xaxis_title='Período Sazonal',
                    yaxis_title='Contribuição (Amplitude)'
                )
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
            # Also add a Holt-Winters model for comparison
            
            # Define seasonal period based on granularity
            if granularity == "Semanal":
                # For weekly data, use 4 (monthly seasonality) or 13 (quarterly seasonality)
                if len(ts_data) >= 104:  # At least 2 years of data (104 weeks)
                    seasonal_period = 52  # Yearly seasonality
                elif len(ts_data) >= 8:  # At least 2 months of data (8 weeks)
                    seasonal_period = 4   # Monthly seasonality
                else:
                    seasonal_period = 2   # Use minimal seasonality
            else:
                # For daily data, use 7 (weekly seasonality)
                seasonal_period = 7
            
            # Fit Holt-Winters model for comparison
            hw_model = ExponentialSmoothing(
                ts_data,
                seasonal_periods=seasonal_period,
                trend='add',
                seasonal='add',
                use_boxcox=False,
                initialization_method="estimated"
            )
            
            try:
                hw_fit = hw_model.fit()
                
                # Generate HW forecast
                hw_forecast = hw_fit.forecast(steps=forecast_periods)
                
                # Calculate HW metrics
                hw_fitted = hw_fit.fittedvalues
                hw_r2 = 1 - (((ts_data - hw_fitted) ** 2).sum() / ((ts_data - ts_data.mean()) ** 2).sum())
                hw_mae = mean_absolute_error(ts_data, hw_fitted)
                
                # Create comparison metrics table
                comparison_data = {
                    "Métrica": ["R² (treino)", "MAE (treino)"],
                    "Fourier+ARIMA": [f"{train_r2:.3f}", f"{train_mae:.3f}"],
                    "Holt-Winters": [f"{hw_r2:.3f}", f"{hw_mae:.3f}"]
                }
                
                comparison_df = pd.DataFrame(comparison_data)
                
                st.subheader("Comparação de Modelos")
                st.dataframe(comparison_df, hide_index=True, use_container_width=True)
                
                # Create combined forecast plot
                fig_compare = go.Figure()
                
                # Add original data
                fig_compare.add_trace(go.Scatter(
                    x=ts_data.index, 
                    y=ts_data.values,
                    mode='lines',
                    name='Dados Originais',
                    line=dict(color='blue')
                ))
                
                # Add Fourier+ARIMA forecast
                fig_compare.add_trace(go.Scatter(
                    x=future_dates, 
                    y=future_forecast,
                    mode='lines',
                    name='Previsão Fourier+ARIMA',
                    line=dict(color='red', dash='dash')
                ))
                
                # Add Holt-Winters forecast
                fig_compare.add_trace(go.Scatter(
                    x=future_dates, 
                    y=hw_forecast.values,
                    mode='lines',
                    name='Previsão Holt-Winters',
                    line=dict(color='green', dash='dash')
                ))
                
                fig_compare.update_layout(
                    title='Comparação das Previsões: Fourier+ARIMA vs. Holt-Winters',
                    xaxis_title='Data',
                    yaxis_title='cov% Globo',
                    legend=dict(orientation="h", y=1.1)
                )
                
                st.plotly_chart(fig_compare, use_container_width=True)
                
                # Calculate trend for insight generation
                # Use the Fourier+ARIMA model for trend extraction
                trend_direction = future_forecast[-1] - future_forecast[0]
                trend_daily_pp = trend_direction / len(future_forecast)
                
                # Convert to daily if weekly
                if granularity == "Semanal":
                    trend_daily_pp = trend_daily_pp / 7
                
                trend_label = "crescimento" if trend_daily_pp > 0 else "queda"
                
                # Display key insights
                st.success(f"""
                **Resultados principais da análise temporal**:
                - **Melhor modelo**: {"Fourier+ARIMA" if train_r2 > hw_r2 else "Holt-Winters"} (R² = {max(train_r2, hw_r2):.2f})
                - **Tendência atual**: {trend_label.title()} ({trend_daily_pp:.4f} p.p./dia)
                - **Principais períodos sazonais**: {", ".join([f"{p}" for p in list(periods_df['Período'])[:2]])}
                """)
                
            except Exception as hw_error:
                st.warning(f"Não foi possível comparar com Holt-Winters: {str(hw_error)}")
                
                # Still display insights from Fourier+ARIMA
                trend_direction = future_forecast[-1] - future_forecast[0]
                trend_daily_pp = trend_direction / len(future_forecast)
                
                # Convert to daily if weekly
                if granularity == "Semanal":
                    trend_daily_pp = trend_daily_pp / 7
                
                trend_label = "crescimento" if trend_daily_pp > 0 else "queda"
                
                st.success(f"""
                **Resultados principais da análise temporal**:
                - **Fourier+ARIMA**: R² = {train_r2:.2f}
                - **Tendência atual**: {trend_label.title()} ({trend_daily_pp:.4f} p.p./dia)
                - **Principais períodos sazonais**: {", ".join([f"{p}" for p in list(periods_df['Período'])[:2]])}
                """)
                
        except Exception as e:
            st.error(f"Erro na análise de Fourier + ARIMA: {str(e)}")
            
            # If there's an error, try a simpler approach
            st.info("""
            **Análise simplificada:**
            
            Encontramos um problema técnico. Vamos mostrar uma análise de tendência simples.
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
    
    # 4. State-Space Model for Baseline Analysis
    st.subheader("Análise de Nível Basal (Modelo de Espaço de Estados)")
    
    st.markdown("""
    Esta análise identifica o "chão" da audiência da Globo - o nível mínimo sustentado mesmo em 
    condições adversas. Este basal é estimado usando um Modelo de Espaço de Estados com Filtro
    de Kalman, que considera a natureza dinâmica da audiência ao longo do tempo.
    """)
    
    if "LINEAR_GLOBO_cov%" in selected_df.columns and len(selected_df) >= 14:
        try:
            # Prepare data for state-space model
            ss_data = selected_df.set_index('data_hora')['LINEAR_GLOBO_cov%']
            
            # Fill any missing values
            ss_data = ss_data.ffill().bfill()
            
            # Create a state-space model using UnobservedComponents
            # We'll use a local linear trend model with stochastic level and trend
            ss_model = UnobservedComponents(
                ss_data,
                level='local linear trend',  # Stochastic level and trend
                stochastic_level=True,
                stochastic_trend=True
            )
            
            ss_fit = ss_model.fit(disp=False)
            
            # Extract the smoothed states
            smoothed_states = ss_fit.smoothed_state
            
            # The first component is the level (latent state)
            smoothed_level = smoothed_states[0]
            
            # The second component is the trend
            smoothed_trend = smoothed_states[1]
            
            # Calculate baseline as the 10th percentile of the estimated level
            # This captures the lower bound of normal operation
            level_percentile = np.percentile(smoothed_level, 10)
            
            # For a more adaptive baseline, we can use a rolling window
            rolling_window = min(30, len(smoothed_level) // 3)  # Use at most 30 days or 1/3 of data
            
            # Calculate a rolling 10th percentile
            rolling_baseline = pd.Series(smoothed_level).rolling(
                window=rolling_window, min_periods=3
            ).quantile(0.1).values
            
            # Fill NaN values at the beginning with the first valid value
            first_valid = next((i for i, x in enumerate(rolling_baseline) if not np.isnan(x)), None)
            if first_valid is not None:
                rolling_baseline[:first_valid] = rolling_baseline[first_valid]
            
            # Check proximity to baseline
            current_level = smoothed_level[-1]
            current_baseline = rolling_baseline[-1]
            
            proximity_threshold = 0.2  # 20% above baseline
            proximity_ratio = (current_level - current_baseline) / current_baseline
            
            # Calculate the trend in the baseline
            if len(rolling_baseline) >= 2:
                baseline_trend = (rolling_baseline[-1] - rolling_baseline[0]) / len(rolling_baseline)
                
                # Convert to daily trend if weekly
                if granularity == "Semanal":
                    baseline_trend = baseline_trend / 7
                
                baseline_trend_label = "crescimento" if baseline_trend > 0 else "queda"
            else:
                baseline_trend = 0
                baseline_trend_label = "estável"
            
            # Plot the results
            fig_ss = go.Figure()
            
            # Add original data
            fig_ss.add_trace(go.Scatter(
                x=ss_data.index, 
                y=ss_data.values,
                mode='lines',
                name='cov% Globo',
                line=dict(color='blue')
            ))
            
            # Add smoothed level
            fig_ss.add_trace(go.Scatter(
                x=ss_data.index, 
                y=smoothed_level,
                mode='lines',
                name='Nível Latente',
                line=dict(color='green')
            ))
            
            # Add rolling baseline
            fig_ss.add_trace(go.Scatter(
                x=ss_data.index, 
                y=rolling_baseline,
                mode='lines',
                name='Basal Adaptativo (10%)',
                line=dict(color='red', dash='dash')
            ))
            
            fig_ss.update_layout(
                title=f'Modelo de Espaço de Estados - Nível Basal Adaptativo',
                xaxis_title='Data',
                yaxis_title='cov%',
                legend=dict(orientation="h", y=1.1)
            )
            
            st.plotly_chart(fig_ss, use_container_width=True)
            
            # Display baseline metrics
            baseline_data = [{
                "Granularidade": granularity,
                "Basal Atual (cov%)": f"{current_baseline:.2f}%",
                "Tendência do Basal (p.p./dia)": f"{baseline_trend:.5f}",
                "Nível Atual/Basal": f"{proximity_ratio * 100:.1f}%"
            }]
            
            baseline_df = pd.DataFrame(baseline_data)
            st.dataframe(baseline_df, hide_index=True, use_container_width=True)
            
            # Generate alerts
            if proximity_ratio < proximity_threshold:
                st.warning(f"""
                ⚠️ **Alerta**: O cov% atual ({current_level:.2f}%) está muito próximo do basal estimado ({current_baseline:.2f}%).
                Isso indica que a audiência está operando próxima do seu nível mínimo histórico.
                """)
            
            if baseline_trend < 0:
                st.warning(f"""
                ⚠️ **Alerta**: A tendência do basal é de queda ({baseline_trend:.5f} p.p./dia),
                sugerindo uma erosão gradual da audiência mínima ao longo do tempo.
                """)
                
            # Calculate projected time to reach baseline if current trend continues
            if 'trend_daily_pp' in locals() and trend_daily_pp < 0:
                days_to_baseline = (current_level - current_baseline) / abs(trend_daily_pp)
                if days_to_baseline < 30:  # Only alert if within a month
                    st.warning(f"""
                    ⚠️ **Alerta de Projeção**: Se a tendência atual continuar, a audiência atingirá o nível basal em 
                    aproximadamente {int(days_to_baseline)} dias.
                    """)
            
        except Exception as e:
            st.error(f"Erro na análise de nível basal: {str(e)}")
            
            # If there's an error, fall back to the original quantile regression approach
            st.info("""
            **Análise simplificada:**
            
            Devido a limitações técnicas, estamos usando Regressão Quantílica para estimar o basal.
            """)
            
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
                
            except Exception as e_inner:
                st.error(f"Erro na análise simplificada de basal: {str(e_inner)}")
    else:
        st.warning("Dados insuficientes para análise de nível basal.")
    
    # 5. Final Conclusions
    st.subheader("Conclusões Finais - TV Globo")
    
    if "LINEAR_GLOBO_cov%" in selected_df.columns:
        # Calculate key metrics
        avg_cov = selected_df['LINEAR_GLOBO_cov%'].mean()
        
        # Get R² from the trend analysis if available
        r_squared_value = train_r2 if 'train_r2' in locals() else None
        if r_squared_value is None and 'r_squared' in locals():
            r_squared_value = r_squared
        
        # Get trend direction and value if available
        trend_daily_pp_value = trend_daily_pp if 'trend_daily_pp' in locals() else None
        
        # Get baseline value if available
        baseline_value = None
        if 'current_baseline' in locals():
            baseline_value = current_baseline
        elif 'current_baseline_value' in locals():
            baseline_value = current_baseline_value
            
        # Create metrics columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("cov% Médio Diário", f"{avg_cov:.2f}%")
            if r_squared_value is not None:
                st.metric("Qualidade do Modelo", f"R² = {r_squared_value:.2f}")
            
        with col2:
            if trend_daily_pp_value is not None:
                trend_label = "Crescimento" if trend_daily_pp_value > 0 else "Queda"
                st.metric("Tendência", f"{trend_label} ({trend_daily_pp_value:.4f} p.p./dia)")
            
            if baseline_value is not None:
                st.metric("Basal Diário", f"{baseline_value:.2f}%")
    
    st.markdown("""
    ## Principais Insights
    
    - **Poder preditivo aprimorado:** A combinação de Transformada de Fourier com ARIMA permite identificar
    os padrões sazonais mais relevantes na audiência da Globo, melhorando a precisão das previsões.
    
    - **Análise de sazonalidade avançada:** Os componentes de Fourier revelam padrões cíclicos específicos
    na audiência, permitindo diferenciar entre efeitos diários, semanais e mensais.
    
    - **Nível basal adaptativo:** O modelo de espaço de estados com filtro de Kalman detecta o piso dinâmico
    da audiência, adaptando-se às mudanças estruturais do mercado ao longo do tempo.
    
    - **Alertas preditivos:** O sistema identifica proativamente quando a audiência está operando próxima
    ao nível basal ou quando há tendências consistentes de queda, possibilitando intervenções antecipadas.
    
    Estes insights fornecem uma visão abrangente e tecnicamente robusta da performance histórica da Globo, 
    permitindo decisões estratégicas melhor fundamentadas para manter e expandir a audiência.
    """)
    
    # 6. Notes and Documentation
    st.subheader("Notas Metodológicas")
    
    with st.expander("Informações sobre metodologia"):
        st.markdown("""
        ### Fonte dos Dados
        
        Os dados de audiência de TV Linear são coletados pela Kantar IBOPE Media.
        
        ### Análise de Tendência e Sazonalidade (Fourier + ARIMA)
        
        Esta abordagem combina duas técnicas poderosas:
        
        1. **Transformada de Fourier** - Decompõe a série temporal em componentes de frequência, identificando 
        os ciclos mais importantes na audiência (diários, semanais, mensais).
        
        2. **Modelo ARIMA com Features Exógenas** - Utiliza os componentes de Fourier como variáveis explicativas 
        em um modelo que captura tendência e correlações temporais.
        
        Esta combinação é mais flexível que o modelo Holt-Winters tradicional, permitindo capturar múltiplos 
        padrões sazonais simultaneamente e se adaptando melhor às características específicas dos dados.
        
        ### Cálculo do Basal de Audiência (Modelo de Espaço de Estados)
        
        O nível basal é estimado utilizando um **Modelo de Espaço de Estados com Filtro de Kalman**, que:
        
        1. Representa a audiência como um processo dinâmico com componentes latentes (não observáveis).
        2. Estima o nível subjacente da audiência, filtrando flutuações de curto prazo.
        3. Calcula o basal como o percentil 10% do nível estimado, usando uma janela móvel para adaptação.
        4. Atualiza continuamente as estimativas conforme novos dados são recebidos.
        
        Esta abordagem é superior à regressão quantílica por considerar a natureza dinâmica do "piso" da audiência, 
        que pode mudar gradualmente conforme o mercado e os hábitos de consumo evoluem.
        """)