# utils/analise_redes_sociais.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

def analise_redes_sociais(df):
    """
    Performs analysis of social media data according to hierarchical structure:
    Rede Social > Conta Específica > Tipo de Métrica
    
    Args:
        df (DataFrame): Processed dataframe with RS_ and LINEAR_ prefixed columns
    """
    
    st.header("🔍 Redes Sociais - Desempenho e Impacto")
    
    # 1. Header section with last update date
    if 'data_hora' in df.columns:
        last_date = df['data_hora'].max()
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
        st.caption(f"Última atualização: {last_date.strftime('%d/%m/%Y')}")
    
    # Ensure data_hora is datetime type
    if 'data_hora' in df.columns and not pd.api.types.is_datetime64_dtype(df['data_hora']):
        df['data_hora'] = pd.to_datetime(df['data_hora'])
    
    # Introduction text
    st.markdown("""
    Esta análise examina o desempenho das diferentes plataformas sociais da Globo, identificando
    quais geram maior impacto na audiência da TV Linear. Navegue pelos diferentes níveis de análise
    para descobrir insights específicos.
    
    Utilize os filtros abaixo para ajustar a análise de acordo com suas necessidades:
    """)
    
    # 2. Global filters section
    col1, col2 = st.columns(2)
    
    with col1:
        # Granularity selection
        granularity = st.selectbox(
            "Granularidade:",
            options=["Horário", "Diário", "Semanal"],
            index=1  # Default to "Diário"
        )
    
    with col2:
        # Metric type selection
        metric_type = st.selectbox(
            "Métricas para Análise:",
            options=["Ativação", "Alcance", "Engajamento"],
            index=2  # Default to "Engajamento"
        )
    
    # 3. Create dataframe for analysis based on granularity
    # Process data based on selected granularity
    df_analysis = df.copy()
    
    if 'data_hora' in df_analysis.columns:
        if granularity == "Diário":
            df_analysis['period'] = df_analysis['data_hora'].dt.date
            period_name = "Diário"
        elif granularity == "Semanal":
            df_analysis['period'] = df_analysis['data_hora'].dt.to_period('W').astype(str)
            period_name = "Semanal"
        else:  # "Horário"
            df_analysis['period'] = df_analysis['data_hora']
            period_name = "Horário"
        
        # Group by period
        if granularity != "Horário":
            # Get numeric columns for aggregation
            numeric_cols = df_analysis.select_dtypes(include=['number']).columns.tolist()
            
            # Group by period
            df_analysis = df_analysis.groupby('period')[numeric_cols].mean().reset_index()
            
            # Convert period to datetime for plotting
            if granularity == "Diário":
                df_analysis['data_hora'] = pd.to_datetime(df_analysis['period'])
            else:  # "Semanal"
                df_analysis['data_hora'] = pd.to_datetime(df_analysis['period'].str.split('/').str[0])
    
    # 4. Identify available social media platforms and accounts
    # Get all RS_ columns
    rs_cols = [col for col in df_analysis.columns if col.startswith('RS_')]
    
    # Extract platforms
    platforms = []
    for col in rs_cols:
        parts = col.split('_')
        if len(parts) > 2:
            platform = parts[2]  # RS_GLOBO_PLATFORM_metric
            if platform not in platforms and platform in ["FACEBOOK", "INSTAGRAM", "TIKTOK", "YOUTUBE"]:
                platforms.append(platform)
    
    # Extract accounts
    accounts = []
    for col in rs_cols:
        parts = col.split('_')
        if len(parts) > 2:
            if "CANAIS" in col:  # RS_CANAIS_PLATFORM_ACCOUNT_metric
                if len(parts) > 3:
                    account = parts[3]
                    if account not in accounts:
                        accounts.append(account)
            else:  # RS_ACCOUNT_PLATFORM_metric
                account = parts[1]
                if account not in accounts:
                    accounts.append(account)
    
    # 5. Create tabs for the hierarchical analysis
    tabs = st.tabs(["Rede Social", "Conta Específica", "Tipo de Métrica"])
    
    # 6. Tab 1: Rede Social Analysis
    with tabs[0]:
        st.subheader("Análise por Rede Social")
        
        st.markdown("""
        Esta seção mostra o desempenho agregado de cada plataforma social, comparando esforço (quantidade de posts)
        versus retorno (métricas de performance) e a correlação com a audiência da TV Linear.
        """)
        
        # Platform selection with "Select All" checkbox
        select_all_platforms = st.checkbox("Selecionar todas as plataformas", value=True)
        
        if select_all_platforms:
            selected_platforms = platforms
        else:
            selected_platforms_tab1 = st.multiselect(
                "Selecione as plataformas para análise:",
                options=platforms,
                default=platforms
            )
            
            if not selected_platforms_tab1:
                st.warning("Por favor, selecione pelo menos uma plataforma para análise.")
                return
            else:
                selected_platforms = selected_platforms_tab1
        
        # For each platform, calculate key metrics based on selected metric type
        platform_metrics = []
        
        for platform in selected_platforms:
            # Get relevant columns for this platform
            if metric_type == "Ativação":
                metric_cols = [col for col in rs_cols if f"_{platform}_posts_quantity" in col]
            elif metric_type == "Alcance":
                metric_cols = [col for col in rs_cols if f"_{platform}_" in col and any(x in col.lower() for x in ["reach", "alcance", "impressions", "impressoes", "views", "videoviews"])]
            else:  # "Engajamento"
                metric_cols = [col for col in rs_cols if f"_{platform}_" in col and any(x in col.lower() for x in ["comments", "comentarios", "reactions", "reacoes", "shares", "saves", "total_interactions", "interacoes"])]
            
            if not metric_cols:
                continue
            
            # Calculate total posts for this platform
            posts_cols = [col for col in rs_cols if f"_{platform}_posts_quantity" in col]
            total_posts = df_analysis[posts_cols].sum().sum() if posts_cols else 0
            
            # Calculate total for selected metrics
            total_metric = df_analysis[metric_cols].sum().sum() if metric_cols else 0
            
            # Calculate correlation with TV Linear if available
            correlation = None
            if 'LINEAR_GLOBO_cov%' in df_analysis.columns and metric_cols:
                # Create aggregated metric for correlation
                df_analysis[f'{platform}_combined'] = df_analysis[metric_cols].sum(axis=1)
                correlation = df_analysis[f'{platform}_combined'].corr(df_analysis['LINEAR_GLOBO_cov%'])
            
            # Add to metrics list
            platform_metrics.append({
                "Plataforma": platform,
                "Posts": total_posts,
                "Valor Métrica": total_metric,
                "Correlação TV Linear": correlation
            })
        
        # Create DataFrame from metrics
        if platform_metrics:
            df_platforms = pd.DataFrame(platform_metrics)
            
            # Calculate percentages
            total_posts = df_platforms["Posts"].sum()
            total_metric = df_platforms["Valor Métrica"].sum()
            
            df_platforms["% Posts"] = df_platforms["Posts"] / total_posts * 100 if total_posts > 0 else 0
            df_platforms["% Métrica"] = df_platforms["Valor Métrica"] / total_metric * 100 if total_metric > 0 else 0
            
            # Add status indicators based on correlation
            def get_status(corr):
                if pd.isna(corr):
                    return "❓"
                elif corr > 0.5:
                    return "✅"
                elif corr > 0.3:
                    return "⚠️"
                else:
                    return "❌"
            
            df_platforms["Status"] = df_platforms["Correlação TV Linear"].apply(get_status)
            
            # Sort by correlation (highest first)
            df_platforms = df_platforms.sort_values("Correlação TV Linear", ascending=False)
            
            # Display table
            display_df = df_platforms.copy()
            
            # Format for display
            display_df["% Posts"] = display_df["% Posts"].apply(lambda x: f"{x:.1f}%")
            display_df["% Métrica"] = display_df["% Métrica"].apply(lambda x: f"{x:.1f}%")
            display_df["Correlação TV Linear"] = display_df["Correlação TV Linear"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            
            # Create a formatted display string
            platform_results = []
            for _, row in display_df.iterrows():
                platform_results.append(
                    f"{row['Plataforma']}: {row['% Posts']} dos Posts | {row['% Métrica']} da Métrica | " +
                    f"Correlação com TV Linear: {row['Correlação TV Linear']} {row['Status']}"
                )
            
            # Display formatted results
            st.subheader("Comparação de Esforço vs. Retorno:")
            for result in platform_results:
                st.markdown(f"- {result}")
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Effort vs. Return Chart
                fig_effort = px.bar(
                    df_platforms,
                    x="Plataforma",
                    y=["% Posts", "% Métrica"],
                    barmode="group",
                    title=f"Esforço vs. Retorno por Plataforma ({metric_type})",
                    labels={"value": "Porcentagem (%)", "variable": "Métrica"}
                )
                
                st.plotly_chart(fig_effort, use_container_width=True)
            
            with col2:
                # Correlation Chart
                fig_corr = px.bar(
                    df_platforms,
                    x="Plataforma",
                    y="Correlação TV Linear",
                    title="Correlação com TV Linear por Plataforma",
                    color="Correlação TV Linear",
                    color_continuous_scale=["red", "yellow", "green"]
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Time series analysis for platforms
            st.subheader("Evolução Temporal por Plataforma")
            
            # Prepare data for time series
            time_data = []
            
            for platform in selected_platforms:
                # Select relevant columns based on metric type
                if metric_type == "Ativação":
                    metric_cols = [col for col in rs_cols if f"_{platform}_posts_quantity" in col]
                elif metric_type == "Alcance":
                    metric_cols = [col for col in rs_cols if f"_{platform}_" in col and any(x in col.lower() for x in ["reach", "alcance", "impressions", "impressoes", "views", "videoviews"])]
                else:  # "Engajamento"
                    metric_cols = [col for col in rs_cols if f"_{platform}_" in col and any(x in col.lower() for x in ["comments", "comentarios", "reactions", "reacoes", "shares", "saves", "total_interactions", "interacoes"])]
                
                if metric_cols:
                    df_analysis[f'{platform}_combined'] = df_analysis[metric_cols].sum(axis=1)
                    
                    for _, row in df_analysis.iterrows():
                        time_data.append({
                            "Data": row['data_hora'] if 'data_hora' in df_analysis.columns else row['period'],
                            "Plataforma": platform,
                            "Valor": row[f'{platform}_combined']
                        })
            
            # Create time series chart with TV Linear on secondary y-axis
            if time_data:
                df_time = pd.DataFrame(time_data)
                
                # Create figure with secondary y-axis
                fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add traces for each platform on primary y-axis
                for platform in selected_platforms:
                    platform_data = df_time[df_time['Plataforma'] == platform]
                    if not platform_data.empty:
                        fig_time.add_trace(
                            go.Scatter(
                                x=platform_data['Data'],
                                y=platform_data['Valor'],
                                name=platform,
                                mode='lines'
                            ),
                            secondary_y=False
                        )
                
                # Add TV Linear data on secondary y-axis if available
                if 'LINEAR_GLOBO_cov%' in df_analysis.columns:
                    fig_time.add_trace(
                        go.Scatter(
                            x=df_analysis['data_hora'] if 'data_hora' in df_analysis.columns else df_analysis['period'],
                            y=df_analysis['LINEAR_GLOBO_cov%'],
                            name='TV Linear (cov%)',
                            mode='lines',
                            line=dict(color='red', width=2, dash='dash')
                        ),
                        secondary_y=True
                    )
                
                # Update axes titles
                fig_time.update_xaxes(title_text="Data")
                fig_time.update_yaxes(title_text=f"{metric_type}", secondary_y=False)
                fig_time.update_yaxes(title_text="TV Linear (cov%)", secondary_y=True)
                
                # Update layout
                fig_time.update_layout(
                    title=f"Evolução Temporal de {metric_type} por Plataforma vs. TV Linear",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Generate insights based on analysis
            st.subheader("Insights Automáticos:")
            
            # Check if one platform dominates
            dominant_platform = df_platforms.iloc[0]
            if dominant_platform["Correlação TV Linear"] > 0.5:
                st.success(f"**{dominant_platform['Plataforma']}** mostra a correlação mais forte com TV Linear ({dominant_platform['Correlação TV Linear']:.2f}), sugerindo que é a plataforma mais efetiva para impactar a audiência de TV.")
            
            # Check for misalignment between effort and return
            for _, row in df_platforms.iterrows():
                if row["% Métrica"] > row["% Posts"] * 1.5:
                    st.info(f"**{row['Plataforma']}** está gerando retorno desproporcional ao esforço (Recebe {row['% Posts']} dos posts mas gera {row['% Métrica']} da métrica). Considere aumentar a presença nesta plataforma.")
                elif row["% Posts"] > row["% Métrica"] * 1.5:
                    st.warning(f"**{row['Plataforma']}** está recebendo mais esforço do que o retorno justifica (Recebe {row['% Posts']} dos posts mas gera apenas {row['% Métrica']} da métrica). Considere otimizar ou reduzir a presença.")
            
            # Check for low correlation platforms
            low_corr_platforms = df_platforms[df_platforms["Correlação TV Linear"] < 0.2]
            if not low_corr_platforms.empty:
                low_platforms = ", ".join(low_corr_platforms["Plataforma"].tolist())
                st.warning(f"As plataformas **{low_platforms}** mostram baixa correlação com TV Linear (<0.2). Podem ser úteis para awareness, mas não para impactar diretamente a audiência de TV.")
            
            # Calculate total posts per day on average
            if 'period' in df_analysis.columns and posts_cols:
                avg_daily_posts = total_posts / df_analysis['period'].nunique()
                st.metric("Média de Posts Diários", f"{avg_daily_posts:.1f}")
                
                # Check for low posting days
                if 'period' in df_analysis.columns and posts_cols:
                    daily_posts = df_analysis.groupby('period')[posts_cols].sum().sum(axis=1)
                    low_posting_days = (daily_posts < avg_daily_posts * 0.8).sum()
                    
                    if low_posting_days > daily_posts.shape[0] * 0.2:
                        st.warning(f"⚠️ Em {low_posting_days} dias ({(low_posting_days/daily_posts.shape[0]*100):.1f}% do período), o número de posts ficou 20% abaixo da média. Isso pode levar à perda de presença orgânica e engajamento.")
        else:
            st.warning("Dados insuficientes para análise por plataforma.")
    
    # 7. Tab 2: Conta Específica Analysis
    with tabs[1]:
        st.subheader("Análise por Conta Específica")
        
        st.markdown("""
        Esta seção examina o desempenho de contas específicas dentro das redes sociais,
        identificando quais geram maior impacto na audiência da TV Linear.
        """)
        
        # Platform selection with "Select All" checkbox
        select_all_platforms_tab2 = st.checkbox("Selecionar todas as plataformas", value=True, key="select_all_platforms_tab2")
        
        if select_all_platforms_tab2:
            selected_platforms_tab2_filtered = platforms
        else:
            selected_platforms_tab2 = st.multiselect(
                "Selecione as plataformas para análise:",
                options=platforms,
                default=platforms,
                key="platforms_tab2"
            )
            
            if not selected_platforms_tab2:
                st.warning("Por favor, selecione pelo menos uma plataforma para análise.")
                return
            else:
                selected_platforms_tab2_filtered = selected_platforms_tab2
        
        # Account selection with "Select All" checkbox
        select_all_accounts = st.checkbox("Selecionar todas as contas", value=True)
        
        if select_all_accounts:
            selected_accounts = accounts
        else:
            selected_accounts_list = st.multiselect(
                "Selecione as contas para análise:",
                options=accounts,
                default=accounts[0:5] if len(accounts) > 5 else accounts
            )
            
            if not selected_accounts_list:
                st.warning("Por favor, selecione pelo menos uma conta para análise.")
                return
            else:
                selected_accounts = selected_accounts_list
        
        # For each account, calculate key metrics based on selected metric type
        account_metrics = []
        
        for account in selected_accounts:
            # Get relevant columns for this account across selected platforms
            account_metric_cols = []
            
            for platform in selected_platforms_tab2_filtered:
                if metric_type == "Ativação":
                    metric_cols = [col for col in rs_cols if f"_{platform}_{account}_posts_quantity" in col or (f"_{account}_" in col and f"_{platform}_posts_quantity" in col)]
                elif metric_type == "Alcance":
                    metric_cols = [col for col in rs_cols if (f"_{platform}_{account}_" in col or (f"_{account}_" in col and f"_{platform}_" in col)) and any(x in col.lower() for x in ["reach", "alcance", "impressions", "impressoes", "views", "videoviews"])]
                else:  # "Engajamento"
                    metric_cols = [col for col in rs_cols if (f"_{platform}_{account}_" in col or (f"_{account}_" in col and f"_{platform}_" in col)) and any(x in col.lower() for x in ["comments", "comentarios", "reactions", "reacoes", "shares", "saves", "total_interactions", "interacoes"])]
                
                account_metric_cols.extend(metric_cols)
            
            if not account_metric_cols:
                continue
            
            # Calculate total posts for this account
            posts_cols = [col for col in rs_cols if f"_{account}_posts_quantity" in col or "posts_quantity" in col and f"_{account}_" in col]
            total_posts = df_analysis[posts_cols].sum().sum() if posts_cols else 0
            
            # Calculate total for selected metrics
            total_metric = df_analysis[account_metric_cols].sum().sum() if account_metric_cols else 0
            
            # Calculate correlation with TV Linear if available
            correlation = None
            if 'LINEAR_GLOBO_cov%' in df_analysis.columns and account_metric_cols:
                # Create aggregated metric for correlation
                df_analysis[f'{account}_combined'] = df_analysis[account_metric_cols].sum(axis=1)
                correlation = df_analysis[f'{account}_combined'].corr(df_analysis['LINEAR_GLOBO_cov%'])
            
            # Determine main platform for this account
            platform_cols = {platform: [col for col in account_metric_cols if f"_{platform}_" in col] for platform in selected_platforms_tab2_filtered}
            platform_totals = {platform: df_analysis[cols].sum().sum() if cols else 0 for platform, cols in platform_cols.items()}
            main_platform = max(platform_totals.items(), key=lambda x: x[1])[0] if platform_totals else "N/A"
            
            # Add to metrics list
            account_metrics.append({
                "Conta": account,
                "Plataforma Principal": main_platform,
                "Posts": total_posts,
                "Valor Métrica": total_metric,
                "Correlação TV Linear": correlation
            })
        
        # Create DataFrame from metrics
        if account_metrics:
            df_accounts = pd.DataFrame(account_metrics)
            
            # Calculate percentages
            total_posts = df_accounts["Posts"].sum()
            total_metric = df_accounts["Valor Métrica"].sum()
            
            df_accounts["% Posts"] = df_accounts["Posts"] / total_posts * 100 if total_posts > 0 else 0
            df_accounts["% Métrica"] = df_accounts["Valor Métrica"] / total_metric * 100 if total_metric > 0 else 0
            
            # Add status indicators based on correlation
            def get_status(corr):
                if pd.isna(corr):
                    return "❓"
                elif corr > 0.5:
                    return "✅"
                elif corr > 0.3:
                    return "⚠️"
                else:
                    return "❌"
            
            df_accounts["Status"] = df_accounts["Correlação TV Linear"].apply(get_status)
            
            # Add metric status
            def get_metric_status(row):
                if row["% Métrica"] > df_accounts["% Métrica"].median() * 1.2:
                    return "✅"
                else:
                    return ""
            
            df_accounts["Métrica Status"] = df_accounts.apply(get_metric_status, axis=1)
            
            # Sort by correlation (highest first)
            df_accounts = df_accounts.sort_values("Correlação TV Linear", ascending=False)
            
            # Display table
            display_df = df_accounts.copy()
            
            # Format for display
            display_df["% Posts"] = display_df["% Posts"].apply(lambda x: f"{x:.1f}%")
            display_df["% Métrica"] = display_df["% Métrica"].apply(lambda x: f"{x:.1f}%")
            display_df["Correlação TV Linear"] = display_df["Correlação TV Linear"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            
            # Create a formatted display string
            account_results = []
            for _, row in display_df.iterrows():
                account_results.append(
                    f"{row['Plataforma Principal']} @{row['Conta']}: {row['% Posts']} dos Posts | " +
                    f"{row['% Métrica']} da Métrica {row['Métrica Status']} | " +
                    f"Correlação com TV Linear: {row['Correlação TV Linear']} {row['Status']}"
                )
            
            # Display formatted results
            st.subheader("Performance por Conta:")
            for result in account_results:
                st.markdown(f"- {result}")
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Effort vs. Return Chart for top 10 accounts
                top_accounts = df_accounts.head(10) if len(df_accounts) > 10 else df_accounts
                
                fig_effort = px.bar(
                    top_accounts,
                    x="Conta",
                    y=["% Posts", "% Métrica"],
                    barmode="group",
                    title=f"Esforço vs. Retorno por Conta ({metric_type})",
                    labels={"value": "Porcentagem (%)", "variable": "Métrica"}
                )
                
                st.plotly_chart(fig_effort, use_container_width=True)
            
            with col2:
                # Correlation Chart for top 10 accounts
                fig_corr = px.bar(
                    top_accounts,
                    x="Conta",
                    y="Correlação TV Linear",
                    title="Correlação com TV Linear por Conta",
                    color="Correlação TV Linear",
                    color_continuous_scale=["red", "yellow", "green"]
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            # Time series analysis for accounts
            st.subheader("Evolução Temporal por Conta")
            
            # Prepare data for time series
            time_data = []
            
            for account in selected_accounts[:5]:  # Limit to top 5 for readability
                # Get combined metric for this account
                if f'{account}_combined' in df_analysis.columns:
                    for _, row in df_analysis.iterrows():
                        time_data.append({
                            "Data": row['data_hora'] if 'data_hora' in df_analysis.columns else row['period'],
                            "Conta": account,
                            "Valor": row[f'{account}_combined']
                        })
            
            # Create time series chart with TV Linear on secondary y-axis
            if time_data:
                df_time = pd.DataFrame(time_data)
                
                # Create figure with secondary y-axis
                fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add traces for each account on primary y-axis
                for account in df_time['Conta'].unique():
                    account_data = df_time[df_time['Conta'] == account]
                    if not account_data.empty:
                        fig_time.add_trace(
                            go.Scatter(
                                x=account_data['Data'],
                                y=account_data['Valor'],
                                name=account,
                                mode='lines'
                            ),
                            secondary_y=False
                        )
                
                # Add TV Linear data on secondary y-axis if available
                if 'LINEAR_GLOBO_cov%' in df_analysis.columns:
                    fig_time.add_trace(
                        go.Scatter(
                            x=df_analysis['data_hora'] if 'data_hora' in df_analysis.columns else df_analysis['period'],
                            y=df_analysis['LINEAR_GLOBO_cov%'],
                            name='TV Linear (cov%)',
                            mode='lines',
                            line=dict(color='red', width=2, dash='dash')
                        ),
                        secondary_y=True
                    )
                
                # Update axes titles
                fig_time.update_xaxes(title_text="Data")
                fig_time.update_yaxes(title_text=f"{metric_type}", secondary_y=False)
                fig_time.update_yaxes(title_text="TV Linear (cov%)", secondary_y=True)
                
                # Update layout
                fig_time.update_layout(
                    title=f"Evolução Temporal de {metric_type} por Conta vs. TV Linear",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
            
            # Grouping accounts by vertical
            # Define mapping of accounts to verticals
            verticals = {
                'Entretenimento': ['BBB', 'Estrela da Casa', 'Multishow', 'GNT', 'VIVA', 'Gshow'],
                'Esportes': ['Sportv', 'ge', 'Cartola'],
                'Notícias': ['g1', 'GloboNews'],
                'Variedades': ['Receitas', 'Mundo Gloob', 'Globoplay']
            }
            
            # Create a lookup dictionary
            account_to_vertical = {}
            for vertical, accts in verticals.items():
                for acct in accts:
                    account_to_vertical[acct] = vertical
            
            # Add vertical to df_accounts
            df_accounts['Vertical'] = df_accounts['Conta'].apply(
                lambda x: account_to_vertical.get(x, 'Outros')
            )
            
            # Aggregate by vertical
            vertical_metrics = df_accounts.groupby('Vertical').agg({
                'Posts': 'sum',
                'Valor Métrica': 'sum',
                'Correlação TV Linear': 'mean'
            }).reset_index()
            
            # Calculate percentages
            total_posts = vertical_metrics["Posts"].sum()
            total_metric = vertical_metrics["Valor Métrica"].sum()
            
            vertical_metrics["% Posts"] = vertical_metrics["Posts"] / total_posts * 100 if total_posts > 0 else 0
            vertical_metrics["% Métrica"] = vertical_metrics["Valor Métrica"] / total_metric * 100 if total_metric > 0 else 0
            
            # Add status indicators
            vertical_metrics["Status"] = vertical_metrics["Correlação TV Linear"].apply(get_status)
            
            # Sort by correlation
            vertical_metrics = vertical_metrics.sort_values("Correlação TV Linear", ascending=False)
            
            # Display vertical analysis
            st.subheader("Análise por Vertical de Conteúdo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Effort vs. Return Chart for verticals
                fig_vertical = px.bar(
                    vertical_metrics,
                    x="Vertical",
                    y=["% Posts", "% Métrica"],
                    barmode="group",
                    title=f"Esforço vs. Retorno por Vertical ({metric_type})",
                    labels={"value": "Porcentagem (%)", "variable": "Métrica"}
                )
                
                st.plotly_chart(fig_vertical, use_container_width=True)
            
            with col2:
                # Correlation Chart for verticals
                fig_corr_vertical = px.bar(
                    vertical_metrics,
                    x="Vertical",
                    y="Correlação TV Linear",
                    title="Correlação com TV Linear por Vertical",
                    color="Correlação TV Linear",
                    color_continuous_scale=["red", "yellow", "green"]
                )
                
                st.plotly_chart(fig_corr_vertical, use_container_width=True)
            
            # Generate insights based on analysis
            st.subheader("Insights Automáticos:")
            
            # Find best performing accounts
            best_account = df_accounts.iloc[0]
            if best_account["Correlação TV Linear"] > 0.5:
                st.success(f"**{best_account['Plataforma Principal']} @{best_account['Conta']}** mostra a correlação mais forte com TV Linear ({best_account['Correlação TV Linear']:.2f}), sugerindo maior impacto na audiência de TV.")
            
            # Find best performing vertical
            best_vertical = vertical_metrics.iloc[0]
            if best_vertical["Correlação TV Linear"] > 0.3:
                st.success(f"A vertical **{best_vertical['Vertical']}** tem a maior correlação média com TV Linear ({best_vertical['Correlação TV Linear']:.2f}), indicando que este tipo de conteúdo tem maior potencial de impactar a audiência.")
            
            # Check for accounts with high return on effort
            high_roi = df_accounts[df_accounts["% Métrica"] > df_accounts["% Posts"] * 1.5]
            if not high_roi.empty:
                top_roi = high_roi.iloc[0]
                st.info(f"**{top_roi['Plataforma Principal']} @{top_roi['Conta']}** apresenta o melhor retorno sobre esforço, gerando {top_roi['% Métrica']:.1f}% da métrica com apenas {top_roi['% Posts']:.1f}% dos posts.")
        else:
            st.warning("Dados insuficientes para análise por conta específica.")
    
    # 8. Tab 3: Tipo de Métrica Analysis
    with tabs[2]:
        st.subheader("Análise por Tipo de Métrica")
        
        st.markdown("""
        Esta seção analisa como métricas específicas (comentários, compartilhamentos, etc.) 
        se correlacionam com a audiência da TV Linear, identificando quais indicadores
        são mais relevantes para prever o desempenho da TV.
        """)
        
        # Reuse platform filter from Tab 1
        all_platforms_option_tab3 = ["Todas as redes sociais"]
        platform_options_tab3 = all_platforms_option_tab3 + platforms
        
        selected_platforms_tab3 = st.multiselect(
            "Selecione as plataformas para análise:",
            options=platform_options_tab3,
            default=all_platforms_option_tab3,
            key="platforms_tab3"
        )
        
        # Handle "Todas as redes sociais" option
        if "Todas as redes sociais" in selected_platforms_tab3:
            selected_platforms_tab3_filtered = platforms
        elif not selected_platforms_tab3:
            st.warning("Por favor, selecione pelo menos uma plataforma para análise.")
            return
        else:
            selected_platforms_tab3_filtered = selected_platforms_tab3
        
        # Account selection with "All accounts" option
        all_accounts_option_tab3 = ["Todas as contas"]
        account_options_tab3 = all_accounts_option_tab3 + accounts
        
        selected_accounts_list_tab3 = st.multiselect(
            "Selecione as contas para análise:",
            options=account_options_tab3,
            default=all_accounts_option_tab3,
            key="accounts_tab3"
        )
        
        # Handle "Todas as contas" option
        if "Todas as contas" in selected_accounts_list_tab3:
            selected_accounts_tab3 = accounts
        elif not selected_accounts_list_tab3:
            st.warning("Por favor, selecione pelo menos uma conta para análise.")
            return
        else:
            selected_accounts_tab3 = selected_accounts_list_tab3
        
        # Extract generic metrics (like "comments", "impressions", etc.)
        metrics = set()
        for col in rs_cols:
            parts = col.split('_')
            if len(parts) > 3:
                # Get just the generic metric name
                metric_type = parts[-1]  # Get the last part which should be the generic metric
                
                # Handle special cases
                if metric_type == "quantity" and parts[-2] == "posts":
                    continue  # Skip posts_quantity
                
                # Common generic metrics to look for
                generic_metrics = {
                    "comments": "comentarios",
                    "reactions": "reacoes", 
                    "shares": "compartilhamentos",
                    "saves": "salvamentos",
                    "reach": "alcance", 
                    "impressions": "impressoes",
                    "views": "visualizacoes",
                    "videoviews": "visualizacoes_video",
                    "interacoes": "interacoes",
                    "interactions": "interacoes"
                }
                
                # Standardize some common metrics
                if metric_type in generic_metrics:
                    metric_type = generic_metrics[metric_type]
                
                if metric_type and len(metric_type) > 0:
                    metrics.add(metric_type)
        
        # Convert to list and sort
        metrics_list = sorted(list(metrics))
        
        # Add "Todas as métricas" option
        all_metrics_option = ["Todas as métricas"]
        metrics_options = all_metrics_option + metrics_list
        
        # Metric selection
        selected_metrics_list = st.multiselect(
            "Selecione as métricas para análise:",
            options=metrics_options,
            default=all_metrics_option
        )
        
        # Handle "Todas as métricas" option
        if "Todas as métricas" in selected_metrics_list:
            selected_metrics = metrics_list
        elif not selected_metrics_list:
            st.warning("Por favor, selecione pelo menos uma métrica para análise.")
            return
        else:
            selected_metrics = selected_metrics_list
        
        # For each metric, calculate correlation with TV Linear
        metric_correlations = []
        
        for metric in selected_metrics:
            for platform in selected_platforms_tab3_filtered:
                # Get columns for this metric and platform for selected accounts
                metric_cols = []
                
                for account in selected_accounts_tab3:
                    # Get columns specifically for this account, platform, and metric
                    account_metric_cols = [
                        col for col in rs_cols 
                        if (f"_{platform}_{account}_" in col or (f"_{account}_" in col and f"_{platform}_" in col)) and (
                            col.endswith(f"_{metric}") or 
                            f"_{metric}" in col or 
                            metric in col.lower()
                        )
                    ]
                    
                    metric_cols.extend(account_metric_cols)
                
                # If no account-specific columns were found, try to get platform-wide columns
                if not metric_cols:
                    metric_cols = [
                        col for col in rs_cols 
                        if f"_{platform}_" in col and (
                            col.endswith(f"_{metric}") or 
                            f"_{metric}" in col or 
                            metric in col.lower()
                        )
                    ]
                
                if not metric_cols:
                    continue
                
                # Calculate total for this metric
                metric_total = df_analysis[metric_cols].sum().sum() if metric_cols else 0
                
                # Calculate correlation with TV Linear if available
                correlation = None
                if 'LINEAR_GLOBO_cov%' in df_analysis.columns and metric_cols:
                    # Create aggregated metric for correlation
                    df_analysis[f'{platform}_{metric}_combined'] = df_analysis[metric_cols].sum(axis=1)
                    correlation = df_analysis[f'{platform}_{metric}_combined'].corr(df_analysis['LINEAR_GLOBO_cov%'])
                
                # Add to correlations list
                metric_correlations.append({
                    "Plataforma": platform,
                    "Métrica": metric,
                    "Total": metric_total,
                    "Correlação TV Linear": correlation
                })
        
        # Create DataFrame from correlations
        if metric_correlations:
            df_correlations = pd.DataFrame(metric_correlations)
            
            # Add status indicators based on correlation
            def get_status(corr):
                if pd.isna(corr):
                    return "❓"
                elif corr > 0.5:
                    return "✅"
                elif corr > 0.3:
                    return "⚠️"
                else:
                    return "❌"
            
            df_correlations["Status"] = df_correlations["Correlação TV Linear"].apply(get_status)
            
            # Sort by correlation (highest first)
            df_correlations = df_correlations.sort_values("Correlação TV Linear", ascending=False)
            
            # Also create a dataframe that aggregates by metric (across all platforms)
            metric_agg = df_correlations.groupby('Métrica').agg({
                'Total': 'sum',
                'Correlação TV Linear': 'mean'
            }).reset_index()
            
            # Add status indicators based on correlation
            metric_agg["Status"] = metric_agg["Correlação TV Linear"].apply(get_status)
            
            # Sort by correlation
            metric_agg = metric_agg.sort_values("Correlação TV Linear", ascending=False)
            
            # Display table
            display_df = df_correlations.copy()
            
            # Format for display
            display_df["Correlação TV Linear"] = display_df["Correlação TV Linear"].apply(lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A")
            
            # Create a formatted display string
            correlation_results = []
            for _, row in display_df.iterrows():
                correlation_results.append(
                    f"{row['Plataforma']} {row['Métrica'].capitalize()}: Correlação com TV Linear: {row['Correlação TV Linear']} {row['Status']}"
                )
            
            # Display formatted results
            st.subheader("Correlação por Métrica:")
            for i, result in enumerate(correlation_results):
                if i < 10:  # Limit to top 10 for readability
                    st.markdown(f"- {result}")
            
            if len(correlation_results) > 10:
                with st.expander("Ver todas as métricas"):
                    for i, result in enumerate(correlation_results):
                        if i >= 10:
                            st.markdown(f"- {result}")
            
            # Create visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Top metrics by platform (top 10)
                top_correlations = df_correlations.head(10)
                
                fig_corr = px.bar(
                    top_correlations,
                    x="Correlação TV Linear",
                    y=top_correlations.apply(lambda x: f"{x['Plataforma']} {x['Métrica'].capitalize()}", axis=1),
                    orientation='h',
                    title="Top 10 Combinações Plataforma-Métrica",
                    labels={"y": "Plataforma e Métrica"},
                    color="Correlação TV Linear",
                    color_continuous_scale=["red", "yellow", "green"]
                )
                
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with col2:
                # Aggregated metrics (across all platforms)
                fig_metric_agg = px.bar(
                    metric_agg,
                    x="Métrica",
                    y="Correlação TV Linear",
                    title="Correlação Média por Tipo de Métrica",
                    color="Correlação TV Linear",
                    color_continuous_scale=["red", "yellow", "green"]
                )
                
                st.plotly_chart(fig_metric_agg, use_container_width=True)
                
                # Display aggregated metrics table
                st.subheader("Correlação Média por Tipo de Métrica:")
                for _, row in metric_agg.iterrows():
                    st.markdown(f"- **{row['Métrica'].capitalize()}**: {row['Correlação TV Linear']:.2f} {row['Status']}")
            
            # Time series analysis for top metrics
            st.subheader("Evolução Temporal das Métricas de Maior Correlação")
            
            # Get top 3 metrics by correlation
            top_metrics = df_correlations.head(3)
            
            # Prepare data for time series
            time_data = []
            
            for _, row in top_metrics.iterrows():
                platform = row['Plataforma']
                metric = row['Métrica']
                
                # Get columns for this metric and platform
                metric_cols = [
                    col for col in rs_cols 
                    if f"_{platform}_" in col and (
                        col.endswith(f"_{metric}") or 
                        f"_{metric}" in col or 
                        metric in col.lower()
                    )
                ]
                
                if metric_cols:
                    df_analysis[f'{platform}_{metric}_combined'] = df_analysis[metric_cols].sum(axis=1)
                    
                    for _, data_row in df_analysis.iterrows():
                        time_data.append({
                            "Data": data_row['data_hora'] if 'data_hora' in df_analysis.columns else data_row['period'],
                            "Métrica": f"{platform} {metric.capitalize()}",
                            "Valor": data_row[f'{platform}_{metric}_combined']
                        })
            
            # Create time series chart with TV Linear on secondary y-axis
            if time_data:
                df_time = pd.DataFrame(time_data)
                
                # Create figure with secondary y-axis
                fig_time = make_subplots(specs=[[{"secondary_y": True}]])
                
                # Add traces for each metric-platform combination on primary y-axis
                for metric_name in df_time['Métrica'].unique():
                    metric_data = df_time[df_time['Métrica'] == metric_name]
                    if not metric_data.empty:
                        fig_time.add_trace(
                            go.Scatter(
                                x=metric_data['Data'],
                                y=metric_data['Valor'],
                                name=metric_name,
                                mode='lines'
                            ),
                            secondary_y=False
                        )
                
                # Add TV Linear data on secondary y-axis if available
                if 'LINEAR_GLOBO_cov%' in df_analysis.columns:
                    fig_time.add_trace(
                        go.Scatter(
                            x=df_analysis['data_hora'] if 'data_hora' in df_analysis.columns else df_analysis['period'],
                            y=df_analysis['LINEAR_GLOBO_cov%'],
                            name='TV Linear (cov%)',
                            mode='lines',
                            line=dict(color='red', width=2, dash='dash')
                        ),
                        secondary_y=True
                    )
                
                # Update axes titles
                fig_time.update_xaxes(title_text="Data")
                fig_time.update_yaxes(title_text="Valor da Métrica", secondary_y=False)
                fig_time.update_yaxes(title_text="TV Linear (cov%)", secondary_y=True)
                
                # Update layout
                fig_time.update_layout(
                    title="Evolução Temporal das Métricas de Maior Correlação vs. TV Linear",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_time, use_container_width=True)
            
            # 9. Predictive Model (Regression Equation)
            st.subheader("Modelagem Final e Equação Preditiva")
            
            st.markdown("""
            Esta seção apresenta uma equação preditiva baseada apenas em métricas de Redes Sociais
            e TV Linear, usando apenas variáveis com correlação significativa (>0.5).
            """)
            
            # Select variables with correlation > 0.5
            significant_vars = df_correlations[df_correlations["Correlação TV Linear"] > 0.5]
            
            if len(significant_vars) > 0:
                # Prepare data for regression
                X_vars = []
                
                for _, row in significant_vars.iterrows():
                    platform = row['Plataforma']
                    metric = row['Métrica']
                    
                    # Create combined variable name
                    var_name = f"{platform}_{metric}_combined"
                    
                    # Check if it exists in the dataframe
                    if var_name in df_analysis.columns:
                        X_vars.append(var_name)
                
                if X_vars and 'LINEAR_GLOBO_cov%' in df_analysis.columns:
                    # Prepare the dataframe
                    regression_df = df_analysis[X_vars + ['LINEAR_GLOBO_cov%']].dropna()
                    
                    if len(regression_df) > 5:  # Ensure we have enough data points
                        # Create X with constant term
                        X = sm.add_constant(regression_df[X_vars])
                        
                        # Create y (target variable)
                        y = regression_df['LINEAR_GLOBO_cov%']
                        
                        # Fit model
                        model = sm.OLS(y, X).fit()
                        
                        # Get coefficients
                        intercept = model.params[0]
                        coefficients = {X_vars[i]: model.params[i+1] for i in range(len(X_vars))}
                        
                        # Get R-squared
                        r_squared = model.rsquared
                        
                        # Format equation
                        equation = f"cov% TV Linear = {intercept:.2f}"
                        
                        for var, coef in coefficients.items():
                            # Get display name
                            parts = var.split('_')
                            platform = parts[0]
                            metric = parts[1].capitalize()  # Exclude 'combined'
                            
                            # Add term to equation
                            sign = "+" if coef > 0 else "-"
                            equation += f" {sign} ({abs(coef):.2f} × {metric} no {platform})"
                        
                        # Display equation
                        st.markdown(f"**Equação Estimada (R² = {r_squared:.2f}):**")
                        st.markdown(f"```\n{equation}\n```")
                        
                        st.markdown("""
                        Esta equação representa as métricas mais relevantes para prever a audiência da TV baseada em Redes Sociais.
                        Use-a para entender o impacto relativo de diferentes métricas e priorizar ações.
                        """)
                        
                        # Generate interpretation
                        st.markdown("**Interpretação:**")
                        
                        for var, coef in coefficients.items():
                            parts = var.split('_')
                            platform = parts[0]
                            metric = parts[1].capitalize()
                            
                            if coef > 0.3:
                                st.markdown(f"- Cada aumento de 1 unidade em **{metric} no {platform}** está associado a um aumento de **{coef:.2f} pontos percentuais** na cobertura TV Linear.")
                    else:
                        st.warning("Dados insuficientes para criar um modelo de regressão.")
                else:
                    st.info("Não foram encontradas variáveis com correlação forte o suficiente (>0.5) para incluir no modelo preditivo.")
            else:
                st.info("Não foram encontradas variáveis com correlação forte o suficiente (>0.5) para incluir no modelo preditivo.")
            
            # 10. Automatic Alerts
            st.subheader("Alertas Automáticos")
            
            # Check for TikTok low correlation
            tiktok_metrics = df_correlations[df_correlations['Plataforma'] == 'TIKTOK']
            if not tiktok_metrics.empty:
                avg_tiktok_corr = tiktok_metrics['Correlação TV Linear'].mean()
                
                if avg_tiktok_corr < 0.2:
                    st.warning(f"⚠️ TikTok continua com baixa correlação com TV Linear ({avg_tiktok_corr:.2f} < 0.2). Pode ser um canal efetivo para awareness, mas não para performance em termos de impacto direto na audiência TV.")
            
            # Check for posting frequency
            if 'period' in df_analysis.columns:
                # Get post quantity columns
                posts_cols = [col for col in rs_cols if 'posts_quantity' in col]
                
                if posts_cols:
                    # Calculate daily posts
                    daily_posts = df_analysis.groupby('period')[posts_cols].sum().sum(axis=1)
                    
                    # Calculate 7-day moving average
                    if len(daily_posts) >= 7:
                        daily_posts_df = pd.DataFrame(daily_posts).reset_index()
                        daily_posts_df.columns = ['date', 'posts']
                        daily_posts_df['rolling_avg'] = daily_posts_df['posts'].rolling(7).mean()
                        
                        # Check for days with posts 20% below moving average
                        daily_posts_df['below_threshold'] = daily_posts_df['posts'] < daily_posts_df['rolling_avg'] * 0.8
                        
                        below_threshold_days = daily_posts_df['below_threshold'].sum()
                        
                        if below_threshold_days > len(daily_posts_df) * 0.2:
                            st.warning(f"⚠️ Em {below_threshold_days} dias ({(below_threshold_days/len(daily_posts_df)*100):.1f}% do período), o número de posts ficou 20% abaixo da média móvel de 7 dias. Isso pode levar à perda de presença orgânica e engajamento.")
        else:
            st.warning("Dados insuficientes para análise por tipo de métrica.")