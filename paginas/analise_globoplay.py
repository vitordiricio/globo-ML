# utils/analise_globoplay.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
    # 2. Granularity Selection Dropdown
    granularity_options = {
        "Diário": df_daily,
        "Semanal": df_weekly,
    }
    
    st.markdown("""
    ### Análise do Consumo de Globoplay

    Esta análise explora os padrões de consumo do Globoplay, examinando diferentes tipos de usuários, dispositivos utilizados e tipos de conteúdo consumido. Os dados são agrupados por período de tempo para identificar tendências gerais e correlações com a audiência da TV Linear.
    """)
    
    st.markdown("""
    **Recomendação:** Para análises mais detalhadas e precisas, recomenda-se utilizar a granularidade **Diária**, pois oferece uma visão mais granular do comportamento dos usuários e permite identificar padrões que podem ser mascarados em agrupamentos semanais.
    """)
    
    granularity = st.selectbox(
        "Selecione a granularidade:",
        options=list(granularity_options.keys())
    )
    
    # Get the selected dataframe
    selected_df = granularity_options[granularity]
    
    # 3. Metrics Tables
    st.subheader("Métricas Resumidas")
    
    st.markdown("""
    As tabelas abaixo apresentam métricas-chave do Globoplay organizadas em três grupos principais:
    
    1. **Tipos de Usuário**: Assinantes, Logados Free e Anônimos - mostra diferenças de comportamento por tipo de acesso
    2. **Dispositivos**: Mobile vs Outros Devices - revela preferências de visualização em diferentes plataformas
    3. **Tipos de Conteúdo**: Simulcasting (ao vivo) vs VOD (sob demanda) - ilustra padrões de consumo por formato
    
    Cada métrica inclui valor médio, desvio padrão e, quando aplicável, a média por usuário (que revela intensidade de uso).
    """)
    
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
            if col_name in data_df.columns:
                metrics_data.append({
                    "Métrica": label,
                    "Valor Médio": f"{data_df[col_name].mean():.2f}",
                    "Desvio Padrão": f"{data_df[col_name].std():.2f}",
                    "Número de Linhas": f"{len(data_df)}"
                })
            else:
                metrics_data.append({
                    "Métrica": label,
                    "Valor Médio": "N/A",
                    "Desvio Padrão": "N/A",
                    "Número de Linhas": "N/A"
                })
        
        return pd.DataFrame(metrics_data)
    
    # ROW 1: Assinantes vs Logados Free vs Anônimos
    row1_col1, row1_col2 = st.columns(2)
    
    # Left column: Reach metrics (users)
    with row1_col1:
        st.markdown("### Usuários: Assinantes vs Logados Free vs Anônimos")
        
        reach_metrics = {
            "Usuários Assinantes": "GP_usuários_assinantes_",
            "Usuários Logados Free": "GP_usuários_de_vídeo_logados_free",
            "Usuários Anônimos": "GP_usuários_anonimos"
        }
        
        reach_df = create_metrics_table(selected_df, reach_metrics)
        st.dataframe(reach_df, hide_index=True)
    
    # Right column: Engagement metrics (hours) + avg per user
    with row1_col2:
        st.markdown("### Horas: Assinantes vs Logados Free vs Anônimos")
        
        engagement_metrics = {
            "Horas Consumidas Assinantes": "GP_horas_consumidas_assinantes",
            "Horas Consumidas Logados Free": "GP_horas_consumidas_de_logados_free",
            "Horas Consumidas Anônimos": "GP_horas_consumidas_de_anonimos"
        }
        
        # Create basic metrics table
        engagement_df = create_metrics_table(selected_df, engagement_metrics)
        
        # Calculate average hours per user
        avg_hours_data = []
        
        # Check if columns exist before calculating
        has_assinantes = ("GP_horas_consumidas_assinantes" in selected_df.columns and 
                          "GP_usuários_assinantes_" in selected_df.columns)
        has_logados = ("GP_horas_consumidas_de_logados_free" in selected_df.columns and 
                       "GP_usuários_de_vídeo_logados_free" in selected_df.columns)
        has_anonimos = ("GP_horas_consumidas_de_anonimos" in selected_df.columns and 
                        "GP_usuários_anonimos" in selected_df.columns)
        
        # Assinantes
        if has_assinantes:
            assinantes_hours = selected_df["GP_horas_consumidas_assinantes"].mean()
            assinantes_users = selected_df["GP_usuários_assinantes_"].mean()
            avg_assinantes = assinantes_hours / assinantes_users if assinantes_users > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Assinante",
                "Valor Médio": f"{avg_assinantes:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Logados Free
        if has_logados:
            logados_hours = selected_df["GP_horas_consumidas_de_logados_free"].mean()
            logados_users = selected_df["GP_usuários_de_vídeo_logados_free"].mean()
            avg_logados = logados_hours / logados_users if logados_users > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Logado Free",
                "Valor Médio": f"{avg_logados:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Anônimos
        if has_anonimos:
            anonimos_hours = selected_df["GP_horas_consumidas_de_anonimos"].mean()
            anonimos_users = selected_df["GP_usuários_anonimos"].mean()
            avg_anonimos = anonimos_hours / anonimos_users if anonimos_users > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Anônimo",
                "Valor Médio": f"{avg_anonimos:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Add to table if we have data
        if avg_hours_data:
            avg_hours_df = pd.DataFrame(avg_hours_data)
            
            # Combine tables
            combined_df = pd.concat([engagement_df, avg_hours_df])
            st.dataframe(combined_df, hide_index=True)
        else:
            st.dataframe(engagement_df, hide_index=True)
    
    # ROW 2: Mobile vs Outros Devices
    row2_col1, row2_col2 = st.columns(2)
    
    # Left column: Reach metrics (users)
    with row2_col1:
        st.markdown("### Usuários: Mobile vs Outros Devices")
        
        reach_metrics = {
            "Usuários Mobile": "GP_usuários_em_mobile",
            "Usuários Outros Devices": "GP_usuários_em_demais_devices"
        }
        
        reach_df = create_metrics_table(selected_df, reach_metrics)
        st.dataframe(reach_df, hide_index=True)
    
    # Right column: Engagement metrics (hours) + avg per user
    with row2_col2:
        st.markdown("### Horas: Mobile vs Outros Devices")
        
        engagement_metrics = {
            "Horas Consumidas Mobile": "GP_horas_consumidas_mobile",
            "Horas Consumidas Outros Devices": "GP_horas_consumidas_em_demais_devices"
        }
        
        # Create basic metrics table
        engagement_df = create_metrics_table(selected_df, engagement_metrics)
        
        # Calculate average hours per user
        avg_hours_data = []
        
        # Check if columns exist before calculating
        has_mobile = ("GP_horas_consumidas_mobile" in selected_df.columns and 
                      "GP_usuários_em_mobile" in selected_df.columns)
        has_outros = ("GP_horas_consumidas_em_demais_devices" in selected_df.columns and 
                      "GP_usuários_em_demais_devices" in selected_df.columns)
        
        # Mobile
        if has_mobile:
            mobile_hours = selected_df["GP_horas_consumidas_mobile"].mean()
            mobile_users = selected_df["GP_usuários_em_mobile"].mean()
            avg_mobile = mobile_hours / mobile_users if mobile_users > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Usuário Mobile",
                "Valor Médio": f"{avg_mobile:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Outros Devices
        if has_outros:
            outros_hours = selected_df["GP_horas_consumidas_em_demais_devices"].mean()
            outros_users = selected_df["GP_usuários_em_demais_devices"].mean()
            avg_outros = outros_hours / outros_users if outros_users > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Usuário Outros Devices",
                "Valor Médio": f"{avg_outros:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Add to table if we have data
        if avg_hours_data:
            avg_hours_df = pd.DataFrame(avg_hours_data)
            
            # Combine tables
            combined_df = pd.concat([engagement_df, avg_hours_df])
            st.dataframe(combined_df, hide_index=True)
        else:
            st.dataframe(engagement_df, hide_index=True)
    
    # ROW 3: Simulcasting vs VOD
    row3_col1, row3_col2 = st.columns(2)
    
    # Left column: Reach metrics (users)
    with row3_col1:
        st.markdown("### Usuários: Simulcasting (TVG ao Vivo) vs VOD")
        
        reach_metrics = {
            "Usuários TVG ao Vivo": "GP_usuários_em_tvg_ao_vivo",
            "Qtd Íntegras Publicadas": "GP_qtd_de_integras_publicadas"
        }
        
        reach_df = create_metrics_table(selected_df, reach_metrics)
        st.dataframe(reach_df, hide_index=True)
    
    # Right column: Engagement metrics (hours) + avg per user
    with row3_col2:
        st.markdown("### Horas: Simulcasting (TVG ao Vivo) vs VOD")
        
        engagement_metrics = {
            "Horas Consumidas TVG ao Vivo": "GP_horas_consumidas_em_tvg_ao_vivo",
            "Horas Disponíveis Íntegras": "GP_qtd_de_horas_disponíveis_integras"
        }
        
        # Create basic metrics table
        engagement_df = create_metrics_table(selected_df, engagement_metrics)
        
        # Calculate average hours per user/item
        avg_hours_data = []
        
        # Check if columns exist before calculating
        has_tvg = ("GP_horas_consumidas_em_tvg_ao_vivo" in selected_df.columns and 
                   "GP_usuários_em_tvg_ao_vivo" in selected_df.columns)
        has_vod = ("GP_qtd_de_horas_disponíveis_integras" in selected_df.columns and 
                   "GP_qtd_de_integras_publicadas" in selected_df.columns)
        
        # TVG ao Vivo
        if has_tvg:
            tvg_hours = selected_df["GP_horas_consumidas_em_tvg_ao_vivo"].mean()
            tvg_users = selected_df["GP_usuários_em_tvg_ao_vivo"].mean()
            avg_tvg = tvg_hours / tvg_users if tvg_users > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Usuário TVG ao Vivo",
                "Valor Médio": f"{avg_tvg:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Íntegras (VOD) - horas por íntegra publicada
        if has_vod:
            vod_hours = selected_df["GP_qtd_de_horas_disponíveis_integras"].mean()
            vod_items = selected_df["GP_qtd_de_integras_publicadas"].mean()
            avg_vod = vod_hours / vod_items if vod_items > 0 else 0
            avg_hours_data.append({
                "Métrica": "Horas Médias por Íntegra Publicada",
                "Valor Médio": f"{avg_vod:.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Add to table if we have data
        if avg_hours_data:
            avg_hours_df = pd.DataFrame(avg_hours_data)
            
            # Combine tables
            combined_df = pd.concat([engagement_df, avg_hours_df])
            st.dataframe(combined_df, hide_index=True)
        else:
            st.dataframe(engagement_df, hide_index=True)
    
    # Now add correlation with TV Linear (if available)
    if 'LINEAR_GLOBO_cov%' in selected_df.columns:
        st.markdown("""
        ### Correlação com TV Linear
        
        As métricas abaixo mostram a correlação entre diferentes categorias de uso do Globoplay e a cobertura da TV Linear (cov%). 
        Valores próximos a 1 indicam forte relação positiva (aumentam juntos), valores próximos a 0 indicam pouca relação, 
        e valores negativos indicam relação inversa (quando um aumenta, o outro diminui).
        """)
        
        # Create columns for correlation cards
        corr_cols = st.columns(3)
        
        # Row 1 - Assinantes vs Logados vs Anônimos
        with corr_cols[0]:
            st.subheader("Tipo de Usuário")
            
            # Calculate correlations
            corr_metrics = []
            
            if "GP_horas_consumidas_assinantes" in selected_df.columns:
                corr_assinantes = selected_df['GP_horas_consumidas_assinantes'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("Assinantes", f"{corr_assinantes:.2f}")
            
            if "GP_horas_consumidas_de_logados_free" in selected_df.columns:
                corr_logados = selected_df['GP_horas_consumidas_de_logados_free'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("Logados Free", f"{corr_logados:.2f}")
            
            if "GP_horas_consumidas_de_anonimos" in selected_df.columns:
                corr_anonimos = selected_df['GP_horas_consumidas_de_anonimos'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("Anônimos", f"{corr_anonimos:.2f}")
        
        # Row 2 - Mobile vs Outros Devices
        with corr_cols[1]:
            st.subheader("Tipo de Device")
            
            # Calculate correlations
            if "GP_horas_consumidas_mobile" in selected_df.columns:
                corr_mobile = selected_df['GP_horas_consumidas_mobile'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("Mobile", f"{corr_mobile:.2f}")
            
            if "GP_horas_consumidas_em_demais_devices" in selected_df.columns:
                corr_outros = selected_df['GP_horas_consumidas_em_demais_devices'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("Outros Devices", f"{corr_outros:.2f}")
        
        # Row 3 - Simulcasting vs VOD
        with corr_cols[2]:
            st.subheader("Tipo de Conteúdo")
            
            # Calculate correlations
            if "GP_horas_consumidas_em_tvg_ao_vivo" in selected_df.columns:
                corr_tvg = selected_df['GP_horas_consumidas_em_tvg_ao_vivo'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("TVG ao Vivo", f"{corr_tvg:.2f}")
            
            if "GP_qtd_de_horas_disponíveis_integras" in selected_df.columns:
                corr_vod = selected_df['GP_qtd_de_horas_disponíveis_integras'].corr(selected_df['LINEAR_GLOBO_cov%'])
                st.metric("VOD (Íntegras)", f"{corr_vod:.2f}")
    
    # Visualization Section
    st.subheader("Visualizações")
    
    st.markdown("""
    Os gráficos a seguir ilustram a evolução temporal do consumo de Globoplay, segmentado por diferentes 
    categorias e comparado com a audiência da TV Linear (quando disponível). Estes gráficos permitem 
    identificar tendências, sazonalidades e relações entre streaming e TV tradicional.
    
    Observe padrões como:
    - Picos de consumo em determinados períodos
    - Diferenças no comportamento entre segmentos de usuários
    - Relação entre audiência de streaming e TV Linear
    """)
    
    # Only include visualizations if we have the necessary data
    
    # Row 1 - Assinantes vs Logados vs Anônimos
    has_user_data = any(col in selected_df.columns for col in [
        'GP_horas_consumidas_assinantes', 
        'GP_horas_consumidas_de_logados_free', 
        'GP_horas_consumidas_de_anonimos'
    ])
    
    if has_user_data:
        st.markdown("#### Engajamento por Tipo de Usuário vs. TV Linear")
        
        # Prepare data for the chart
        plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
        
        if 'GP_horas_consumidas_assinantes' in selected_df.columns:
            plot_data['Horas Consumidas Assinantes'] = selected_df['GP_horas_consumidas_assinantes']
        
        if 'GP_horas_consumidas_de_logados_free' in selected_df.columns:
            plot_data['Horas Consumidas Logados Free'] = selected_df['GP_horas_consumidas_de_logados_free']
        
        if 'GP_horas_consumidas_de_anonimos' in selected_df.columns:
            plot_data['Horas Consumidas Anônimos'] = selected_df['GP_horas_consumidas_de_anonimos']
        
        if 'LINEAR_GLOBO_cov%' in selected_df.columns:
            plot_data['cov% TV Linear'] = selected_df['LINEAR_GLOBO_cov%']
        
        # Get y-columns (exclude 'Data')
        y_cols = [col for col in plot_data.columns if col != 'Data']
        
        if y_cols:  # Only create chart if we have y data
            # Separar as colunas para eixos esquerdo e direito
            left_y_cols = [col for col in y_cols if col != 'cov% TV Linear']
            right_y_cols = ['cov% TV Linear'] if 'cov% TV Linear' in y_cols else []
            
            # Criar figura com subplots compartilhando o eixo x
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adicionar linhas para o eixo esquerdo
            for col in left_y_cols:
                fig.add_trace(
                    go.Scatter(x=plot_data['Data'], y=plot_data[col], name=col),
                    secondary_y=False
                )
            
            # Adicionar linhas para o eixo direito
            for col in right_y_cols:
                fig.add_trace(
                    go.Scatter(x=plot_data['Data'], y=plot_data[col], name=col),
                    secondary_y=True
                )
            
            # Atualizar títulos dos eixos
            fig.update_yaxes(title_text="Horas Consumidas", secondary_y=False)
            fig.update_yaxes(title_text="Coverage %", secondary_y=True)
            fig.update_xaxes(title_text="Data")
            
            # Configurar layout
            fig.update_layout(
                title=f"Engajamento por Tipo de Usuário - {granularity}",
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretação:** Este gráfico mostra como o consumo varia entre diferentes tipos de usuários ao longo do tempo.
            Observe que assinantes geralmente têm maior consumo por pessoa, enquanto usuários logados free e anônimos
            apresentam padrões de consumo mais leves. A linha de cobertura TV Linear permite avaliar se há
            complementaridade ou canibalização entre streaming e TV tradicional.
            """)
        
    # Row 2 - Mobile vs Outros Devices
    has_device_data = any(col in selected_df.columns for col in [
        'GP_horas_consumidas_mobile', 
        'GP_horas_consumidas_em_demais_devices'
    ])
    
    if has_device_data:
        st.markdown("#### Engajamento por Tipo de Device vs. TV Linear")
        
        # Prepare data for the chart
        plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
        
        if 'GP_horas_consumidas_mobile' in selected_df.columns:
            plot_data['Horas Consumidas Mobile'] = selected_df['GP_horas_consumidas_mobile']
        
        if 'GP_horas_consumidas_em_demais_devices' in selected_df.columns:
            plot_data['Horas Consumidas Outros Devices'] = selected_df['GP_horas_consumidas_em_demais_devices']
        
        if 'LINEAR_GLOBO_cov%' in selected_df.columns:
            plot_data['cov% TV Linear'] = selected_df['LINEAR_GLOBO_cov%']
        
        # Get y-columns (exclude 'Data')
        y_cols = [col for col in plot_data.columns if col != 'Data']
        
        if y_cols:  # Only create chart if we have y data
            # Separar as colunas para eixos esquerdo e direito
            left_y_cols = [col for col in y_cols if col != 'cov% TV Linear']
            right_y_cols = ['cov% TV Linear'] if 'cov% TV Linear' in y_cols else []
            
            # Importar módulos necessários (adicione no início do arquivo)
            # from plotly.subplots import make_subplots
            # import plotly.graph_objects as go
            
            # Criar figura com subplots compartilhando o eixo x
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adicionar linhas para o eixo esquerdo
            for col in left_y_cols:
                fig.add_trace(
                    go.Scatter(x=plot_data['Data'], y=plot_data[col], name=col),
                    secondary_y=False
                )
            
            # Adicionar linhas para o eixo direito
            for col in right_y_cols:
                fig.add_trace(
                    go.Scatter(x=plot_data['Data'], y=plot_data[col], name=col),
                    secondary_y=True
                )
            
            # Atualizar títulos dos eixos
            fig.update_yaxes(title_text="Horas Consumidas", secondary_y=False)
            fig.update_yaxes(title_text="Coverage %", secondary_y=True)
            fig.update_xaxes(title_text="Data")
            
            # Configurar layout
            fig.update_layout(
                title=f"Engajamento por Tipo de Device - {granularity}",
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretação:** Este gráfico ilustra como o consumo se distribui entre dispositivos móveis e outros tipos de tela.
            Tendências crescentes no consumo mobile podem indicar mudanças importantes no comportamento da audiência,
            como maior consumo em deslocamento ou preferência por telas menores para determinados conteúdos.
            Observe também como eventos sazonais podem afetar diferentemente o consumo em cada tipo de dispositivo.
            """)
    
    # Row 3 - Simulcasting vs VOD
    has_content_data = any(col in selected_df.columns for col in [
        'GP_horas_consumidas_em_tvg_ao_vivo', 
        'GP_qtd_de_horas_disponíveis_integras'
    ])
    
    if has_content_data:
        st.markdown("#### Simulcasting vs VOD vs. TV Linear")
        
        # Prepare data for the chart
        plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
        
        if 'GP_horas_consumidas_em_tvg_ao_vivo' in selected_df.columns:
            plot_data['Horas Consumidas TVG ao Vivo'] = selected_df['GP_horas_consumidas_em_tvg_ao_vivo']
        
        if 'GP_qtd_de_horas_disponíveis_integras' in selected_df.columns:
            plot_data['Horas Disponíveis VOD'] = selected_df['GP_qtd_de_horas_disponíveis_integras']
        
        if 'LINEAR_GLOBO_cov%' in selected_df.columns:
            plot_data['cov% TV Linear'] = selected_df['LINEAR_GLOBO_cov%']
        
        # Get y-columns (exclude 'Data')
        y_cols = [col for col in plot_data.columns if col != 'Data']
        
        if y_cols:  # Only create chart if we have y data
            # Vamos supor que queremos 'cov% TV Linear' no eixo direito
            left_y_cols = [col for col in y_cols if col != 'cov% TV Linear']
            right_y_cols = ['cov% TV Linear'] if 'cov% TV Linear' in y_cols else []
            
            # Criar figura com subplots compartilhando o eixo x
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Adicionar linhas para o eixo esquerdo
            for col in left_y_cols:
                fig.add_trace(
                    go.Scatter(x=plot_data['Data'], y=plot_data[col], name=col),
                    secondary_y=False
                )
            
            # Adicionar linhas para o eixo direito
            for col in right_y_cols:
                fig.add_trace(
                    go.Scatter(x=plot_data['Data'], y=plot_data[col], name=col),
                    secondary_y=True
                )
            
            # Atualizar títulos dos eixos
            fig.update_yaxes(title_text="Horas", secondary_y=False)
            fig.update_yaxes(title_text="Coverage %", secondary_y=True)
            fig.update_xaxes(title_text="Data")
            
            # Configurar layout
            fig.update_layout(
                title=f"Simulcasting vs VOD - {granularity}",
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretação:** Este gráfico compara o consumo de conteúdo ao vivo (simulcasting) versus conteúdo sob demanda (VOD).
            Observe como eventos especiais ou temporadas de programas populares podem causar picos no consumo ao vivo,
            enquanto o consumo VOD tende a ser mais estável ao longo do tempo. A relação com a audiência de TV Linear
            fornece insights sobre a possível complementaridade ou competição entre estas modalidades de consumo.
            """)
    
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