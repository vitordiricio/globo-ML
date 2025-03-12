# utils/analise_globoplay.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
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

    1. **Audiência**: Mostra o volume de usuários por tipo de acesso
    2. **Engajamento**: Revela as horas totais de consumo por tipo de usuário
    3. **Ativação**: Indica a intensidade de uso (horas médias por usuário)

    Cada métrica oferece uma perspectiva diferente sobre o comportamento dos usuários na plataforma.
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

    # Function to create pie charts
    def create_pie_chart(data_dict, title):
        if not data_dict:
            return None
        
        labels = list(data_dict.keys())
        values = list(data_dict.values())
        
        fig = px.pie(
            values=values,
            names=labels,
            title=title,
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=300,
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        )
        
        return fig

    # ROW 1: Assinantes vs Logados Free vs Anônimos
    st.markdown("### Usuários: Assinantes vs Logados Free vs Anônimos")

    # Get values for pie charts - User Types
    user_type_values = {}
    user_type_hours = {}
    user_type_avg_hours = {}

    # Check if columns exist before calculating
    has_assinantes = ("GP_horas_consumidas_assinantes" in selected_df.columns and 
                    "GP_usuários_assinantes_" in selected_df.columns)
    has_logados = ("GP_horas_consumidas_de_logados_free" in selected_df.columns and 
                "GP_usuários_de_vídeo_logados_free" in selected_df.columns)
    has_anonimos = ("GP_horas_consumidas_de_anonimos" in selected_df.columns and 
                    "GP_usuários_anonimos" in selected_df.columns)

    # Users
    if has_assinantes:
        user_type_values["Assinantes"] = selected_df["GP_usuários_assinantes_"].mean()
    if has_logados:
        user_type_values["Logados Free"] = selected_df["GP_usuários_de_vídeo_logados_free"].mean()
    if has_anonimos:
        user_type_values["Anônimos"] = selected_df["GP_usuários_anonimos"].mean()

    # Hours
    if has_assinantes:
        user_type_hours["Assinantes"] = selected_df["GP_horas_consumidas_assinantes"].mean()
    if has_logados:
        user_type_hours["Logados Free"] = selected_df["GP_horas_consumidas_de_logados_free"].mean()
    if has_anonimos:
        user_type_hours["Anônimos"] = selected_df["GP_horas_consumidas_de_anonimos"].mean()

    # Average hours per user
    if has_assinantes:
        assinantes_hours = selected_df["GP_horas_consumidas_assinantes"].mean()
        assinantes_users = selected_df["GP_usuários_assinantes_"].mean()
        user_type_avg_hours["Assinantes"] = assinantes_hours / assinantes_users if assinantes_users > 0 else 0

    if has_logados:
        logados_hours = selected_df["GP_horas_consumidas_de_logados_free"].mean()
        logados_users = selected_df["GP_usuários_de_vídeo_logados_free"].mean()
        user_type_avg_hours["Logados Free"] = logados_hours / logados_users if logados_users > 0 else 0

    if has_anonimos:
        anonimos_hours = selected_df["GP_horas_consumidas_de_anonimos"].mean()
        anonimos_users = selected_df["GP_usuários_anonimos"].mean()
        user_type_avg_hours["Anônimos"] = anonimos_hours / anonimos_users if anonimos_users > 0 else 0

    # Create three columns for tables
    col1, col2, col3 = st.columns(3)

    # First column: Audiência (Users)
    with col1:
        st.markdown("#### Audiência")
        reach_metrics = {
            "Usuários Assinantes": "GP_usuários_assinantes_",
            "Usuários Logados Free": "GP_usuários_de_vídeo_logados_free",
            "Usuários Anônimos": "GP_usuários_anonimos"
        }
        
        reach_df = create_metrics_table(selected_df, reach_metrics)
        st.dataframe(reach_df, hide_index=True)

    # Second column: Engajamento (Hours)
    with col2:
        st.markdown("#### Engajamento")
        engagement_metrics = {
            "Horas Consumidas Assinantes": "GP_horas_consumidas_assinantes",
            "Horas Consumidas Logados Free": "GP_horas_consumidas_de_logados_free",
            "Horas Consumidas Anônimos": "GP_horas_consumidas_de_anonimos"
        }
        
        engagement_df = create_metrics_table(selected_df, engagement_metrics)
        st.dataframe(engagement_df, hide_index=True)

    # Third column: Ativação (Average Hours)
    with col3:
        st.markdown("#### Ativação")
        avg_hours_data = []
        
        # Add average hours data if available
        if user_type_avg_hours:
            for user_type, avg_hours in user_type_avg_hours.items():
                avg_hours_data.append({
                    "Métrica": f"Horas Médias por {user_type}",
                    "Valor Médio": f"{avg_hours:.2f}",
                    "Desvio Padrão": "N/A",
                    "Número de Linhas": f"{len(selected_df)}"
                })
        
        # Add to table if we have data
        if avg_hours_data:
            avg_hours_df = pd.DataFrame(avg_hours_data)
            st.dataframe(avg_hours_df, hide_index=True)

    # Create three columns for charts
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # First column: Users pie chart
    with chart_col1:
        if user_type_values:
            users_pie = create_pie_chart(user_type_values, "Distribuição de Usuários")
            st.plotly_chart(users_pie, use_container_width=True)
    
    # Second column: Hours pie chart
    with chart_col2:
        if user_type_hours:
            hours_pie = create_pie_chart(user_type_hours, "Distribuição de Horas Consumidas")
            st.plotly_chart(hours_pie, use_container_width=True)
    
    # Third column: Average hours pie chart
    with chart_col3:
        if user_type_avg_hours:
            avg_hours_pie = create_pie_chart(user_type_avg_hours, "Horas Médias por Usuário")
            st.plotly_chart(avg_hours_pie, use_container_width=True)

    # Add insights for user types
    if user_type_values and user_type_hours and user_type_avg_hours:
        # Determine which user type has the highest average hours
        highest_avg = max(user_type_avg_hours.items(), key=lambda x: x[1])
        highest_user_share = user_type_values[highest_avg[0]] / sum(user_type_values.values()) * 100
        highest_hours_share = user_type_hours[highest_avg[0]] / sum(user_type_hours.values()) * 100
        
        st.info(f"""
        **Insights - Tipos de Usuário:**
        
        📊 **{highest_avg[0]}** apresentam o maior consumo médio com **{highest_avg[1]:.2f} horas/usuário**.
        
        👥 Este grupo representa **{highest_user_share:.1f}%** dos usuários, mas gera **{highest_hours_share:.1f}%** do consumo total de conteúdo.
        
        💡 **Oportunidade:** {
            "Focar em converter usuários anônimos e logados free para assinantes, dado o alto engajamento desse grupo." 
            if highest_avg[0] == "Assinantes" else
            "Explorar estratégias para monetizar o alto engajamento deste grupo através de ofertas personalizadas."
        }
        """)

    # ROW 2: Mobile vs Outros Devices
    st.markdown("### Usuários: Mobile vs Outros Devices")

    # Get values for pie charts - Devices
    device_type_values = {}
    device_type_hours = {}
    device_type_avg_hours = {}

    # Check if columns exist before calculating
    has_mobile = ("GP_horas_consumidas_mobile" in selected_df.columns and 
                "GP_usuários_em_mobile" in selected_df.columns)
    has_outros = ("GP_horas_consumidas_em_demais_devices" in selected_df.columns and 
                "GP_usuários_em_demais_devices" in selected_df.columns)

    # Users
    if has_mobile:
        device_type_values["Mobile"] = selected_df["GP_usuários_em_mobile"].mean()
    if has_outros:
        device_type_values["Outros Devices"] = selected_df["GP_usuários_em_demais_devices"].mean()

    # Hours
    if has_mobile:
        device_type_hours["Mobile"] = selected_df["GP_horas_consumidas_mobile"].mean()
    if has_outros:
        device_type_hours["Outros Devices"] = selected_df["GP_horas_consumidas_em_demais_devices"].mean()

    # Average hours per user
    if has_mobile:
        mobile_hours = selected_df["GP_horas_consumidas_mobile"].mean()
        mobile_users = selected_df["GP_usuários_em_mobile"].mean()
        device_type_avg_hours["Mobile"] = mobile_hours / mobile_users if mobile_users > 0 else 0

    if has_outros:
        outros_hours = selected_df["GP_horas_consumidas_em_demais_devices"].mean()
        outros_users = selected_df["GP_usuários_em_demais_devices"].mean()
        device_type_avg_hours["Outros Devices"] = outros_hours / outros_users if outros_users > 0 else 0

    # Create three columns for tables
    col1, col2, col3 = st.columns(3)

    # First column: Audiência (Users)
    with col1:
        st.markdown("#### Audiência")
        reach_metrics = {
            "Usuários Mobile": "GP_usuários_em_mobile",
            "Usuários Outros Devices": "GP_usuários_em_demais_devices"
        }
        
        reach_df = create_metrics_table(selected_df, reach_metrics)
        st.dataframe(reach_df, hide_index=True)

    # Second column: Engajamento (Hours)
    with col2:
        st.markdown("#### Engajamento")
        engagement_metrics = {
            "Horas Consumidas Mobile": "GP_horas_consumidas_mobile",
            "Horas Consumidas Outros Devices": "GP_horas_consumidas_em_demais_devices"
        }
        
        engagement_df = create_metrics_table(selected_df, engagement_metrics)
        st.dataframe(engagement_df, hide_index=True)

    # Third column: Ativação (Average Hours)
    with col3:
        st.markdown("#### Ativação")
        avg_hours_data = []
        
        # Add average hours data if available
        if device_type_avg_hours:
            for device_type, avg_hours in device_type_avg_hours.items():
                avg_hours_data.append({
                    "Métrica": f"Horas Médias por Usuário {device_type}",
                    "Valor Médio": f"{avg_hours:.2f}",
                    "Desvio Padrão": "N/A",
                    "Número de Linhas": f"{len(selected_df)}"
                })
        
        # Add to table if we have data
        if avg_hours_data:
            avg_hours_df = pd.DataFrame(avg_hours_data)
            st.dataframe(avg_hours_df, hide_index=True)

    # Create three columns for charts
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # First column: Users pie chart
    with chart_col1:
        if device_type_values:
            users_pie = create_pie_chart(device_type_values, "Distribuição de Usuários por Dispositivo")
            st.plotly_chart(users_pie, use_container_width=True)
    
    # Second column: Hours pie chart
    with chart_col2:
        if device_type_hours:
            hours_pie = create_pie_chart(device_type_hours, "Distribuição de Horas Consumidas por Dispositivo")
            st.plotly_chart(hours_pie, use_container_width=True)
    
    # Third column: Average hours pie chart
    with chart_col3:
        if device_type_avg_hours:
            avg_hours_pie = create_pie_chart(device_type_avg_hours, "Horas Médias por Usuário por Dispositivo")
            st.plotly_chart(avg_hours_pie, use_container_width=True)

    # Add insights for device types
    if device_type_values and device_type_hours and device_type_avg_hours:
        # Compare mobile vs other devices
        mobile_user_pct = device_type_values.get("Mobile", 0) / sum(device_type_values.values()) * 100 if sum(device_type_values.values()) > 0 else 0
        mobile_hours_pct = device_type_hours.get("Mobile", 0) / sum(device_type_hours.values()) * 100 if sum(device_type_hours.values()) > 0 else 0
        
        mobile_avg = device_type_avg_hours.get("Mobile", 0)
        outros_avg = device_type_avg_hours.get("Outros Devices", 0)
        
        if mobile_avg > outros_avg:
            engagement_insight = "Os usuários mobile mostram maior engajamento por pessoa, sugerindo uma experiência mais imersiva em dispositivos móveis."
        elif outros_avg > mobile_avg:
            engagement_insight = "Usuários em outros dispositivos (como TVs e desktops) mostram maior engajamento por pessoa, provavelmente devido à experiência de tela maior."
        else:
            engagement_insight = "O engajamento por usuário é similar entre dispositivos móveis e outros, sugerindo uma experiência de qualidade em ambas as plataformas."
        
        st.info(f"""
        **Insights - Tipos de Dispositivo:**
        
        📱 Dispositivos móveis representam **{mobile_user_pct:.1f}%** dos usuários e **{mobile_hours_pct:.1f}%** do consumo total.
        
        ⏱️ {engagement_insight}
        
        💡 **Recomendação:** {
            "Priorizar otimizações para a experiência mobile, dado seu alto engajamento por usuário." 
            if mobile_avg > outros_avg else
            "Continuar investindo na experiência para dispositivos maiores, que geram maior tempo de visualização por usuário."
        }
        """)

    # ROW 3: Simulcasting vs VOD
    st.markdown("### Usuários: Simulcasting (TVG ao Vivo) vs VOD")

    # Get values for pie charts - Content Types
    content_type_values = {}
    content_type_hours = {}
    content_type_avg_hours = {}

    # Check if columns exist before calculating
    has_tvg = ("GP_horas_consumidas_em_tvg_ao_vivo" in selected_df.columns and 
            "GP_usuários_em_tvg_ao_vivo" in selected_df.columns)
    has_vod = ("GP_qtd_de_horas_disponíveis_integras" in selected_df.columns and 
            "GP_qtd_de_integras_publicadas" in selected_df.columns)

    # Users/Items
    if has_tvg:
        content_type_values["TVG ao Vivo"] = selected_df["GP_usuários_em_tvg_ao_vivo"].mean()
    if has_vod:
        content_type_values["VOD (Íntegras)"] = selected_df["GP_qtd_de_integras_publicadas"].mean()

    # Hours
    if has_tvg:
        content_type_hours["TVG ao Vivo"] = selected_df["GP_horas_consumidas_em_tvg_ao_vivo"].mean()
    if has_vod:
        content_type_hours["VOD (Íntegras)"] = selected_df["GP_qtd_de_horas_disponíveis_integras"].mean()

    # Average hours per user/item
    if has_tvg:
        tvg_hours = selected_df["GP_horas_consumidas_em_tvg_ao_vivo"].mean()
        tvg_users = selected_df["GP_usuários_em_tvg_ao_vivo"].mean()
        content_type_avg_hours["TVG ao Vivo"] = tvg_hours / tvg_users if tvg_users > 0 else 0

    if has_vod:
        vod_hours = selected_df["GP_qtd_de_horas_disponíveis_integras"].mean()
        vod_items = selected_df["GP_qtd_de_integras_publicadas"].mean()
        content_type_avg_hours["VOD (Íntegras)"] = vod_hours / vod_items if vod_items > 0 else 0

    # Create three columns for tables
    col1, col2, col3 = st.columns(3)

    # First column: Audiência (Users)
    with col1:
        st.markdown("#### Audiência")
        reach_metrics = {
            "Usuários TVG ao Vivo": "GP_usuários_em_tvg_ao_vivo",
            "Qtd Íntegras Publicadas": "GP_qtd_de_integras_publicadas"
        }
        
        reach_df = create_metrics_table(selected_df, reach_metrics)
        st.dataframe(reach_df, hide_index=True)

    # Second column: Engajamento (Hours)
    with col2:
        st.markdown("#### Engajamento")
        engagement_metrics = {
            "Horas Consumidas TVG ao Vivo": "GP_horas_consumidas_em_tvg_ao_vivo",
            "Horas Disponíveis Íntegras": "GP_qtd_de_horas_disponíveis_integras"
        }
        
        engagement_df = create_metrics_table(selected_df, engagement_metrics)
        st.dataframe(engagement_df, hide_index=True)

    # Third column: Ativação (Average Hours)
    with col3:
        st.markdown("#### Ativação")
        avg_hours_data = []
        
        # Add average hours data if available
        if has_tvg:
            avg_hours_data.append({
                "Métrica": "Horas Médias por Usuário TVG ao Vivo",
                "Valor Médio": f"{content_type_avg_hours.get('TVG ao Vivo', 0):.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        if has_vod:
            avg_hours_data.append({
                "Métrica": "Horas Médias por Íntegra Publicada",
                "Valor Médio": f"{content_type_avg_hours.get('VOD (Íntegras)', 0):.2f}",
                "Desvio Padrão": "N/A",
                "Número de Linhas": f"{len(selected_df)}"
            })
        
        # Add to table if we have data
        if avg_hours_data:
            avg_hours_df = pd.DataFrame(avg_hours_data)
            st.dataframe(avg_hours_df, hide_index=True)

    # Create three columns for charts
    chart_col1, chart_col2, chart_col3 = st.columns(3)

    # First column: Users pie chart
    with chart_col1:
        if content_type_values:
            users_pie = create_pie_chart(content_type_values, "Distribuição de Usuários/Itens por Tipo de Conteúdo")
            st.plotly_chart(users_pie, use_container_width=True)
    
    # Second column: Hours pie chart
    with chart_col2:
        if content_type_hours:
            hours_pie = create_pie_chart(content_type_hours, "Distribuição de Horas por Tipo de Conteúdo")
            st.plotly_chart(hours_pie, use_container_width=True)
    
    # Third column: Average hours pie chart
    with chart_col3:
        if content_type_avg_hours:
            avg_hours_pie = create_pie_chart(content_type_avg_hours, "Horas Médias por Usuário/Item por Tipo de Conteúdo")
            st.plotly_chart(avg_hours_pie, use_container_width=True)

    # Add insights for content types
    if content_type_values and content_type_hours:
        # Compare live vs VOD
        tvg_hours_pct = content_type_hours.get("TVG ao Vivo", 0) / sum(content_type_hours.values()) * 100 if sum(content_type_hours.values()) > 0 else 0
        
        if has_tvg and has_vod:
            st.success(f"""
            **Insights - Tipos de Conteúdo:**
            
            📺 O conteúdo ao vivo (TVG) representa **{tvg_hours_pct:.1f}%** das horas consumidas.
            
            🔍 **Análise:** {
                "O consumo é dominado por conteúdo ao vivo, sugerindo alto valor para transmissões em tempo real." 
                if tvg_hours_pct > 60 else
                "Existe um equilíbrio saudável entre consumo ao vivo e sob demanda, indicando uma plataforma versátil." 
                if tvg_hours_pct > 40 else
                "O consumo é dominado por conteúdo sob demanda (VOD), refletindo a tendência de assistir no próprio ritmo."
            }
            
            💡 **Oportunidade:** {
                "Aproveitar o forte interesse em conteúdo ao vivo para criar mais eventos especiais e transmissões exclusivas." 
                if tvg_hours_pct > 60 else
                "Manter o equilíbrio atual entre transmissões ao vivo e conteúdo sob demanda, que atende diferentes perfis de usuários." 
                if tvg_hours_pct > 40 else
                "Expandir o catálogo de conteúdo sob demanda e aprimorar as ferramentas de descoberta de conteúdo para potencializar ainda mais o consumo VOD."
            }
            """)

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
        
        # Add insight about correlations
        # Find highest correlation
        corr_values = []
        corr_labels = []
        
        if "GP_horas_consumidas_assinantes" in selected_df.columns:
            corr_values.append(corr_assinantes)
            corr_labels.append("Assinantes")
        
        if "GP_horas_consumidas_de_logados_free" in selected_df.columns:
            corr_values.append(corr_logados)
            corr_labels.append("Usuários Logados Free")
        
        if "GP_horas_consumidas_de_anonimos" in selected_df.columns:
            corr_values.append(corr_anonimos)
            corr_labels.append("Usuários Anônimos")
        
        if "GP_horas_consumidas_mobile" in selected_df.columns:
            corr_values.append(corr_mobile)
            corr_labels.append("Mobile")
        
        if "GP_horas_consumidas_em_demais_devices" in selected_df.columns:
            corr_values.append(corr_outros)
            corr_labels.append("Outros Dispositivos")
        
        if "GP_horas_consumidas_em_tvg_ao_vivo" in selected_df.columns:
            corr_values.append(corr_tvg)
            corr_labels.append("TVG ao Vivo")
        
        if "GP_qtd_de_horas_disponíveis_integras" in selected_df.columns:
            corr_values.append(corr_vod)
            corr_labels.append("VOD (Íntegras)")
        
        if corr_values and corr_labels:
            max_corr_idx = corr_values.index(max(corr_values, key=abs))
            max_corr_value = corr_values[max_corr_idx]
            max_corr_label = corr_labels[max_corr_idx]
            
            relationship_type = "complementar" if max_corr_value > 0 else "inversa"
            
            st.warning(f"""
            **Insight sobre correlações:**
            
            📊 A mais forte correlação encontrada é entre **{max_corr_label}** e TV Linear: **{max_corr_value:.2f}**
            
            🔍 **Interpretação:** Existe uma relação **{relationship_type}** significativa, o que sugere que {
                "o consumo em streaming e TV Linear crescem juntos, reforçando-se mutuamente." 
                if max_corr_value > 0 else
                "quando o consumo em um formato aumenta, o outro tende a diminuir, indicando uma possível substituição de mídia."
            }
            
            💡 **Implicação estratégica:** {
                "Aproveitar esta sinergia com promoções cruzadas entre streaming e TV Linear, destacando conteúdos complementares nas duas plataformas." 
                if max_corr_value > 0 else
                "Considerar estratégias específicas para cada plataforma, reconhecendo que atendem a momentos ou necessidades diferentes do consumidor."
            }
            """)

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

    # Add dropdown for metrics type
    metric_type = st.selectbox(
        "Selecione o tipo de métrica para visualização:",
        options=["Engajamento (Horas Consumidas)", "Audiência (Número de Usuários)", "Ativação (Horas Médias)"]
    )

    # Only include visualizations if we have the necessary data

    # Row 1 - Assinantes vs Logados vs Anônimos
    has_user_data = any(col in selected_df.columns for col in [
        'GP_horas_consumidas_assinantes', 
        'GP_horas_consumidas_de_logados_free', 
        'GP_horas_consumidas_de_anonimos',
        'GP_usuários_assinantes_',
        'GP_usuários_de_vídeo_logados_free',
        'GP_usuários_anonimos'
    ])

    if has_user_data:
        st.markdown("#### Tipos de Usuário vs. TV Linear")
        
        # Prepare data for the chart based on selected metric type
        plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
        
        if metric_type == "Engajamento (Horas Consumidas)":
            if 'GP_horas_consumidas_assinantes' in selected_df.columns:
                plot_data['Assinantes'] = selected_df['GP_horas_consumidas_assinantes']
            
            if 'GP_horas_consumidas_de_logados_free' in selected_df.columns:
                plot_data['Logados Free'] = selected_df['GP_horas_consumidas_de_logados_free']
            
            if 'GP_horas_consumidas_de_anonimos' in selected_df.columns:
                plot_data['Anônimos'] = selected_df['GP_horas_consumidas_de_anonimos']
            
            y_axis_title = "Horas Consumidas"
        
        elif metric_type == "Audiência (Número de Usuários)":
            if 'GP_usuários_assinantes_' in selected_df.columns:
                plot_data['Assinantes'] = selected_df['GP_usuários_assinantes_']
            
            if 'GP_usuários_de_vídeo_logados_free' in selected_df.columns:
                plot_data['Logados Free'] = selected_df['GP_usuários_de_vídeo_logados_free']
            
            if 'GP_usuários_anonimos' in selected_df.columns:
                plot_data['Anônimos'] = selected_df['GP_usuários_anonimos']
            
            y_axis_title = "Número de Usuários"
        
        elif metric_type == "Ativação (Horas Médias)":
            # Calculate average hours per user
            if has_assinantes:
                plot_data['Assinantes'] = selected_df['GP_horas_consumidas_assinantes'] / selected_df['GP_usuários_assinantes_']
            
            if has_logados:
                plot_data['Logados Free'] = selected_df['GP_horas_consumidas_de_logados_free'] / selected_df['GP_usuários_de_vídeo_logados_free']
            
            if has_anonimos:
                plot_data['Anônimos'] = selected_df['GP_horas_consumidas_de_anonimos'] / selected_df['GP_usuários_anonimos']
            
            y_axis_title = "Horas Médias por Usuário"
        
        # Add TV Linear data if available
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
            fig.update_yaxes(title_text=y_axis_title, secondary_y=False)
            fig.update_yaxes(title_text="Coverage %", secondary_y=True)
            fig.update_xaxes(title_text="Data")
            
            # Configurar layout
            fig.update_layout(
                title=f"{metric_type} por Tipo de Usuário - {granularity}",
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Interpretação:** Este gráfico mostra como o {metric_type.split('(')[0].strip().lower()} varia entre diferentes tipos de usuários ao longo do tempo.
            {
                "Observe que assinantes geralmente têm maior consumo total, refletindo seu maior engajamento com a plataforma." 
                if metric_type == "Engajamento (Horas Consumidas)" else
                "Observe a distribuição dos usuários entre os diferentes tipos de acesso, que revela a estrutura da base de usuários." 
                if metric_type == "Audiência (Número de Usuários)" else
                "Observe que assinantes geralmente têm maior consumo por pessoa, refletindo seu maior engajamento individual com a plataforma."
            }
            A linha de cobertura TV Linear permite avaliar se há
            complementaridade ou canibalização entre streaming e TV tradicional.
            """)
        
    # Row 2 - Mobile vs Outros Devices
    has_device_data = any(col in selected_df.columns for col in [
        'GP_horas_consumidas_mobile', 
        'GP_horas_consumidas_em_demais_devices',
        'GP_usuários_em_mobile',
        'GP_usuários_em_demais_devices'
    ])

    if has_device_data:
        st.markdown("#### Tipos de Device vs. TV Linear")
        
        # Prepare data for the chart based on selected metric type
        plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
        
        if metric_type == "Engajamento (Horas Consumidas)":
            if 'GP_horas_consumidas_mobile' in selected_df.columns:
                plot_data['Mobile'] = selected_df['GP_horas_consumidas_mobile']
            
            if 'GP_horas_consumidas_em_demais_devices' in selected_df.columns:
                plot_data['Outros Devices'] = selected_df['GP_horas_consumidas_em_demais_devices']
            
            y_axis_title = "Horas Consumidas"
        
        elif metric_type == "Audiência (Número de Usuários)":
            if 'GP_usuários_em_mobile' in selected_df.columns:
                plot_data['Mobile'] = selected_df['GP_usuários_em_mobile']
            
            if 'GP_usuários_em_demais_devices' in selected_df.columns:
                plot_data['Outros Devices'] = selected_df['GP_usuários_em_demais_devices']
            
            y_axis_title = "Número de Usuários"
        
        elif metric_type == "Ativação (Horas Médias)":
            # Calculate average hours per user
            if has_mobile:
                plot_data['Mobile'] = selected_df['GP_horas_consumidas_mobile'] / selected_df['GP_usuários_em_mobile']
            
            if has_outros:
                plot_data['Outros Devices'] = selected_df['GP_horas_consumidas_em_demais_devices'] / selected_df['GP_usuários_em_demais_devices']
            
            y_axis_title = "Horas Médias por Usuário"
        
        # Add TV Linear data if available
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
            fig.update_yaxes(title_text=y_axis_title, secondary_y=False)
            fig.update_yaxes(title_text="Coverage %", secondary_y=True)
            fig.update_xaxes(title_text="Data")
            
            # Configurar layout
            fig.update_layout(
                title=f"{metric_type} por Tipo de Device - {granularity}",
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Interpretação:** Este gráfico ilustra como o {metric_type.split('(')[0].strip().lower()} se distribui entre dispositivos móveis e outros tipos de tela.
            {
                "Tendências no consumo por tipo de dispositivo refletem mudanças importantes na forma como os usuários acessam o conteúdo." 
                if metric_type == "Engajamento (Horas Consumidas)" else
                "A distribuição de usuários entre dispositivos móveis e outros revela preferências de acesso à plataforma." 
                if metric_type == "Audiência (Número de Usuários)" else
                "O tempo médio por usuário em cada tipo de dispositivo indica diferenças na profundidade de engajamento em cada tela."
            }
            Observe também como eventos sazonais podem afetar diferentemente o comportamento em cada tipo de dispositivo.
            """)

    # Row 3 - Simulcasting vs VOD
    has_content_data = any(col in selected_df.columns for col in [
        'GP_horas_consumidas_em_tvg_ao_vivo', 
        'GP_qtd_de_horas_disponíveis_integras',
        'GP_usuários_em_tvg_ao_vivo',
        'GP_qtd_de_integras_publicadas'
    ])

    if has_content_data:
        st.markdown("#### Simulcasting vs VOD vs. TV Linear")
        
        # Prepare data for the chart based on selected metric type
        plot_data = pd.DataFrame({'Data': selected_df['data_hora']})
        
        if metric_type == "Engajamento (Horas Consumidas)":
            if 'GP_horas_consumidas_em_tvg_ao_vivo' in selected_df.columns:
                plot_data['TVG ao Vivo'] = selected_df['GP_horas_consumidas_em_tvg_ao_vivo']
            
            if 'GP_qtd_de_horas_disponíveis_integras' in selected_df.columns:
                plot_data['VOD (Íntegras)'] = selected_df['GP_qtd_de_horas_disponíveis_integras']
            
            y_axis_title = "Horas Consumidas/Disponíveis"
        
        elif metric_type == "Audiência (Número de Usuários)":
            if 'GP_usuários_em_tvg_ao_vivo' in selected_df.columns:
                plot_data['TVG ao Vivo'] = selected_df['GP_usuários_em_tvg_ao_vivo']
            
            if 'GP_qtd_de_integras_publicadas' in selected_df.columns:
                plot_data['VOD (Íntegras)'] = selected_df['GP_qtd_de_integras_publicadas']
            
            y_axis_title = "Número de Usuários/Itens"
        
        elif metric_type == "Ativação (Horas Médias)":
            # Calculate average hours per user/item
            if has_tvg:
                plot_data['TVG ao Vivo'] = selected_df['GP_horas_consumidas_em_tvg_ao_vivo'] / selected_df['GP_usuários_em_tvg_ao_vivo']
            
            if has_vod:
                plot_data['VOD (Íntegras)'] = selected_df['GP_qtd_de_horas_disponíveis_integras'] / selected_df['GP_qtd_de_integras_publicadas']
            
            y_axis_title = "Horas Médias por Usuário/Item"
        
        # Add TV Linear data if available
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
            fig.update_yaxes(title_text=y_axis_title, secondary_y=False)
            fig.update_yaxes(title_text="Coverage %", secondary_y=True)
            fig.update_xaxes(title_text="Data")
            
            # Configurar layout
            fig.update_layout(
                title=f"{metric_type} por Tipo de Conteúdo - {granularity}",
                font=dict(family="Roboto, Arial", color="#212121"),
                legend=dict(orientation="h", y=1.1),
                margin=dict(t=50, l=50, r=20, b=50),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Interpretação:** Este gráfico compara o {metric_type.split('(')[0].strip().lower()} de conteúdo ao vivo (simulcasting) versus conteúdo sob demanda (VOD).
            {
                "Observe como eventos especiais ou temporadas podem causar picos no consumo ao vivo, enquanto o consumo VOD tende a ser mais estável ao longo do tempo." 
                if metric_type == "Engajamento (Horas Consumidas)" else
                "A distribuição de usuários entre consumo ao vivo e sob demanda revela preferências importantes de comportamento de consumo." 
                if metric_type == "Audiência (Número de Usuários)" else
                "As horas médias por usuário/item revelam diferenças na intensidade de consumo entre conteúdo ao vivo e sob demanda."
            }
            A relação com a audiência de TV Linear fornece insights sobre a possível complementaridade ou competição entre estas modalidades de consumo.
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