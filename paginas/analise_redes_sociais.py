# utils/analise_redes_sociais.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px

def analise_redes_sociais(df):
    """
    Realiza análises estáticas de redes sociais utilizando o dataframe tratado,
    onde as colunas seguem o padrão 'RS_GLOBO_PLATAFORMA_metrica'.
    """
    st.markdown("""
    ## Análise de Redes Sociais
    
    Nesta análise, avaliamos o desempenho das diferentes plataformas sociais da Globo, comparando métricas 
    de engajamento como comentários, reações, compartilhamentos e visualizações. O objetivo é identificar:
    
    - Quais plataformas geram maior engajamento total
    - Qual é a eficiência de cada plataforma (engajamento por post)
    - Como está a distribuição de esforço (quantidade de posts) entre plataformas
    - Se o esforço está concentrado nas plataformas de melhor desempenho
    """)
    
    # Definição das plataformas e das métricas originais
    platforms = ["FACEBOOK", "TIKTOK", "INSTAGRAM", "YOUTUBE"]
    metrics = [
        "nr_comments", "nr_reactions", "nr_saves",
        "nr_shares", "nr_reach", "total_interactions", "nr_views", "nr_impressions"
    ]

    # Verificar as colunas existentes no dataframe
    prefixed_platforms = {}
    for platform in platforms:
        platform_cols = [col for col in df.columns if f"RS_GLOBO_{platform}" in col]
        if platform_cols:
            prefixed_platforms[platform] = f"RS_GLOBO_{platform}"
    
    # Calcular o engajamento total para cada plataforma (soma de todas as métricas de engajamento)
    for platform, prefix in prefixed_platforms.items():
        # Encontrar colunas para esta plataforma
        cols = []
        for m in metrics:
            col_name = f"{prefix}_{m}"
            if col_name in df.columns:
                cols.append(col_name)
        
        if cols:
            df[f"{platform}_engagement"] = df[cols].sum(axis=1)
        else:
            df[f"{platform}_engagement"] = 0

        # Calcular o engajamento médio por post, utilizando a coluna de quantidade de posts
        posts_col = f"{prefix}_posts_quantity"
        if posts_col in df.columns and df[posts_col].sum() != 0:
            df[f"{platform}_avg_engagement"] = df[f"{platform}_engagement"] / df[posts_col]
        else:
            df[f"{platform}_avg_engagement"] = 0

    # Determina a melhor plataforma em termos de engajamento médio (média ao longo do tempo)
    avg_engagements = {
        platform: df[f"{platform}_avg_engagement"].mean() 
        for platform in prefixed_platforms.keys() if f"{platform}_avg_engagement" in df.columns
    }
    best_platform = max(avg_engagements, key=avg_engagements.get) if avg_engagements else "N/D"

    # Determina a plataforma com maior quantidade de posts (soma total)
    posts = {}
    for platform, prefix in prefixed_platforms.items():
        posts_col = f"{prefix}_posts_quantity"
        if posts_col in df.columns:
            posts[platform] = df[posts_col].sum()
        else:
            posts[platform] = 0
            
    most_posts_platform = max(posts, key=posts.get) if posts else "N/D"

    # Texto dinâmico com as principais conclusões
    if best_platform == most_posts_platform:
        st.success(f"**Vocês têm mais engajamento em {best_platform} e investem mais em {most_posts_platform}.**")
        st.markdown(f"""
        ### 📊 Alinhamento de Esforço e Resultado
        
        A estratégia atual está bem alinhada! A plataforma **{best_platform}** apresenta o melhor engajamento médio e 
        é também onde vocês concentram a maior quantidade de posts. Isso indica uma alocação eficiente de recursos,
        priorizando o canal que traz os melhores resultados por post.
        """)
    else:
        st.error(f"**Vocês têm mais engajamento em {best_platform} e investem mais em {most_posts_platform}.**")
        st.markdown(f"""
        ### ⚠️ Desalinhamento de Esforço e Resultado
        
        Existe uma oportunidade de otimização na estratégia! Enquanto a plataforma **{best_platform}** apresenta o melhor 
        engajamento médio, a maior quantidade de posts está sendo direcionada para **{most_posts_platform}**. 
        
        **Recomendação:** Considere realocar parte do esforço para a plataforma de maior engajamento, 
        ou investigar por que o engajamento no {most_posts_platform} está abaixo do esperado, apesar do investimento.
        """)

    # --- Gráficos ---
    # Cria um dataframe derretido com as colunas que possuem as métricas originais
    cols_to_melt = []
    for col in df.columns:
        for m in metrics:
            if col.endswith(m) and col.startswith("RS_GLOBO_"):
                cols_to_melt.append(col)
                break

    # Verificar se há uma coluna de data no dataframe
    date_col = 'data_hora'
    if date_col not in df.columns:
        st.warning("Coluna 'data_hora' não encontrada no dataframe. Alguns gráficos temporais não poderão ser exibidos.")
    else:
        if not pd.api.types.is_datetime64_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col])
            
        df_melted = df.melt(
            id_vars=[date_col],
            value_vars=cols_to_melt,
            var_name="col",
            value_name="valor"
        )
        
        # Extrai a plataforma e a métrica a partir do nome da coluna
        df_melted["platform"] = df_melted["col"].apply(lambda x: x.split("_")[2] if len(x.split("_")) > 2 else "")
        df_melted["metrica"] = df_melted["col"].apply(lambda x: "_".join(x.split("_")[3:]) if len(x.split("_")) > 3 else "")

        # Filtra para manter somente as métricas originais
        df_plot1 = df_melted[df_melted["metrica"].isin(metrics)]

        st.markdown("""
        ### Análise Comparativa das Plataformas
        
        Os gráficos abaixo apresentam uma visão detalhada do desempenho de cada plataforma:
        
        1. **Engajamento total por plataforma:** Soma de todas as métricas de engajamento para cada plataforma
        2. **Engajamento médio por plataforma:** Engajamento por post, mostrando a eficiência de cada canal
        3. **Quantidade de posts por plataforma:** Distribuição do esforço entre os canais
        4. **Engajamento ao longo do tempo:** Tendências temporais no engajamento de cada plataforma
        """)
        
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # Plot 1: Total Engagement per Platform (soma de cada métrica ao longo de todas as datas)
        total_engagement = df_plot1.groupby(["platform", "metrica"])["valor"].sum().reset_index()
        sns.barplot(x="platform", y="valor", hue="metrica", data=total_engagement, ax=axes[0, 0])
        axes[0, 0].set_title("Engajamento total por plataforma")
        axes[0, 0].set_xlabel("Plataforma")
        axes[0, 0].set_ylabel("Total de Engajamento")

        # Plot 2: Average Engagement per Post per Platform
        available_platforms = list(prefixed_platforms.keys())
        avg_values = [df[f"{platform}_avg_engagement"].mean() for platform in available_platforms if f"{platform}_avg_engagement" in df.columns]
        
        if available_platforms and len(avg_values) > 0:
            sns.barplot(x=available_platforms, y=avg_values, ax=axes[0, 1])
            axes[0, 1].set_title("Engajamento médio por plataforma")
            axes[0, 1].set_xlabel("Plataforma")
            axes[0, 1].set_ylabel("Engajamento Médio por Post")
        else:
            axes[0, 1].text(0.5, 0.5, "Dados insuficientes", ha='center', va='center')
            axes[0, 1].set_title("Engajamento médio por plataforma")

        # Plot 3: Número de Posts por Plataforma
        posts_values = []
        for platform, prefix in prefixed_platforms.items():
            posts_col = f"{prefix}_posts_quantity"
            if posts_col in df.columns:
                posts_values.append(df[posts_col].sum())
            else:
                posts_values.append(0)
                
        if available_platforms and len(posts_values) > 0:
            sns.barplot(x=available_platforms, y=posts_values, ax=axes[1, 0])
            axes[1, 0].set_title("Qtd. de posts por plataforma")
            axes[1, 0].set_xlabel("Plataforma")
            axes[1, 0].set_ylabel("Quantidade de Posts")
        else:
            axes[1, 0].text(0.5, 0.5, "Dados insuficientes", ha='center', va='center')
            axes[1, 0].set_title("Qtd. de posts por plataforma")

        # Plot 4: Engagement Over Time para cada plataforma
        for platform in available_platforms:
            col_eng = f"{platform}_engagement"
            if col_eng in df.columns:
                sns.lineplot(x=date_col, y=col_eng, data=df, label=platform, ax=axes[1, 1])
        
        axes[1, 1].set_title("Engajamento x tempo")
        axes[1, 1].set_xlabel("Data")
        axes[1, 1].set_ylabel("Engajamento")

        plt.tight_layout(w_pad=10.0, h_pad=5.0)
        st.pyplot(fig)
        
        st.markdown("""
        ### Interpretação dos Resultados
        
        **Insights importantes:**
        
        1. **Tipos de engajamento** - Diferentes plataformas destacam-se em diferentes tipos de engajamento. Por exemplo, 
        o Instagram geralmente tem mais comentários, enquanto o YouTube tende a ter mais visualizações de vídeo.
        
        2. **Eficiência vs. Volume** - Uma plataforma pode ter alto engajamento total devido ao grande volume de posts, 
        mas baixa eficiência por post. O ideal é focar nas plataformas com maior engajamento médio.
        
        3. **Tendências temporais** - O gráfico de engajamento ao longo do tempo pode revelar sazonalidades ou 
        crescimento/declínio do desempenho, o que ajuda a entender a evolução da audiência digital.
        
        **Como usar esta análise:** Os resultados podem orientar decisões sobre onde concentrar esforços de produção de conteúdo,
        quais plataformas priorizar para determinados tipos de conteúdo, e como balancear a distribuição de recursos entre canais.
        """)

    # Novo Dashboard de Redes Sociais
    st.title("Redes Sociais - Desempenho Digital")
    
    # Última atualização
    if 'data_hora' in df.columns:
        last_date = df['data_hora'].max()
        if isinstance(last_date, pd.Timestamp):
            last_date = last_date.to_pydatetime()
            st.caption(f"Última atualização: {last_date.strftime('%d/%m/%Y')}")
    
    # Seção de filtros
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Filtro de granularidade
        granularity = st.selectbox(
            "Selecione a granularidade:",
            options=["Horário", "Diário", "Semanal"],
            index=1  # Padrão: Diário
        )

    with col2:
        # Filtro de conta
        # Detectar as contas disponíveis no dataframe
        canais_cols = [col for col in df.columns if col.startswith('RS_CANAIS_')]
        contas = []
        
        for col in canais_cols:
            parts = col.split('_')
            if len(parts) > 3:  # RS_CANAIS_PLATAFORMA_CONTA_METRICA
                conta = parts[3]
                if conta not in contas:
                    contas.append(conta)
        
        # Adicionar 'GLOBO' como opção padrão
        if not contas:
            contas = ['GLOBO']
        elif 'GLOBO' not in contas:
            contas.insert(0, 'GLOBO')
        
        selected_account = st.selectbox(
            "Selecione a conta:",
            options=contas,
            index=0  # Padrão: GLOBO
        )
    
    # Preparar dataframe de acordo com a granularidade selecionada
    df_analysis = df.copy()
    
    # Garantir que a coluna de data está no formato datetime
    if 'data_hora' in df_analysis.columns and not pd.api.types.is_datetime64_dtype(df_analysis['data_hora']):
        df_analysis['data_hora'] = pd.to_datetime(df_analysis['data_hora'])
    
    # Definir coluna de data para agregação
    date_col = 'data_hora'
    
    # Agregar dados de acordo com a granularidade
    if granularity == "Semanal":
        df_analysis['period'] = df_analysis[date_col].dt.isocalendar().week
        period_name = "Semana"
    elif granularity == "Diário":
        df_analysis['period'] = df_analysis[date_col].dt.date
        period_name = "Dia"
    else:  # Horário
        df_analysis['period'] = df_analysis[date_col]
        period_name = "Hora"
    
    # Filtrar colunas com base na conta selecionada
    if selected_account == 'GLOBO':
        # Usar colunas de GLOBO (RS_GLOBO_*)
        account_cols = [col for col in df_analysis.columns if col.startswith('RS_GLOBO_')]
    else:
        # Usar colunas específicas da conta (RS_CANAIS_*_CONTA_*)
        account_cols = [col for col in df_analysis.columns if col.startswith(f'RS_CANAIS_') and f'_{selected_account}_' in col]
    
    # Adicionar coluna de período
    account_cols.append('period')
    
    # Adicionar colunas de TV Linear para correlação
    tv_linear_cols = [col for col in df_analysis.columns if col.startswith('LINEAR_GLOBO_')]
    analysis_cols = account_cols + tv_linear_cols
    
    # Filtrar o dataframe apenas com as colunas necessárias
    columns_exist = [col for col in analysis_cols if col in df_analysis.columns]
    
    if columns_exist:
        df_filtered = df_analysis[columns_exist].copy()
    
        # Agregar dados por período
        df_agg = df_filtered.groupby('period').agg('mean').reset_index()
        
        # Seção 2: Métricas Globais por Rede Social
        st.header("Métricas Globais por Rede Social")
        
        # Extrair métricas importantes do dataframe
        metrics_cards = st.columns(5)
        
        # Detectar plataformas disponíveis para a conta selecionada
        if selected_account == 'GLOBO':
            available_platforms = [platform for platform, prefix in prefixed_platforms.items()]
            prefix = 'RS_GLOBO_'
        else:
            # Detectar plataformas disponíveis nos dados de canais
            available_platforms = set()
            for col in account_cols:
                if col.startswith('RS_CANAIS_'):
                    parts = col.split('_')
                    if len(parts) > 3:
                        platform = parts[2]  # RS_CANAIS_PLATAFORMA_CONTA_METRICA
                        available_platforms.add(platform)
            available_platforms = list(available_platforms)
            prefix = 'RS_CANAIS_'
        
        # Métricas para os cards
        with metrics_cards[0]:
            num_accounts = len(contas) if contas and contas[0] != 'GLOBO' else 1
            st.metric("Contas Monitoradas", f"{num_accounts}")
        
        with metrics_cards[1]:
            # Média de posts por semana
            posts_cols = [col for col in df_filtered.columns if 'posts_quantity' in col]
            if posts_cols:
                avg_posts_per_week = df_filtered[posts_cols].mean().sum() * 7  # Multiplicar por 7 para obter semanal
                st.metric("Posts por Semana", f"{avg_posts_per_week:.0f}")
            else:
                st.metric("Posts por Semana", "N/D")
        
        with metrics_cards[2]:
            # Impressões totais
            impressions_cols = [col for col in df_filtered.columns if 'impressions' in col or 'impressoes' in col]
            if impressions_cols:
                total_impressions = df_filtered[impressions_cols].sum().sum()
                st.metric("Impressões Totais", f"{total_impressions:,.0f}".replace(',', '.'))
            else:
                st.metric("Impressões Totais", "N/D")
        
        with metrics_cards[3]:
            # Interações totais
            interactions_cols = [col for col in df_filtered.columns if 'interactions' in col or 'interacoes' in col]
            if interactions_cols:
                total_interactions = df_filtered[interactions_cols].sum().sum()
                st.metric("Interações Totais", f"{total_interactions:,.0f}".replace(',', '.'))
            else:
                st.metric("Interações Totais", "N/D")
        
        with metrics_cards[4]:
            # CTR médio
            if impressions_cols and interactions_cols:
                total_imp = df_filtered[impressions_cols].sum().sum()
                total_int = df_filtered[interactions_cols].sum().sum()
                if total_imp > 0:
                    ctr = (total_int / total_imp) * 100
                    st.metric("CTR Médio", f"{ctr:.2f}%")
                else:
                    st.metric("CTR Médio", "N/D")
            else:
                st.metric("CTR Médio", "N/D")
        
        # Seleção de métrica para análise
        metric_types = {
            "Alcance": ["reach", "alcance", "impressions", "impressoes", "views", "videoviews"],
            "Engajamento": ["comments", "comentarios", "reactions", "reacoes", "shares", "saves", "total_interactions", "interacoes"],
            "Ativação": ["posts_quantity"]
        }
        
        selected_metric_type = st.selectbox(
            "Selecione o tipo de métrica para análise:",
            options=list(metric_types.keys())
        )
        
        # Filtrar colunas com base no tipo de métrica selecionado
        metric_keywords = metric_types[selected_metric_type]
        
        # Tabs para os três níveis de análise
        tabs = st.tabs(["Rede Social x TV Linear", "Verticais x TV Linear", "Conta x TV Linear"])
        
        # Nível 1: Rede Social x TV Linear
        with tabs[0]:
            st.subheader("Rede Social x TV Linear")
            
            # Filtrar colunas para o tipo de métrica selecionado
            if selected_account == 'GLOBO':
                # Para GLOBO, usamos o padrão RS_GLOBO_PLATAFORMA_metrica
                platform_metrics = {}
                for platform in available_platforms:
                    platform_cols = []
                    for keyword in metric_keywords:
                        cols = [col for col in df_filtered.columns if f'RS_GLOBO_{platform}' in col and keyword in col.lower()]
                        platform_cols.extend(cols)
                    
                    if platform_cols:
                        platform_metrics[platform] = df_filtered[platform_cols].sum(axis=1).mean()
            else:
                # Para contas específicas, usamos o padrão RS_CANAIS_PLATAFORMA_CONTA_metrica
                platform_metrics = {}
                for platform in available_platforms:
                    platform_cols = []
                    for keyword in metric_keywords:
                        cols = [col for col in df_filtered.columns if f'RS_CANAIS_{platform}_{selected_account}' in col and keyword in col.lower()]
                        platform_cols.extend(cols)
                    
                    if platform_cols:
                        platform_metrics[platform] = df_filtered[platform_cols].sum(axis=1).mean()
            
            # Criar dataframe para a tabela
            platform_data = []
            
            # Coluna de TV Linear para correlação (usar cov%)
            tv_col = 'LINEAR_GLOBO_cov%' if 'LINEAR_GLOBO_cov%' in df_filtered.columns else None
            
            for platform, avg_value in platform_metrics.items():
                # Extrair todas as colunas para esta plataforma e tipo de métrica
                platform_cols = []
                for keyword in metric_keywords:
                    if selected_account == 'GLOBO':
                        cols = [col for col in df_filtered.columns if f'RS_GLOBO_{platform}' in col and keyword in col.lower()]
                    else:
                        cols = [col for col in df_filtered.columns if f'RS_CANAIS_{platform}_{selected_account}' in col and keyword in col.lower()]
                    platform_cols.extend(cols)
                
                # Calcular estatísticas
                if platform_cols:
                    platform_values = df_filtered[platform_cols].sum(axis=1)
                    std_value = platform_values.std()
                    
                    # Calcular correlação com TV Linear se disponível
                    corr_value = None
                    if tv_col and tv_col in df_filtered.columns:
                        corr_value = platform_values.corr(df_filtered[tv_col])
                    
                    platform_data.append({
                        "Rede Social": platform,
                        "Valor Médio": avg_value,
                        "Desvio Padrão": std_value,
                        "Correlação com TV Linear": corr_value if corr_value is not None else None
                    })
            
            # Criar dataframe e exibir a tabela
            if platform_data:
                df_platform = pd.DataFrame(platform_data)
                
                # Formatar valores para exibição
                df_platform["Valor Médio"] = df_platform["Valor Médio"].map(lambda x: f"{x:,.2f}".replace(',', '.'))
                df_platform["Desvio Padrão"] = df_platform["Desvio Padrão"].map(lambda x: f"{x:,.2f}".replace(',', '.'))
                df_platform["Correlação com TV Linear"] = df_platform["Correlação com TV Linear"].map(lambda x: f"{x:.2f}" if x is not None else "N/D")
                
                st.dataframe(df_platform, use_container_width=True)
                
                # Gráfico de evolução
                st.subheader(f"Evolução de {selected_metric_type} por Rede Social vs. TV Linear")
                
                # Preparar dados para o gráfico
                # Agrupar por período para cada plataforma
                plot_data = []
                date_col_plot = 'period'
                
                for platform, _ in platform_metrics.items():
                    platform_cols = []
                    for keyword in metric_keywords:
                        if selected_account == 'GLOBO':
                            cols = [col for col in df_filtered.columns if f'RS_GLOBO_{platform}' in col and keyword in col.lower()]
                        else:
                            cols = [col for col in df_filtered.columns if f'RS_CANAIS_{platform}_{selected_account}' in col and keyword in col.lower()]
                        platform_cols.extend(cols)
                    
                    if platform_cols:
                        # Agrupar por período
                        platform_data = df_filtered.groupby(date_col_plot)[platform_cols].sum().sum(axis=1).reset_index()
                        platform_data.columns = [date_col_plot, 'value']
                        platform_data['platform'] = platform
                        plot_data.append(platform_data)
                
                # Adicionar dados da TV Linear
                if tv_col:
                    tv_data = df_filtered.groupby(date_col_plot)[tv_col].mean().reset_index()
                    tv_data.columns = [date_col_plot, 'value']
                    tv_data['platform'] = 'TV Linear (cov%)'
                    plot_data.append(tv_data)
                
                # Combinar os dados
                if plot_data:
                    df_plot = pd.concat(plot_data)
                    
                    # Criar gráfico com Plotly
                    fig = px.line(
                        df_plot, 
                        x=date_col_plot, 
                        y='value', 
                        color='platform',
                        title=f"Evolução de {selected_metric_type} por Rede Social vs. TV Linear ({granularity})",
                        labels={
                            date_col_plot: period_name,
                            'value': f'Valor de {selected_metric_type}',
                            'platform': 'Plataforma'
                        }
                    )
                    
                    # Melhorar layout
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                        hovermode="x unified"
                    )
                    
                    # Mostrar o gráfico
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dados insuficientes para gerar o gráfico de evolução.")
            else:
                st.info("Dados insuficientes para análise de Rede Social x TV Linear.")
        
        # Nível 2: Verticais x TV Linear
        with tabs[1]:
            st.subheader("Verticais x TV Linear")
            
            # Identificar verticais disponíveis
            verticais = ['Entretenimento', 'Esportes', 'Notícias', 'Variedades']
            
            # Mapear verticais para contas
            vertical_mapping = {
                'Entretenimento': ['BBB', 'Estrela da Casa', 'Multishow', 'GNT', 'VIVA', 'Gshow'],
                'Esportes': ['Sportv', 'ge', 'Cartola'],
                'Notícias': ['g1', 'GloboNews'],
                'Variedades': ['Receitas', 'Mundo Gloob', 'Globoplay']
            }
            
            # Verificar quais verticais possuem dados disponíveis
            available_verticais = []
            for vertical, contas in vertical_mapping.items():
                has_data = False
                for conta in contas:
                    conta_cols = [col for col in df_filtered.columns if f'_{conta}_' in col]
                    if conta_cols:
                        has_data = True
                        break
                
                if has_data:
                    available_verticais.append(vertical)
            
            if not available_verticais and selected_account == 'GLOBO':
                # Para GLOBO, considerar todas as verticais disponíveis
                available_verticais = verticais
            
            # Criar dataframe para a tabela
            vertical_data = []
            
            # Colunas de TV Linear para correlação
            tv_col = 'LINEAR_GLOBO_cov%' if 'LINEAR_GLOBO_cov%' in df_filtered.columns else None
            
            for vertical in available_verticais:
                # Extrair todas as contas para esta vertical
                vertical_accounts = vertical_mapping.get(vertical, [])
                
                # Colunas para esta vertical
                vertical_cols = []
                for account in vertical_accounts:
                    for keyword in metric_keywords:
                        # Verificar se estamos analisando GLOBO ou uma conta específica
                        if selected_account == 'GLOBO':
                            # Procurar colunas RS_CANAIS para todas as contas da vertical
                            cols = [col for col in df_filtered.columns if f'RS_CANAIS_' in col and f'_{account}_' in col and keyword in col.lower()]
                        else:
                            # Procurar apenas colunas da conta selecionada
                            if account == selected_account:
                                cols = [col for col in df_filtered.columns if f'RS_CANAIS_' in col and f'_{account}_' in col and keyword in col.lower()]
                            else:
                                cols = []
                        
                        vertical_cols.extend(cols)
                
                # Calcular estatísticas se temos colunas disponíveis
                if vertical_cols:
                    vertical_values = df_filtered[vertical_cols].sum(axis=1)
                    avg_value = vertical_values.mean()
                    std_value = vertical_values.std()
                    
                    # Calcular correlação com TV Linear
                    corr_value = None
                    if tv_col and tv_col in df_filtered.columns:
                        corr_value = vertical_values.corr(df_filtered[tv_col])
                    
                    vertical_data.append({
                        "Vertical": vertical,
                        "Valor Médio": avg_value,
                        "Desvio Padrão": std_value,
                        "Correlação com TV Linear": corr_value if corr_value is not None else None
                    })
            
            # Criar dataframe e exibir a tabela
            if vertical_data:
                df_vertical = pd.DataFrame(vertical_data)
                
                # Formatar valores para exibição
                df_vertical["Valor Médio"] = df_vertical["Valor Médio"].map(lambda x: f"{x:,.2f}".replace(',', '.'))
                df_vertical["Desvio Padrão"] = df_vertical["Desvio Padrão"].map(lambda x: f"{x:,.2f}".replace(',', '.'))
                df_vertical["Correlação com TV Linear"] = df_vertical["Correlação com TV Linear"].map(lambda x: f"{x:.2f}" if x is not None else "N/D")
                
                st.dataframe(df_vertical, use_container_width=True)
                
                # Gráfico de evolução
                st.subheader(f"Evolução de {selected_metric_type} por Vertical vs. TV Linear")
                
                # Preparar dados para o gráfico
                plot_data = []
                date_col_plot = 'period'
                
                for vertical in available_verticais:
                    # Extrair contas para esta vertical
                    vertical_accounts = vertical_mapping.get(vertical, [])
                    
                    # Colunas para esta vertical
                    vertical_cols = []
                    for account in vertical_accounts:
                        for keyword in metric_keywords:
                            if selected_account == 'GLOBO':
                                cols = [col for col in df_filtered.columns if f'RS_CANAIS_' in col and f'_{account}_' in col and keyword in col.lower()]
                            else:
                                if account == selected_account:
                                    cols = [col for col in df_filtered.columns if f'RS_CANAIS_' in col and f'_{account}_' in col and keyword in col.lower()]
                                else:
                                    cols = []
                            
                            vertical_cols.extend(cols)
                    
                    if vertical_cols:
                        # Agrupar por período
                        vertical_period_data = df_filtered.groupby(date_col_plot)[vertical_cols].sum().sum(axis=1).reset_index()
                        vertical_period_data.columns = [date_col_plot, 'value']
                        vertical_period_data['vertical'] = vertical
                        plot_data.append(vertical_period_data)
                
                # Adicionar dados da TV Linear
                if tv_col:
                    tv_data = df_filtered.groupby(date_col_plot)[tv_col].mean().reset_index()
                    tv_data.columns = [date_col_plot, 'value']
                    tv_data['vertical'] = 'TV Linear (cov%)'
                    plot_data.append(tv_data)
                
                # Combinar os dados
                if plot_data:
                    df_plot = pd.concat(plot_data)
                    
                    # Criar gráfico com Plotly
                    fig = px.line(
                        df_plot, 
                        x=date_col_plot, 
                        y='value', 
                        color='vertical',
                        title=f"Evolução de {selected_metric_type} por Vertical vs. TV Linear ({granularity})",
                        labels={
                            date_col_plot: period_name,
                            'value': f'Valor de {selected_metric_type}',
                            'vertical': 'Vertical'
                        }
                    )
                    
                    # Melhorar layout
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                        hovermode="x unified"
                    )
                    
                    # Mostrar o gráfico
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dados insuficientes para gerar o gráfico de evolução.")
            else:
                st.info("Dados insuficientes para análise de Verticais x TV Linear.")
        
        # Nível 3: Conta x TV Linear
        with tabs[2]:
            st.subheader("Conta x TV Linear")
            
            # Identificar contas disponíveis
            available_accounts = []
            
            # Se estamos analisando GLOBO, mostrar todas as contas disponíveis
            if selected_account == 'GLOBO':
                for col in df_filtered.columns:
                    if col.startswith('RS_CANAIS_'):
                        parts = col.split('_')
                        if len(parts) > 3:
                            account = parts[3]  # RS_CANAIS_PLATAFORMA_CONTA_METRICA
                            if account not in available_accounts:
                                available_accounts.append(account)
            else:
                # Se estamos analisando uma conta específica, mostrar apenas ela
                available_accounts = [selected_account]
            
            # Criar dataframe para a tabela
            account_data = []
            
            # Colunas de TV Linear para correlação
            tv_col = 'LINEAR_GLOBO_cov%' if 'LINEAR_GLOBO_cov%' in df_filtered.columns else None
            
            for account in available_accounts:
                # Encontrar a vertical da conta
                account_vertical = "Outro"
                for vertical, accounts in vertical_mapping.items():
                    if account in accounts:
                        account_vertical = vertical
                        break
                
                # Encontrar plataformas para esta conta
                account_platforms = []
                for col in df_filtered.columns:
                    if f'RS_CANAIS_' in col and f'_{account}_' in col:
                        parts = col.split('_')
                        if len(parts) > 3:
                            platform = parts[2]  # RS_CANAIS_PLATAFORMA_CONTA_METRICA
                            if platform not in account_platforms:
                                account_platforms.append(platform)
                
                # Para cada plataforma desta conta
                for platform in account_platforms:
                    # Colunas para esta plataforma e conta
                    account_cols = []
                    for keyword in metric_keywords:
                        cols = [col for col in df_filtered.columns if f'RS_CANAIS_{platform}_{account}' in col and keyword in col.lower()]
                        account_cols.extend(cols)
                    
                    # Calcular estatísticas
                    if account_cols:
                        account_values = df_filtered[account_cols].sum(axis=1)
                        avg_value = account_values.mean()
                        std_value = account_values.std()
                        
                        # Calcular correlação com TV Linear
                        corr_value = None
                        if tv_col and tv_col in df_filtered.columns:
                            corr_value = account_values.corr(df_filtered[tv_col])
                        
                        account_data.append({
                            "Rede Social": platform,
                            "Vertical": account_vertical,
                            "Conta": account,
                            "Valor Médio": avg_value,
                            "Desvio Padrão": std_value,
                            "Correlação com TV Linear": corr_value if corr_value is not None else None
                        })
            
            # Criar dataframe e exibir a tabela
            if account_data:
                df_account = pd.DataFrame(account_data)
                
                # Formatar valores para exibição
                df_account["Valor Médio"] = df_account["Valor Médio"].map(lambda x: f"{x:,.2f}".replace(',', '.'))
                df_account["Desvio Padrão"] = df_account["Desvio Padrão"].map(lambda x: f"{x:,.2f}".replace(',', '.'))
                df_account["Correlação com TV Linear"] = df_account["Correlação com TV Linear"].map(lambda x: f"{x:.2f}" if x is not None else "N/D")
                
                st.dataframe(df_account, use_container_width=True)
                
                # Gráfico de evolução
                st.subheader(f"Evolução de {selected_metric_type} por Conta vs. TV Linear")
                
                # Permitir seleção de contas para o gráfico se houver muitas
                if len(available_accounts) > 5:
                    selected_accounts_for_plot = st.multiselect(
                        "Selecione as contas para visualizar:",
                        options=available_accounts,
                        default=available_accounts[:5]  # Mostrar as 5 primeiras por padrão
                    )
                else:
                    selected_accounts_for_plot = available_accounts
                
                # Preparar dados para o gráfico
                plot_data = []
                date_col_plot = 'period'
                
                for account in selected_accounts_for_plot:
                    # Colunas para esta conta
                    account_cols = []
                    for keyword in metric_keywords:
                        cols = [col for col in df_filtered.columns if f'_{account}_' in col and keyword in col.lower()]
                        account_cols.extend(cols)
                    
                    if account_cols:
                        # Agrupar por período
                        account_period_data = df_filtered.groupby(date_col_plot)[account_cols].sum().sum(axis=1).reset_index()
                        account_period_data.columns = [date_col_plot, 'value']
                        account_period_data['account'] = account
                        plot_data.append(account_period_data)
                
                # Adicionar dados da TV Linear
                if tv_col:
                    tv_data = df_filtered.groupby(date_col_plot)[tv_col].mean().reset_index()
                    tv_data.columns = [date_col_plot, 'value']
                    tv_data['account'] = 'TV Linear (cov%)'
                    plot_data.append(tv_data)
                
                # Combinar os dados
                if plot_data:
                    df_plot = pd.concat(plot_data)
                    
                    # Criar gráfico com Plotly
                    fig = px.line(
                        df_plot, 
                        x=date_col_plot, 
                        y='value', 
                        color='account',
                        title=f"Evolução de {selected_metric_type} por Conta vs. TV Linear ({granularity})",
                        labels={
                            date_col_plot: period_name,
                            'value': f'Valor de {selected_metric_type}',
                            'account': 'Conta'
                        }
                    )
                    
                    # Melhorar layout
                    fig.update_layout(
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        margin=dict(l=40, r=40, t=40, b=40),
                        hovermode="x unified"
                    )
                    
                    # Mostrar o gráfico
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Dados insuficientes para gerar o gráfico de evolução.")
            else:
                st.info("Dados insuficientes para análise de Conta x TV Linear.")
    else:
        st.warning("Dados insuficientes para análise. Certifique-se de que o dataframe contém as colunas necessárias.")