# analises_estaticas.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

@st.cache_data
def analise_redes_sociais(df):
    """
    Realiza análises estáticas de redes sociais utilizando o dataframe tratado,
    onde as colunas seguem o padrão 'PLATAFORMA_metrica'.
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

    # Calcula o engajamento total para cada plataforma (soma de todas as métricas de engajamento)
    for platform in platforms:
        cols = [f"{platform}_{m}" for m in metrics if f"{platform}_{m}" in df.columns]
        if cols:
            df[f"{platform}_engagement"] = df[cols].sum(axis=1)
        else:
            df[f"{platform}_engagement"] = 0

        # Calcula o engajamento médio por post, utilizando a coluna de quantidade de posts
        posts_col = f"{platform}_posts_quantity"
        if posts_col in df.columns and df[posts_col].sum() != 0:
            df[f"{platform}_avg_engagement"] = df[f"{platform}_engagement"] / df[posts_col]
        else:
            df[f"{platform}_avg_engagement"] = 0

    # Determina a melhor plataforma em termos de engajamento médio (média ao longo do tempo)
    avg_engagements = {
        platform: df[f"{platform}_avg_engagement"].mean() 
        for platform in platforms if f"{platform}_avg_engagement" in df.columns
    }
    best_platform = max(avg_engagements, key=avg_engagements.get) if avg_engagements else "N/D"

    # Determina a plataforma com maior quantidade de posts (soma total)
    posts = {
        platform: df[f"{platform}_posts_quantity"].sum() 
        for platform in platforms if f"{platform}_posts_quantity" in df.columns
    }
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
            if col.endswith(m):
                cols_to_melt.append(col)
                break

    df_melted = df.melt(
        id_vars=["ts_published_brt"],
        value_vars=cols_to_melt,
        var_name="col",
        value_name="valor"
    )
    # Extrai a plataforma e a métrica a partir do nome da coluna
    df_melted["platform"] = df_melted["col"].apply(lambda x: x.split("_")[0])
    df_melted["metrica"] = df_melted["col"].apply(lambda x: "_".join(x.split("_")[1:]))

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
    avg_values = [df[f"{platform}_avg_engagement"].mean() for platform in platforms]
    sns.barplot(x=platforms, y=avg_values, ax=axes[0, 1])
    axes[0, 1].set_title("Engajamento médio por plataforma")
    axes[0, 1].set_xlabel("Plataforma")
    axes[0, 1].set_ylabel("Engajamento Médio por Post")

    # Plot 3: Número de Posts por Plataforma
    posts_values = [df[f"{platform}_posts_quantity"].sum() for platform in platforms]
    sns.barplot(x=platforms, y=posts_values, ax=axes[1, 0])
    axes[1, 0].set_title("Qtd. de posts por plataforma")
    axes[1, 0].set_xlabel("Plataforma")
    axes[1, 0].set_ylabel("Quantidade de Posts")

    # Plot 4: Engagement Over Time para cada plataforma
    for platform in platforms:
        col_eng = f"{platform}_engagement"
        if col_eng in df.columns:
            sns.lineplot(x="ts_published_brt", y=col_eng, data=df, label=platform, ax=axes[1, 1])
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

@st.cache_data
def analise_streaming_vs_linear(df):
    """
    Análise para responder perguntas sobre como o streaming está impactando a TV linear.
    """
    st.header("O streaming está reduzindo a relevância da TV linear?")
    
    st.markdown("""
    Esta análise investiga a relação entre o consumo de streaming (Globoplay) e a audiência tradicional de TV Linear,
    buscando entender se há um padrão de substituição (quando um canal reduz o outro) ou complementaridade
    (quando ambos os canais se reforçam mutuamente).
    
    Três questões principais serão analisadas:
    1. O impacto do Globoplay na audiência da TV Linear
    2. O perfil de consumo entre usuários de streaming vs. TV Linear
    3. Comportamentos de novas gerações em relação aos dois meios
    """)
    
    # Verificar se temos os dados necessários
    gp_cols = [col for col in df.columns if col.startswith('GP_')]
    linear_cols = [col for col in df.columns if col.startswith('LINEAR_')]
    
    if len(gp_cols) > 0 and len(linear_cols) > 0:
        # 1.1 O aumento de usuários e visualizações no Globoplay impacta negativamente o alcance da TV Linear?
        st.subheader("1.1 O aumento de usuários e visualizações no Globoplay impacta negativamente o alcance da TV Linear?")
        
        st.markdown("""
        Nesta primeira análise, investigamos se existe uma correlação inversa entre o consumo de Globoplay e a audiência de TV Linear.
        
        **Metodologia:**
        - Agrupamos os dados por dia para obter médias diárias
        - Calculamos a correlação entre usuários do Globoplay e o alcance da TV Linear
        - Analisamos a correlação entre horas consumidas no Globoplay e o rating de TV Linear
        - Observamos a tendência temporal para identificar padrões de longo prazo
        
        **Como interpretar:**
        - Correlação negativa forte (próxima de -1): indica substituição (mais streaming = menos TV)
        - Correlação positiva forte (próxima de +1): indica complementaridade (ambos crescem juntos)
        - Correlação próxima de zero: indica que os serviços atendem a necessidades diferentes
        """)
        
        # Agrupar por data para análise diária
        df_diario = df.copy()
        df_diario['data'] = pd.to_datetime(df_diario['data_hora']).dt.date
        
        # Selecionar apenas colunas numéricas para agregação
        numeric_cols = df_diario.select_dtypes(include=['number']).columns.tolist()
        # Garantir que 'data' não está nas colunas para agregação
        if 'data' in numeric_cols:
            numeric_cols.remove('data')
            
        # Agrupar apenas colunas numéricas
        df_diario = df_diario.groupby('data')[numeric_cols].mean().reset_index()
        
        # Selecionar métricas relevantes
        gp_users = 'GP_usuários_assinantes_'
        gp_hours = 'GP_horas_consumidas_assinantes'
        linear_reach = 'LINEAR_GLOBO_cov%'
        linear_rating = 'LINEAR_GLOBO_rat%'
        
        # Verificar se as métricas existem
        metrics_exist = all(metric in df_diario.columns for metric in [gp_users, gp_hours, linear_reach, linear_rating])
        
        if metrics_exist:
            # Calcular correlações
            correlation_users_reach = df_diario[gp_users].corr(df_diario[linear_reach])
            correlation_hours_rating = df_diario[gp_hours].corr(df_diario[linear_rating])
            
            # Criar visualização de correlação
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.scatter(df_diario, x=gp_users, y=linear_reach,
                                trendline="ols", 
                                title=f"Correlação: Usuários Globoplay vs Alcance TV Linear (r = {correlation_users_reach:.2f})")
                fig.update_layout(xaxis_title="Usuários Assinantes Globoplay", 
                                yaxis_title="% Cobertura TV Linear")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(df_diario, x=gp_hours, y=linear_rating,
                                trendline="ols", 
                                title=f"Correlação: Horas Consumidas vs Rating TV Linear (r = {correlation_hours_rating:.2f})")
                fig.update_layout(xaxis_title="Horas Consumidas Globoplay", 
                                yaxis_title="% Rating TV Linear")
                st.plotly_chart(fig, use_container_width=True)
            
            # Análise automática baseada nas correlações
            if correlation_users_reach < -0.3:
                st.error("**Há uma correlação negativa significativa entre usuários do Globoplay e o alcance da TV Linear, sugerindo substituição de mídia.**")
                st.markdown("""
                Os dados mostram uma correlação negativa significativa, indicando que quando mais pessoas usam o Globoplay, 
                menos pessoas assistem à TV Linear. Este padrão sugere um efeito de substituição, onde o streaming está 
                efetivamente substituindo o consumo tradicional de TV.
                
                **Implicações:** Este resultado indica que pode ser necessário desenvolver estratégias específicas para cada canal, 
                reconhecendo que eles estão competindo pelo tempo do espectador, em vez de se complementarem.
                """)
            elif correlation_users_reach > 0.3:
                st.success("**Há uma correlação positiva entre usuários do Globoplay e o alcance da TV Linear, sugerindo complementaridade.**")
                st.markdown("""
                Os dados mostram uma correlação positiva significativa, indicando que o crescimento de usuários do Globoplay 
                está associado a um aumento no alcance da TV Linear. Isso sugere que os serviços são complementares - provavelmente 
                porque o conteúdo em um meio estimula o interesse no outro.
                
                **Implicações:** Este resultado favorece estratégias de conteúdo integradas que criam sinergias entre plataformas,
                como usar o Globoplay para estender a experiência de programas da TV Linear ou vice-versa.
                """)
            else:
                st.info("**Não existe correlação forte entre usuários do Globoplay e o alcance da TV Linear, sugerindo que os serviços atendem a necessidades diferentes.**")
                st.markdown("""
                A correlação fraca indica que o uso do Globoplay e o consumo de TV Linear são relativamente independentes entre si.
                Isso sugere que os serviços atendem a diferentes necessidades dos consumidores ou a diferentes momentos de consumo.
                
                **Implicações:** Esta independência sugere que é possível desenvolver estratégias específicas para cada plataforma
                sem grandes preocupações com canibalização. Os conteúdos podem ser customizados para atender às expectativas
                específicas de cada meio.
                """)
            
            # Tendência ao longo do tempo
            fig = go.Figure()
            
            # Normalizar valores para comparação
            gp_users_norm = df_diario[gp_users] / df_diario[gp_users].max()
            linear_reach_norm = df_diario[linear_reach] / df_diario[linear_reach].max()
            
            fig.add_trace(go.Scatter(x=df_diario['data'], y=gp_users_norm, 
                                    name='Usuários Globoplay (Normalizado)',
                                    line=dict(color='#1f77b4')))
            
            fig.add_trace(go.Scatter(x=df_diario['data'], y=linear_reach_norm, 
                                    name='Alcance TV Linear (Normalizado)',
                                    line=dict(color='#ff7f0e')))
            
            fig.update_layout(title='Tendência: Usuários Globoplay vs Alcance TV Linear',
                            xaxis_title='Data',
                            yaxis_title='Valor Normalizado (0-1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretação da tendência temporal:**
            
            O gráfico acima mostra a evolução normalizada de usuários do Globoplay e do alcance da TV Linear ao longo do tempo.
            Padrões a observar:
            
            - **Movimentos na mesma direção:** Indicam complementaridade (ambos crescem ou diminuem juntos)
            - **Movimentos em direções opostas:** Indicam substituição (quando um cresce, o outro diminui)
            - **Mudanças sazonais:** Podem revelar períodos específicos onde a relação entre os meios se altera
            
            Esta visualização temporal complementa a análise de correlação, ajudando a entender como a relação
            entre streaming e TV linear evolui com o tempo.
            """)
        else:
            st.warning("Algumas métricas necessárias não estão disponíveis para análise completa.")
        
        # 1.2 O consumidor que usa streaming tende a não consumir TV Linear?
        st.subheader("1.2 O consumidor que usa streaming tende a não consumir TV Linear?")
        
        st.markdown("""
        Aqui investigamos o comportamento dos consumidores para identificar se há um perfil de "ou um ou outro"
        ou se os mesmos consumidores utilizam ambos os serviços em momentos diferentes.
        
        **Metodologia:**
        - Analisamos o consumo por período do dia (manhã, tarde, noite, madrugada)
        - Segmentamos o consumo por nível de uso do Globoplay
        - Observamos como o rating da TV Linear varia entre os diferentes níveis de uso de streaming
        
        **Como interpretar:**
        - Rating menor nos grupos de alto uso de streaming: sugere substituição
        - Rating maior nos grupos de alto uso de streaming: sugere complementaridade
        - Variações por período do dia: revelam diferentes comportamentos ao longo do dia
        """)
        
        # Para esta análise, idealmente precisaríamos de dados no nível de usuário
        # Como não temos esse nível de detalhe, vamos fazer uma análise de segmentos de tempo
        
        # Dividir o dia em períodos (manhã, tarde, noite)
        df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
        df['periodo'] = pd.cut(df['hora'], 
                              bins=[0, 6, 12, 18, 24], 
                              labels=['Madrugada', 'Manhã', 'Tarde', 'Noite'])
        
        # Selecionar apenas colunas numéricas para agregação
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Analisar consumo por período
        df_periodo = df.groupby('periodo')[numeric_cols].mean().reset_index()
        
        # Verificar se as colunas existem
        if 'GP_usuários_assinantes_' in df_periodo.columns and 'LINEAR_GLOBO_rat%' in df_periodo.columns:
            fig = px.bar(df_periodo, x='periodo', 
                        y=['GP_usuários_assinantes_', 'LINEAR_GLOBO_rat%'],
                        barmode='group',
                        title='Consumo de Streaming vs TV Linear por Período do Dia',
                        labels={'value': 'Valor Médio', 'periodo': 'Período do Dia', 
                               'variable': 'Métrica'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Análise por período do dia:**
            
            O gráfico acima mostra a distribuição do consumo de Globoplay e TV Linear em diferentes períodos do dia.
            
            Pontos importantes:
            - **Períodos de pico para cada meio:** Identificar quando cada plataforma apresenta maior consumo
            - **Padrões de uso complementar:** Verificar se há períodos onde ambos são consumidos intensamente
            - **Janelas de oportunidade:** Períodos de baixo consumo em ambas as plataformas representam oportunidades
            
            Esta análise temporal é crucial para entender como os hábitos de consumo se distribuem ao longo do dia,
            permitindo estratégias de programação e promoção mais eficazes.
            """)
        
        # Calcular tendência de uso complementar ou substituto
        if 'GP_usuários_assinantes_' in df.columns and 'LINEAR_GLOBO_rat%' in df.columns:
            # Criar bins baseados nos quartis de usuários Globoplay
            df_temp = df.copy()
            # Usar apenas valores não-nan para criar os bins
            valid_mask = ~df_temp['GP_usuários_assinantes_'].isna()
            if valid_mask.sum() > 0:
                df_temp.loc[valid_mask, 'gp_users_bin'] = pd.qcut(
                    df_temp.loc[valid_mask, 'GP_usuários_assinantes_'], 
                    4, 
                    labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto']
                )
                
                # Calcular rating médio de TV Linear para cada bin
                rating_by_gp_bin = df_temp.groupby('gp_users_bin')['LINEAR_GLOBO_rat%'].mean().reset_index()
                
                fig = px.bar(rating_by_gp_bin, x='gp_users_bin', y='LINEAR_GLOBO_rat%',
                            title='Rating TV Linear por Nível de Uso do Globoplay',
                            labels={'gp_users_bin': 'Nível de Uso do Globoplay', 
                                   'LINEAR_GLOBO_rat%': 'Rating Médio TV Linear (%)'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análise automática
                trend = rating_by_gp_bin['LINEAR_GLOBO_rat%'].values
                if len(trend) >= 3 and trend[0] > trend[-1]:
                    st.error("**Os dados sugerem que usuários mais ativos no Globoplay tendem a consumir menos TV Linear, indicando substituição.**")
                    st.markdown("""
                    **Análise de substituição detectada:**
                    
                    O gráfico revela um padrão claro: quanto maior o uso do Globoplay, menor o consumo de TV Linear.
                    Esta tendência decrescente confirma um efeito de substituição, onde o tempo gasto em streaming
                    reduz diretamente o tempo disponível para TV Linear.
                    
                    **Implicações estratégicas:**
                    - Considerar estratégias de diferenciação de conteúdo entre as plataformas
                    - Desenvolver planos de migração gradual de audiência para streaming mantendo receitas
                    - Avaliar modelos de negócio que capitalizem o comportamento de substituição
                    """)
                elif len(trend) >= 3 and trend[0] < trend[-1]:
                    st.success("**Os dados sugerem que usuários mais ativos no Globoplay também consomem mais TV Linear, indicando complementaridade.**")
                    st.markdown("""
                    **Análise de complementaridade detectada:**
                    
                    O gráfico mostra um padrão ascendente: quanto maior o uso do Globoplay, maior também o consumo de TV Linear.
                    Esta relação positiva confirma que os serviços são complementares, provavelmente reforçando o interesse mútuo.
                    
                    **Implicações estratégicas:**
                    - Investir em estratégias cross-media que criem jornadas entre as plataformas
                    - Desenvolver conteúdos que se complementem entre streaming e TV Linear
                    - Explorar promoções conjuntas que incentivem o uso de ambas as plataformas
                    """)
                else:
                    st.info("**Não há um padrão claro de substituição ou complementaridade entre o uso de Globoplay e TV Linear.**")
                    st.markdown("""
                    **Análise de independência:**
                    
                    O gráfico não apresenta uma tendência consistente, sugerindo que a relação entre uso de Globoplay e 
                    consumo de TV Linear é complexa e possivelmente influenciada por outros fatores como tipo de conteúdo,
                    horário, ou perfil demográfico.
                    
                    **Implicações estratégicas:**
                    - Investigar fatores contextuais ou segmentos específicos onde há complementaridade ou substituição
                    - Considerar estratégias diferenciadas por tipo de conteúdo ou momento de consumo
                    - Explorar modelos híbridos que maximizem o valor em ambas as plataformas
                    """)
        
        # 1.3 Análise de comportamento de novas gerações
        st.subheader("1.3 Análise de comportamento de novas gerações podem revelar padrões de substituição ou complementaridade")
        
        st.markdown("🚧 **WORK IN PROGRESS** 🚧")
        st.info("Esta análise requer dados demográficos que não estão disponíveis no conjunto de dados atual. Seria necessário ter informações sobre faixa etária dos usuários para avaliar comportamentos geracionais.")
        
        st.markdown("""
        **O que pretendemos analisar aqui:**
        
        Esta seção buscaria identificar diferenças nos padrões de consumo entre diferentes gerações, 
        especialmente focando nas gerações mais jovens (Z e Alpha) versus gerações mais velhas.
        
        **Metodologia que seria utilizada:**
        - Segmentação dos dados por faixa etária
        - Análise de preferências de dispositivo por geração
        - Padrões de consumo simultâneo ou exclusivo por faixa etária
        - Tendências temporais específicas de cada geração
        
        **Dados necessários para esta análise:**
        - Informações demográficas dos usuários (idade/geração)
        - Métricas de uso por dispositivo segmentadas por idade
        - Histórico de consumo longitudinal por faixa etária
        
        Esta análise será implementada quando os dados demográficos estiverem disponíveis.
        """)
        
    else:
        st.warning("Dados insuficientes para realizar a análise. É necessário ter métricas de Globoplay (GP_) e TV Linear (LINEAR_).")

@st.cache_data
def analise_social_impacto(df):
    """
    Análise para responder perguntas sobre como os sites e redes sociais impactam a TV linear.
    """
    st.header("Os sites e Social da Globo servem como chamariz da TV linear?")
    
    st.markdown("""
    Esta análise investiga como as plataformas digitais da Globo (redes sociais e sites) influenciam
    a audiência da TV Linear. O objetivo é entender se elas funcionam como complemento, atraindo
    audiência para a TV, ou se competem pela atenção do público.
    
    Exploramos três questões principais:
    1. O impacto das publicações nas redes sociais no alcance da TV Linear
    2. O uso dos sites como segunda tela durante o consumo de TV Linear
    3. O papel das diferentes plataformas sociais como complemento à experiência de TV
    """)
    
    # Verificar se temos os dados necessários
    rs_cols = [col for col in df.columns if col.startswith('RS_')]
    linear_cols = [col for col in df.columns if col.startswith('LINEAR_')]
    
    if len(rs_cols) > 0 and len(linear_cols) > 0:
        # 2.1 Mais publicações dos nossos conteúdos aumentam o nosso alcance?
        st.subheader("2.1 Mais publicações dos nossos conteúdos aumentam o nosso alcance (principalmente para novas gerações)?")
        
        st.markdown("""
        Esta análise examina a relação entre o volume de publicações nas redes sociais e o alcance da TV Linear,
        buscando entender se existe um efeito de "chamariz" onde posts nas redes sociais atraem audiência para a TV.
        
        **Metodologia:**
        - Analisamos a correlação entre quantidade de posts e cobertura de TV
        - Aplicamos um lag de 1 dia para avaliar o efeito posterior dos posts
        - Comparamos o impacto de diferentes plataformas sociais
        - Analisamos o efeito especificamente em horários com público mais jovem
        
        **Como interpretar:**
        - Correlação positiva forte: indica que posts sociais aumentam a cobertura TV
        - Diferenças entre plataformas: revelam quais redes têm maior impacto na TV
        - Comparações por horário: mostram se o efeito é maior para públicos jovens
        """)
        
        # Identificar colunas de quantidade de posts
        posts_cols = [col for col in rs_cols if 'posts_quantity' in col]
        
        if posts_cols and 'LINEAR_GLOBO_cov%' in df.columns:
            # Criar métrica agregada de posts
            df['total_posts'] = df[posts_cols].sum(axis=1)
            
            # Análise de lag para verificar efeito de posts na cobertura TV
            # Criar lag de 1 dia para posts
            df['data_apenas'] = pd.to_datetime(df['data_hora']).dt.date
            df_daily = df.groupby('data_apenas').agg({
                'total_posts': 'sum',
                'LINEAR_GLOBO_cov%': 'mean'
            }).reset_index()
            
            df_daily['posts_lag1'] = df_daily['total_posts'].shift(1)
            df_daily = df_daily.dropna()
            
            # Correlação entre posts e cobertura
            corr_posts_cov = df_daily['posts_lag1'].corr(df_daily['LINEAR_GLOBO_cov%'])
            
            fig = px.scatter(df_daily, x='posts_lag1', y='LINEAR_GLOBO_cov%',
                            trendline="ols", 
                            title=f"Efeito de Posts de Redes Sociais na Cobertura TV (r = {corr_posts_cov:.2f})")
            fig.update_layout(xaxis_title="Quantidade de Posts (Dia Anterior)", 
                            yaxis_title="% Cobertura TV Linear")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Interpretação da relação entre posts e cobertura TV:**
            
            O gráfico acima mostra a relação entre a quantidade de posts nas redes sociais em um dia e a 
            cobertura da TV Linear no dia seguinte, com correlação de **{corr_posts_cov:.2f}**.
            
            {
                "**A correlação positiva indica que posts nas redes sociais efetivamente funcionam como chamariz para a TV Linear, aumentando sua cobertura.**" 
                if corr_posts_cov > 0.3 else
                "**A correlação negativa sugere que maior atividade nas redes sociais pode estar desviando a atenção da TV Linear.**" 
                if corr_posts_cov < -0.3 else
                "**A correlação fraca sugere que o volume de posts nas redes sociais tem impacto limitado na cobertura da TV Linear.**"
            }
            
            Este resultado tem implicações importantes para a estratégia de distribuição de conteúdo e promoção cruzada entre plataformas.
            """)
            
            # Análise por plataforma
            st.subheader("Impacto dos posts por plataforma na cobertura de TV")
            
            platforms = ["FACEBOOK", "INSTAGRAM", "TIKTOK", "YOUTUBE"]
            corrs = []
            
            for platform in platforms:
                platform_posts = [col for col in posts_cols if platform in col]
                if platform_posts:
                    df[f'posts_{platform}'] = df[platform_posts].sum(axis=1)
                    df_plat = df.groupby('data_apenas').agg({
                        f'posts_{platform}': 'sum',
                        'LINEAR_GLOBO_cov%': 'mean'
                    }).reset_index()
                    
                    df_plat[f'posts_{platform}_lag1'] = df_plat[f'posts_{platform}'].shift(1)
                    df_plat = df_plat.dropna()
                    
                    if len(df_plat) > 5:  # Garantir dados suficientes
                        corr = df_plat[f'posts_{platform}_lag1'].corr(df_plat['LINEAR_GLOBO_cov%'])
                        corrs.append({'Plataforma': platform, 'Correlação': corr})
            
            if corrs:
                df_corrs = pd.DataFrame(corrs)
                fig = px.bar(df_corrs, x='Plataforma', y='Correlação',
                            title="Impacto de posts por plataforma na cobertura TV",
                            color='Correlação',
                            color_continuous_scale=['red', 'yellow', 'green'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Identificar plataforma com maior impacto
                best_platform = df_corrs.loc[df_corrs['Correlação'].idxmax()]
                if best_platform['Correlação'] > 0.2:
                    st.success(f"**{best_platform['Plataforma']} é a plataforma com maior impacto positivo na cobertura TV (r = {best_platform['Correlação']:.2f})**")
                    st.markdown(f"""
                    **Análise por plataforma:**
                    
                    Entre todas as plataformas sociais, **{best_platform['Plataforma']}** demonstra o maior impacto 
                    positivo na cobertura da TV Linear, com correlação de **{best_platform['Correlação']:.2f}**.
                    
                    **Implicações práticas:**
                    - Priorizar publicações no {best_platform['Plataforma']} para promover conteúdo da TV Linear
                    - Avaliar características específicas do conteúdo nesta plataforma que geram maior engajamento
                    - Considerar aumentar investimento em conteúdo para {best_platform['Plataforma']}
                    
                    Esta diferença de eficácia entre plataformas pode estar relacionada ao perfil demográfico dos usuários,
                    ao formato do conteúdo ou às características específicas de consumo em cada plataforma.
                    """)
                else:
                    st.info("**Nenhuma plataforma demonstra impacto significativo positivo na cobertura TV**")
                    st.markdown("""
                    **Análise por plataforma:**
                    
                    Nenhuma plataforma social demonstra um impacto significativamente positivo na cobertura da TV Linear.
                    Isso sugere que os posts em redes sociais atualmente não estão funcionando efetivamente como "chamariz"
                    para a TV.
                    
                    **Possíveis razões:**
                    - Desconexão entre o conteúdo publicado e a programação da TV
                    - Audiências distintas entre redes sociais e TV Linear
                    - Falta de chamadas para ação efetivas direcionando para a TV
                    
                    Recomenda-se revisar a estratégia de conteúdo social para criar pontes mais efetivas
                    com a programação de TV.
                    """)
            
            # Novas gerações - análise por horário (proxy)
            df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
            # Horários com maior presença de público jovem (19h-23h)
            df['horario_jovem'] = df['hora'].between(19, 23)
            
            df_horario = df.groupby('horario_jovem').agg({
                'total_posts': 'mean',
                'LINEAR_GLOBO_cov%': 'mean'
            }).reset_index()
            
            st.subheader("Impacto de posts em horários com público mais jovem")
            st.write(df_horario)
            
            if len(df_horario) > 1:
                impact_diff = ((df_horario.loc[df_horario['horario_jovem']==True, 'LINEAR_GLOBO_cov%'].values[0] /
                             df_horario.loc[df_horario['horario_jovem']==False, 'LINEAR_GLOBO_cov%'].values[0]) - 1) * 100
                
                if impact_diff > 5:
                    st.success(f"**Em horários com maior presença de público jovem, a cobertura TV é {impact_diff:.1f}% maior**")
                    st.markdown(f"""
                    **Análise para públicos jovens:**
                    
                    A cobertura da TV Linear durante horários com maior presença de público jovem (19h-23h) é 
                    **{impact_diff:.1f}% maior** comparada a outros horários. Isso sugere que o público jovem 
                    está efetivamente consumindo conteúdo de TV Linear, especialmente no horário nobre.
                    
                    **Implicações:**
                    - A percepção de que jovens não consomem TV Linear pode estar equivocada
                    - O horário nobre continua sendo relevante para novas gerações
                    - Estratégias específicas para este horário podem ter maior impacto no público jovem
                    """)
                elif impact_diff < -5:
                    st.error(f"**Em horários com maior presença de público jovem, a cobertura TV é {abs(impact_diff):.1f}% menor**")
                    st.markdown(f"""
                    **Análise para públicos jovens:**
                    
                    A cobertura da TV Linear durante horários com maior presença de público jovem (19h-23h) é 
                    **{abs(impact_diff):.1f}% menor** comparada a outros horários. Isso confirma a tendência de 
                    menor consumo de TV Linear entre as novas gerações.
                    
                    **Implicações:**
                    - Estratégias específicas para atrair público jovem para a TV Linear são necessárias
                    - Considerar formatos e conteúdos que ressoem mais com novas gerações
                    - Avaliar plataformas digitais como principal canal para alcançar este público
                    """)
                else:
                    st.info("**Não há diferença significativa na cobertura TV em horários com maior presença de público jovem**")
                    st.markdown("""
                    **Análise para públicos jovens:**
                    
                    Não há diferença significativa na cobertura da TV Linear entre horários com maior presença 
                    de público jovem (19h-23h) e outros horários. Isso sugere que o padrão de consumo pode estar 
                    mais relacionado ao tipo de conteúdo do que ao horário em si.
                    
                    **Implicações:**
                    - Focar na qualidade e relevância do conteúdo independentemente do horário
                    - Investigar outros fatores que possam influenciar o consumo entre novas gerações
                    - Considerar estratégias mais granulares baseadas em interesses específicos e não apenas horário
                    """)
        
        # 2.2 Os sites (G1, GE e GShow) são complementares, sendo usados como segunda tela da TV linear?
        st.subheader("2.2 Os sites (G1, GE e GShow) são complementares, sendo usados como segunda tela da TV linear?")
        
        st.markdown("""
        Esta análise investiga se os sites da Globo (G1, GE e GShow) são utilizados como "segunda tela"
        durante o consumo de TV Linear. Uma "segunda tela" refere-se ao uso de um dispositivo digital
        simultaneamente à TV, complementando a experiência.
        
        **Metodologia:**
        - Analisamos a correlação entre interações nos sites e rating TV por hora do dia
        - Focamos especialmente em horários de maior audiência TV (horário nobre)
        - Comparamos o comportamento entre diferentes sites
        
        **Como interpretar:**
        - Correlação alta em horário nobre: indica uso como segunda tela
        - Variações entre sites: revelam quais portais têm maior sinergia com a TV
        - Padrões distintos ao longo do dia: mostram momentos de uso complementar
        """)
        
        # Identificar colunas relacionadas aos sites
        site_cols = [col for col in rs_cols if any(site in col for site in ['g1', 'ge', 'Gshow'])]
        
        if site_cols and 'LINEAR_GLOBO_rat%' in df.columns:
            # Identificar métricas de interação nos sites
            site_interactions = [col for col in site_cols if 'interacoes' in col]
            
            # Análise por hora do dia para detectar uso de segunda tela
            df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
            
            # Para cada site, calcular correlação por hora
            sites = ['g1', 'ge', 'Gshow']
            hora_prime = list(range(19, 23))  # Horário nobre
            
            results = []
            for site in sites:
                site_cols_filtered = [col for col in site_interactions if site in col]
                if site_cols_filtered:
                    df[f'{site}_interactions'] = df[site_cols_filtered].sum(axis=1)
                    
                    for hora in range(24):
                        df_hora = df[df['hora'] == hora]
                        if len(df_hora) > 5:  # Garantir dados suficientes
                            corr = df_hora[f'{site}_interactions'].corr(df_hora['LINEAR_GLOBO_rat%'])
                            is_prime = hora in hora_prime
                            results.append({
                                'Site': site,
                                'Hora': hora,
                                'Correlação': corr,
                                'Prime': is_prime
                            })
            
            if results:
                df_results = pd.DataFrame(results)
                
                # Plot de correlação hora a hora por site
                fig = px.line(df_results, x='Hora', y='Correlação', color='Site',
                            title="Correlação entre interações nos sites e rating TV por hora",
                            labels={'Correlação': 'Correlação com Rating TV'},
                            line_shape='spline')
                
                # Adicionar faixa para horário nobre
                fig.add_vrect(x0=19, x1=23, 
                            fillcolor="LightSalmon", opacity=0.2,
                            layer="below", line_width=0,
                            annotation_text="Horário Nobre")
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretação do uso como segunda tela:**
                
                O gráfico acima mostra como a correlação entre interações nos sites e o rating da TV Linear
                varia ao longo do dia. As áreas destacadas correspondem ao horário nobre (19h-23h), quando
                a audiência da TV tende a ser maior.
                
                **O que procurar no gráfico:**
                - **Picos durante o horário nobre:** Indicam uso como segunda tela durante programas de alta audiência
                - **Diferenças entre sites:** Revelam qual portal tem maior sinergia com a programação da TV
                - **Variações ao longo do dia:** Mostram como o comportamento muda em diferentes momentos
                
                Correlações positivas fortes durante o horário nobre sugerem que os usuários estão consumindo
                conteúdo no site enquanto assistem TV, caracterizando o comportamento de segunda tela.
                """)
                
                # Análise automática
                prime_corrs = df_results[df_results['Prime'] == True].groupby('Site')['Correlação'].mean()
                non_prime_corrs = df_results[df_results['Prime'] == False].groupby('Site')['Correlação'].mean()
                
                for site in prime_corrs.index:
                    if site in non_prime_corrs.index:
                        prime_corr = prime_corrs[site]
                        non_prime_corr = non_prime_corrs[site]
                        
                        if prime_corr > 0.3 and prime_corr > non_prime_corr:
                            st.success(f"**{site} apresenta forte indício de uso como segunda tela durante o horário nobre (r = {prime_corr:.2f})**")
                            st.markdown(f"""
                            **Análise detalhada - {site}:**
                            
                            O portal **{site}** mostra uma correlação significativa de **{prime_corr:.2f}** com o rating TV
                            durante o horário nobre, substancialmente maior que nos outros horários ({non_prime_corr:.2f}).
                            Isso é uma forte evidência de uso como segunda tela.
                            
                            **Comportamento do usuário:**
                            - Os espectadores provavelmente buscam informações complementares sobre o conteúdo que estão assistindo
                            - O site serve como extensão da experiência de TV, aprofundando temas abordados na programação
                            - Existe uma sinergia natural entre o conteúdo da TV e o portal {site}
                            
                            **Recomendação:** Explorar ainda mais essa sinergia com conteúdos específicos para segunda tela
                            durante programas de alta audiência, criando uma experiência mais imersiva.
                            """)
                        elif prime_corr > 0.15:
                            st.info(f"**{site} apresenta indício moderado de uso como segunda tela (r = {prime_corr:.2f})**")
                            st.markdown(f"""
                            **Análise detalhada - {site}:**
                            
                            O portal **{site}** mostra uma correlação moderada de **{prime_corr:.2f}** com o rating TV
                            durante o horário nobre. Isso sugere algum uso como segunda tela, mas o comportamento
                            não é tão pronunciado.
                            
                            **Possíveis razões:**
                            - O conteúdo do site pode não estar totalmente alinhado com a programação TV do momento
                            - Os usuários podem estar dividindo atenção entre múltiplas plataformas, não apenas TV e site
                            - Pode haver oportunidades não exploradas para criar mais conexões entre conteúdos
                            
                            **Recomendação:** Avaliar oportunidades para fortalecer a conexão entre o conteúdo do site
                            e a programação de TV, especialmente em programas específicos com maior potencial de sinergia.
                            """)
                        else:
                            st.warning(f"**{site} não apresenta evidência forte de uso como segunda tela (r = {prime_corr:.2f})**")
                            st.markdown(f"""
                            **Análise detalhada - {site}:**
                            
                            O portal **{site}** mostra uma correlação baixa de **{prime_corr:.2f}** com o rating TV,
                            sugerindo que não está sendo significativamente utilizado como segunda tela durante o
                            consumo de TV Linear.
                            
                            **Possíveis razões:**
                            - O conteúdo do site pode estar atendendo a interesses diferentes dos abordados na TV
                            - Os usuários podem estar consumindo o site e a TV em momentos distintos
                            - Pode haver uma desconexão entre a estratégia de conteúdo do site e da TV
                            
                            **Recomendação:** Reavaliar a estratégia de conteúdo, buscando criar mais pontos de
                            contato com a programação TV ou aceitar que o site atende a necessidades diferentes
                            e desenvolver estratégias independentes.
                            """)
        
        # 2.3 Social são complementares, sendo usados como segunda tela da TV linear?
        st.subheader("2.3 Social (IG, TikTok, etc) são complementares, sendo usados como segunda tela da TV linear?")
        
        st.markdown("""
        Esta análise examina se as redes sociais funcionam como "segunda tela" durante o consumo
        de TV Linear, de maneira similar à análise anterior para os sites.
        
        **Metodologia:**
        - Analisamos a correlação entre interações nas redes sociais e rating TV por hora
        - Focamos especialmente no horário nobre (19h-23h)
        - Comparamos o comportamento entre diferentes plataformas sociais
        
        **Como interpretar:**
        - Correlação alta em horário nobre: indica uso complementar à TV
        - Variações entre plataformas: revelam quais redes sociais têm maior sinergia com a TV
        - Padrões distintos ao longo do dia: mostram momentos de uso complementar
        """)
        
        platforms = ["FACEBOOK", "INSTAGRAM", "TIKTOK", "YOUTUBE"]
        
        # Análise similar à anterior, mas focada nas plataformas sociais
        hora_prime = list(range(19, 23))  # Horário nobre
        results = []
        
        for platform in platforms:
            platform_cols = [col for col in rs_cols if platform in col and 'total_interactions' in col]
            if platform_cols:
                df[f'{platform}_total_interactions'] = df[platform_cols].sum(axis=1)
                
                for hora in range(24):
                    df_hora = df[df['hora'] == hora]
                    if len(df_hora) > 5:  # Garantir dados suficientes
                        corr = df_hora[f'{platform}_total_interactions'].corr(df_hora['LINEAR_GLOBO_rat%'])
                        is_prime = hora in hora_prime
                        results.append({
                            'Plataforma': platform,
                            'Hora': hora,
                            'Correlação': corr,
                            'Prime': is_prime
                        })
        
        if results:
            df_results = pd.DataFrame(results)
            
            # Plot de correlação hora a hora por plataforma
            fig = px.line(df_results, x='Hora', y='Correlação', color='Plataforma',
                        title="Correlação entre interações sociais e rating TV por hora",
                        labels={'Correlação': 'Correlação com Rating TV'},
                        line_shape='spline')
            
            # Adicionar faixa para horário nobre
            fig.add_vrect(x0=19, x1=23, 
                        fillcolor="LightSalmon", opacity=0.2,
                        layer="below", line_width=0,
                        annotation_text="Horário Nobre")
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretação do uso de redes sociais como segunda tela:**
            
            O gráfico acima mostra como a correlação entre interações nas redes sociais e o rating da TV Linear
            varia ao longo do dia. A área destacada corresponde ao horário nobre (19h-23h), quando
            a audiência da TV tende a ser maior.
            
            **Padrões a observar:**
            - **Picos de correlação:** Momentos onde há uso sincronizado de redes sociais e TV
            - **Quedas durante certos horários:** Podem indicar competição pela atenção do usuário
            - **Diferenças entre plataformas:** Revelam quais redes têm maior sinergia com a TV
            
            Picos de correlação positiva durante programas específicos sugerem comentários em tempo real
            sobre o conteúdo, caracterizando o comportamento de "TV Social".
            """)
            
            # Análise por plataforma durante horário nobre
            df_prime = df_results[df_results['Prime'] == True].groupby('Plataforma')['Correlação'].mean().reset_index()
            df_prime = df_prime.sort_values('Correlação', ascending=False)
            
            fig = px.bar(df_prime, x='Plataforma', y='Correlação',
                       title="Correlação média das redes sociais com TV durante horário nobre",
                       color='Correlação',
                       color_continuous_scale=['red', 'yellow', 'green'])
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Identificar a melhor plataforma como segunda tela
            best_platform = df_prime.iloc[0]
            if best_platform['Correlação'] > 0.3:
                st.success(f"**{best_platform['Plataforma']} mostra forte evidência de uso como segunda tela no horário nobre (r = {best_platform['Correlação']:.2f})**")
                st.markdown(f"""
                **Destaque para {best_platform['Plataforma']}:**
                
                A plataforma **{best_platform['Plataforma']}** apresenta a correlação mais forte (**{best_platform['Correlação']:.2f}**) 
                com o rating TV durante o horário nobre, indicando um forte uso como segunda tela.
                
                **Características deste comportamento:**
                - Os espectadores provavelmente estão comentando em tempo real sobre os programas
                - A plataforma funciona como espaço de discussão e engajamento com o conteúdo da TV
                - Existe um efeito de "TV Social" pronunciado nesta plataforma
                
                **Oportunidades:**
                - Desenvolver estratégias específicas para {best_platform['Plataforma']} que estimulem o engajamento durante programas de alta audiência
                - Considerar promoções e hashtags específicas para criar momentos virais durante a programação
                - Monitorar e participar das conversas em tempo real para fortalecer o engajamento
                """)
            elif best_platform['Correlação'] > 0.15:
                st.info(f"**{best_platform['Plataforma']} mostra evidência moderada de uso como segunda tela (r = {best_platform['Correlação']:.2f})**")
                st.markdown(f"""
                **Análise para {best_platform['Plataforma']}:**
                
                A plataforma **{best_platform['Plataforma']}** apresenta correlação moderada (**{best_platform['Correlação']:.2f}**) 
                com o rating TV durante o horário nobre, sugerindo algum uso como segunda tela, mas não tão intenso.
                
                **Características deste comportamento:**
                - Alguns espectadores usam a plataforma enquanto assistem TV, mas não é um comportamento dominante
                - O engajamento pode estar concentrado em programas específicos, não em toda a programação
                - Existe potencial para aumentar a sinergia entre a plataforma e a TV
                
                **Recomendação:** Identificar quais tipos de programas geram maior engajamento simultâneo e focar
                estratégias nesses momentos específicos, fortalecendo gradualmente o comportamento de segunda tela.
                """)
            else:
                st.warning("**Nenhuma plataforma social mostra evidência forte de uso como segunda tela**")
                st.markdown("""
                **Análise geral das plataformas sociais:**
                
                Nenhuma plataforma social apresenta evidência forte de uso como segunda tela durante
                o consumo de TV Linear. Isso sugere que as redes sociais e a TV estão sendo consumidas
                em momentos distintos ou por públicos diferentes.
                
                **Possíveis razões:**
                - Fragmentação da atenção: usuários preferem dedicar atenção total a uma tela por vez
                - Desconexão de conteúdo: temas abordados nas redes podem não estar alinhados com a programação TV
                - Comportamento geracional: diferentes grupos etários podem ter preferências distintas de consumo
                
                **Recomendação:** Considerar estratégias que reconheçam a separação entre os meios, focando
                em criar pontes de conteúdo que possam transferir audiência de um para outro, em vez de
                assumir consumo simultâneo.
                """)
    else:
        st.warning("Dados insuficientes para realizar a análise. É necessário ter métricas de Redes Sociais (RS_) e TV Linear (LINEAR_).")

@st.cache_data
def analise_grandes_eventos(df):
    """
    Análise para responder perguntas sobre como os grandes eventos impactam a audiência.
    """
    st.header("Grandes eventos são catalisadores de audiência e engajamento")
    
    st.markdown("""
    Esta análise investiga como eventos significativos afetam a audiência da TV Linear e o engajamento
    nas redes sociais. O objetivo é entender o poder dos grandes eventos como catalisadores de audiência
    e seu papel no "agendamento social" - quando conteúdos se tornam importantes momentos culturais compartilhados.
    
    Exploramos três questões principais:
    1. O impacto da exclusividade de conteúdos vinculados a grandes eventos
    2. O efeito da temporalidade e relevância de eventos no agendamento social
    3. A comparação entre diferentes tipos de eventos e seu impacto na audiência
    """)
    
    # Verificar se temos dados sobre eventos externos
    eventos_cols = [col for col in df.columns if col.startswith('EXTERNO_') and col != 'EXTERNO_dolar' 
                    and col != 'EXTERNO_unemployment_rate' and col != 'EXTERNO_inflation_ipca'
                    and col != 'EXTERNO_selic_rate' and col != 'EXTERNO_indice_cond_economicas']
    
    if 'EXTERNO_FUTEBOL_CONCORRENTE_ON' in df.columns or eventos_cols:
        # 3.1 A exclusividade de conteúdos vinculados a grandes eventos fortalece o agendamento social?
        st.subheader("3.1 A exclusividade de conteúdos vinculados a grandes eventos fortalece o agendamento social?")
        
        st.markdown("""
        Esta análise examina como eventos importantes, especialmente aqueles com caráter de exclusividade,
        impactam a audiência da TV Linear e o engajamento social. O agendamento social refere-se à capacidade
        de um conteúdo criar um "compromisso coletivo" para assistir em tempo real.
        
        **Metodologia:**
        - Comparamos audiência e engajamento durante eventos vs. períodos regulares
        - Analisamos o impacto de eventos de futebol (um exemplo clássico de conteúdo "ao vivo")
        - Medimos o efeito nas interações sociais durante estes eventos
        
        **Como interpretar:**
        - Aumento de rating durante eventos: indica poder de atração de audiência
        - Aumento de share: revela capacidade de dominar a atenção disponível
        - Picos de interações sociais: mostram o "efeito conversação" do evento
        """)
        
        # Análise de audiência durante eventos
        if 'EXTERNO_FUTEBOL_CONCORRENTE_ON' in df.columns:
            # Comparar audiência quando há futebol vs. quando não há
            df_futebol = df.groupby('EXTERNO_FUTEBOL_CONCORRENTE_ON').agg({
                'LINEAR_GLOBO_rat%': 'mean',
                'LINEAR_GLOBO_shr%': 'mean',
                'RS_GLOBO_FACEBOOK_total_interactions': 'mean',
                'RS_GLOBO_INSTAGRAM_total_interactions': 'mean',
                'RS_GLOBO_TIKTOK_total_interactions': 'mean',
                'RS_GLOBO_YOUTUBE_total_interactions': 'mean'
            }).reset_index()
            
            df_futebol['EXTERNO_FUTEBOL_CONCORRENTE_ON'] = df_futebol['EXTERNO_FUTEBOL_CONCORRENTE_ON'].map({
                0: 'Sem Futebol',
                1: 'Com Futebol'
            })
            
            fig = px.bar(df_futebol, x='EXTERNO_FUTEBOL_CONCORRENTE_ON', 
                       y=['LINEAR_GLOBO_rat%', 'LINEAR_GLOBO_shr%'],
                       barmode='group',
                       title='Impacto de Eventos de Futebol na Audiência da TV Linear',
                       labels={'value': 'Valor Médio (%)', 
                              'EXTERNO_FUTEBOL_CONCORRENTE_ON': '', 
                              'variable': 'Métrica'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **Interpretação do impacto de eventos de futebol:**
            
            O gráfico acima compara o rating (% da população total assistindo) e o share (% das TVs ligadas
            sintonizadas no canal) durante períodos com e sem eventos de futebol.
            
            **O que analisar:**
            - **Diferença no Rating:** Mostra a capacidade do evento atrair audiência adicional
            - **Diferença no Share:** Revela a capacidade do evento capturar a atenção disponível
            - **Comparação entre as métricas:** Uma diferença maior no share que no rating sugere que o evento redistribui a audiência existente mais do que atrai novos espectadores
            
            Eventos de futebol são um exemplo clássico de conteúdo "ao vivo" com alto potencial de agendamento social.
            """)
            
            # Impacto nas redes sociais
            social_cols = [col for col in df_futebol.columns if 'total_interactions' in col]
            if social_cols:
                df_social = df_futebol.melt(
                    id_vars=['EXTERNO_FUTEBOL_CONCORRENTE_ON'],
                    value_vars=social_cols,
                    var_name='Plataforma',
                    value_name='Interações'
                )
                
                # Extrair nome da plataforma da coluna
                df_social['Plataforma'] = df_social['Plataforma'].str.extract(r'RS_GLOBO_(\w+)_total')
                
                fig = px.bar(df_social, x='EXTERNO_FUTEBOL_CONCORRENTE_ON', y='Interações',
                           color='Plataforma', barmode='group',
                           title='Impacto de Eventos de Futebol nas Interações em Redes Sociais',
                           labels={'EXTERNO_FUTEBOL_CONCORRENTE_ON': ''})
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretação do impacto social dos eventos:**
                
                O gráfico acima mostra como as interações nas redes sociais variam durante eventos de futebol
                comparados a períodos regulares. Este é um indicador importante do "efeito conversação" que
                grandes eventos podem gerar.
                
                **O que observar:**
                - **Aumento de interações durante eventos:** Indica que o conteúdo está gerando conversação
                - **Variação entre plataformas:** Revela quais redes sociais são mais responsivas a eventos
                - **Proporção do aumento:** Quantifica o poder do evento como catalisador de engajamento
                
                Um aumento significativo nas interações sociais durante eventos confirma seu papel como
                aglutinadores de atenção coletiva e momentos culturais compartilhados.
                """)
                
                # Análise automática
                with_football = df_futebol[df_futebol['EXTERNO_FUTEBOL_CONCORRENTE_ON'] == 'Com Futebol']
                without_football = df_futebol[df_futebol['EXTERNO_FUTEBOL_CONCORRENTE_ON'] == 'Sem Futebol']
                
                if len(with_football) > 0 and len(without_football) > 0:
                    rating_diff = ((with_football['LINEAR_GLOBO_rat%'].values[0] / 
                                  without_football['LINEAR_GLOBO_rat%'].values[0]) - 1) * 100
                    
                    if rating_diff > 10:
                        st.success(f"**Durante eventos de futebol, o rating da TV Linear é {rating_diff:.1f}% maior**")
                        st.markdown(f"""
                        **Conclusão sobre o poder dos eventos:**
                        
                        Os dados mostram um impacto muito significativo dos eventos de futebol na audiência,
                        com um aumento de **{rating_diff:.1f}%** no rating durante estes eventos.
                        
                        Este resultado confirma que conteúdos exclusivos e eventos importantes funcionam como
                        poderosos catalisadores de audiência, criando momentos de "compromisso coletivo" para
                        assistir em tempo real - o verdadeiro agendamento social.
                        
                        **Implicações estratégicas:**
                        - Priorizar aquisição de direitos exclusivos para eventos de alto impacto
                        - Criar estratégias de promoção que amplifiquem o sentido de "momento imperdível"
                        - Desenvolver conteúdos complementares que estendam a experiência do evento
                        """)
                    elif rating_diff < -10:
                        st.error(f"**Durante eventos de futebol, o rating da TV Linear é {abs(rating_diff):.1f}% menor**")
                        st.markdown(f"""
                        **Conclusão sobre competição por eventos:**
                        
                        Os dados mostram que durante eventos de futebol, a audiência da TV Linear cai
                        **{abs(rating_diff):.1f}%**. Isso sugere que eventos concorrentes (possivelmente
                        transmitidos em outros canais) estão atraindo a audiência.
                        
                        **Implicações estratégicas:**
                        - Avaliar a estratégia de programação durante eventos concorrentes
                        - Considerar contra-programação específica que atraia públicos diferentes
                        - Analisar a possibilidade de adquirir direitos para os eventos de maior impacto
                        """)
                    else:
                        st.info("**Eventos de futebol não mostram impacto significativo no rating da TV Linear**")
                        st.markdown(f"""
                        **Conclusão sobre o impacto dos eventos:**
                        
                        Os dados mostram que eventos de futebol não têm um impacto significativo no rating
                        da TV Linear (variação de apenas {rating_diff:.1f}%). Isso sugere que:
                        
                        - O público da TV Linear pode ter pouca sobreposição com fãs de futebol
                        - A programação regular tem poder de retenção similar aos eventos
                        - Outros fatores podem estar influenciando mais a audiência que os eventos
                        
                        **Recomendação:** Focar na qualidade e consistência da programação regular, que parece 
                        ter poder de atração similar aos eventos especiais para o público atual.
                        """)
        
        # 3.2 A temporalidade e relevância de eventos impulsionam o agendamento social?
        st.subheader("3.2 A temporalidade e relevância de eventos impulsionam o agendamento social?")
        
        st.markdown("""
        Esta análise examina como diferentes tipos de eventos afetam a audiência e o engajamento social,
        buscando entender quais características tornam um evento mais poderoso como catalisador.
        
        **Metodologia:**
        - Comparamos o impacto de diferentes eventos externos na audiência TV
        - Medimos tanto o efeito no rating quanto no engajamento social
        - Analisamos a correlação entre impacto na audiência e nas redes sociais
        
        **Como interpretar:**
        - Variação no impacto entre eventos: revela quais características são mais relevantes
        - Correlação entre rating e engajamento: mostra se o impacto é consistente em ambos os meios
        - Eventos de maior efeito: oferecem insights sobre o que gera maior agendamento social
        """)
        
        # Analisar outros eventos externos, se disponíveis
        if eventos_cols:
            # Criar dataframe com média de audiência e engajamento para cada evento
            event_results = []
            
            for event_col in eventos_cols:
                event_name = event_col.replace('EXTERNO_', '')
                
                event_data = df.groupby(event_col).agg({
                    'LINEAR_GLOBO_rat%': 'mean',
                    'LINEAR_GLOBO_shr%': 'mean',
                    'RS_GLOBO_INSTAGRAM_total_interactions': 'mean',
                    'RS_GLOBO_FACEBOOK_total_interactions': 'mean'
                }).reset_index()
                
                if len(event_data) > 1:  # Garantir que temos tanto 0 quanto 1
                    with_event = event_data[event_data[event_col] == 1]
                    without_event = event_data[event_data[event_col] == 0]
                    
                    if len(with_event) > 0 and len(without_event) > 0:
                        rating_impact = (with_event['LINEAR_GLOBO_rat%'].values[0] / 
                                      without_event['LINEAR_GLOBO_rat%'].values[0] - 1) * 100
                        
                        social_cols = [col for col in event_data.columns if 'total_interactions' in col]
                        social_impact = 0
                        
                        if social_cols:
                            social_impacts = [(with_event[col].values[0] / 
                                            without_event[col].values[0] - 1) * 100 
                                           for col in social_cols]
                            social_impact = sum(social_impacts) / len(social_impacts)
                        
                        event_results.append({
                            'Evento': event_name,
                            'Impacto Rating (%)': rating_impact,
                            'Impacto Social (%)': social_impact
                        })
            
            if event_results:
                df_events = pd.DataFrame(event_results)
                
                # Ordenar por impacto no rating
                df_events = df_events.sort_values('Impacto Rating (%)', ascending=False)
                
                # Visualizar impacto dos eventos
                fig = px.bar(df_events, x='Evento', y=['Impacto Rating (%)', 'Impacto Social (%)'],
                           barmode='group',
                           title='Impacto de Diferentes Eventos na Audiência e Engajamento Social',
                           labels={'value': 'Impacto (%)', 'variable': 'Tipo de Impacto'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretação do impacto comparativo entre eventos:**
                
                O gráfico acima compara o impacto percentual de diferentes eventos tanto na audiência da TV Linear
                (Rating) quanto no engajamento em redes sociais.
                
                **O que observar:**
                - **Variação entre eventos:** Alguns eventos têm impacto muito maior que outros
                - **Diferença entre rating e social:** Certos eventos podem afetar mais um meio que outro
                - **Eventos de maior impacto:** Suas características podem revelar o que gera maior agendamento social
                
                Esta análise comparativa é fundamental para identificar quais tipos de conteúdo têm maior poder
                de atração e engajamento, orientando decisões de programação e investimento.
                """)
                
                # Identificar eventos com maior impacto
                top_rating_event = df_events.iloc[0]
                top_social_event = df_events.sort_values('Impacto Social (%)', ascending=False).iloc[0]
                
                st.subheader("Eventos com maior impacto:")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Maior impacto no Rating TV", 
                             top_rating_event['Evento'],
                             f"{top_rating_event['Impacto Rating (%)']:.1f}%")
                
                with col2:
                    st.metric("Maior impacto no Engajamento Social", 
                             top_social_event['Evento'],
                             f"{top_social_event['Impacto Social (%)']:.1f}%")
                
                # Correlação entre impacto no rating e impacto social
                corr = df_events['Impacto Rating (%)'].corr(df_events['Impacto Social (%)'])
                
                if abs(corr) > 0.7:
                    st.success(f"**Há uma forte correlação ({corr:.2f}) entre o impacto de eventos no rating TV e nas redes sociais**")
                    st.markdown(f"""
                    **Análise de correlação entre impactos:**
                    
                    Existe uma correlação forte ({corr:.2f}) entre o impacto dos eventos na audiência TV
                    e no engajamento social. Isso indica que os eventos que atraem mais espectadores também
                    geram mais conversação nas redes sociais.
                    
                    **Implicações:**
                    - Eventos que geram alto agendamento social beneficiam ambos os meios
                    - A audiência TV e o engajamento social se reforçam mutuamente
                    - Estratégias integradas que promovam ambos os meios têm maior potencial de sucesso
                    
                    Esta sinergia sugere que investir em eventos de alto impacto traz benefícios amplificados
                    através do efeito combinado na TV e nas redes sociais.
                    """)
                elif abs(corr) > 0.3:
                    st.info(f"**Há uma correlação moderada ({corr:.2f}) entre o impacto de eventos no rating TV e nas redes sociais**")
                    st.markdown(f"""
                    **Análise de correlação entre impactos:**
                    
                    Existe uma correlação moderada ({corr:.2f}) entre o impacto dos eventos na audiência TV
                    e no engajamento social. Isso sugere que há alguma relação, mas também diferenças importantes
                    no modo como os eventos afetam cada meio.
                    
                    **Implicações:**
                    - Alguns eventos podem ser mais "feitos para TV" enquanto outros são mais "viralizáveis"
                    - É importante analisar as características específicas que fazem um evento performar melhor em cada canal
                    - Estratégias personalizadas para cada meio podem ser necessárias para maximizar o impacto
                    """)
                else:
                    st.warning(f"**Não há correlação significativa ({corr:.2f}) entre o impacto de eventos no rating TV e nas redes sociais**")
                    st.markdown(f"""
                    **Análise de correlação entre impactos:**
                    
                    A correlação fraca ({corr:.2f}) entre impacto na TV e nas redes sociais sugere que
                    estes meios respondem a características diferentes dos eventos. Um evento que gera
                    alta audiência TV não necessariamente gera alto engajamento social, e vice-versa.
                    
                    **Implicações:**
                    - TV e redes sociais atendem a públicos com interesses distintos
                    - É essencial entender as características específicas que funcionam em cada meio
                    - Estratégias separadas para TV e redes sociais podem ser mais eficazes que uma abordagem única
                    
                    Esta distinção sugere a necessidade de uma análise mais granular para identificar
                    os fatores de sucesso específicos para cada plataforma.
                    """)
        
        # 3.3 Alguns eventos têm um impacto maior do que outros na TV linear?
        st.subheader("3.3 Alguns eventos de algumas verticais de conteúdo têm um impacto maior do que outros na TV linear?")
        
        st.markdown("""
        Esta análise compara o impacto de diferentes categorias de eventos na audiência da TV Linear,
        buscando identificar quais verticais de conteúdo têm maior poder de atração.
        
        **Metodologia:**
        - Agrupamos eventos em categorias (Esportes, Entretenimento, Notícias, etc.)
        - Calculamos o impacto médio de cada categoria na audiência
        - Comparamos o efeito no rating e no share por categoria
        
        **Como interpretar:**
        - Diferenças entre categorias: revelam quais verticais têm maior impacto
        - Impacto no rating vs. share: mostra capacidade de atrair novos espectadores vs. redistribuir audiência
        - Número de eventos por categoria: indica consistência do impacto
        """)
        
        if 'EXTERNO_FUTEBOL_CONCORRENTE_ON' in df.columns and eventos_cols:
            # Separar eventos por categoria (se possível identificar)
            eventos_categorias = {}
            for event_col in eventos_cols:
                event_name = event_col.replace('EXTERNO_', '')
                
                # Tentar classificar eventos em categorias
                if 'FUTEBOL' in event_name or 'COPA' in event_name or 'CAMPEONATO' in event_name:
                    categoria = 'Esportes'
                elif 'CARNAVAL' in event_name or 'FESTIVAL' in event_name or 'SHOW' in event_name:
                    categoria = 'Entretenimento'
                elif 'ELEICAO' in event_name or 'POLITICA' in event_name:
                    categoria = 'Notícias'
                else:
                    categoria = 'Outros'
                
                if categoria not in eventos_categorias:
                    eventos_categorias[categoria] = []
                
                eventos_categorias[categoria].append(event_col)
            
            # Calcular impacto médio por categoria
            category_results = []
            
            for categoria, events in eventos_categorias.items():
                if events:
                    # Criar uma coluna que indica se qualquer evento da categoria está ocorrendo
                    df[f'categoria_{categoria}'] = df[events].max(axis=1)
                    
                    # Calcular métricas por categoria
                    cat_data = df.groupby(f'categoria_{categoria}').agg({
                        'LINEAR_GLOBO_rat%': 'mean',
                        'LINEAR_GLOBO_shr%': 'mean'
                    }).reset_index()
                    
                    if len(cat_data) > 1:  # Garantir que temos tanto 0 quanto 1
                        with_cat = cat_data[cat_data[f'categoria_{categoria}'] == 1]
                        without_cat = cat_data[cat_data[f'categoria_{categoria}'] == 0]
                        
                        if len(with_cat) > 0 and len(without_cat) > 0:
                            rating_impact = (with_cat['LINEAR_GLOBO_rat%'].values[0] / 
                                          without_cat['LINEAR_GLOBO_rat%'].values[0] - 1) * 100
                            
                            share_impact = (with_cat['LINEAR_GLOBO_shr%'].values[0] / 
                                          without_cat['LINEAR_GLOBO_shr%'].values[0] - 1) * 100
                            
                            category_results.append({
                                'Categoria': categoria,
                                'Impacto Rating (%)': rating_impact,
                                'Impacto Share (%)': share_impact,
                                'Número de Eventos': len(events)
                            })
            
            if category_results:
                df_categories = pd.DataFrame(category_results)
                
                # Ordenar por impacto no rating
                df_categories = df_categories.sort_values('Impacto Rating (%)', ascending=False)
                
                # Visualizar impacto por categoria
                fig = px.bar(df_categories, x='Categoria', y=['Impacto Rating (%)', 'Impacto Share (%)'],
                           barmode='group',
                           title='Impacto de Categorias de Eventos na TV Linear',
                           labels={'value': 'Impacto (%)', 'variable': 'Tipo de Impacto'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretação do impacto por categoria de evento:**
                
                O gráfico acima compara o impacto percentual de diferentes categorias de eventos na audiência da TV Linear,
                tanto em termos de rating (% da população assistindo) quanto de share (% das TVs ligadas sintonizadas no canal).
                
                **O que observar:**
                - **Variação entre categorias:** Algumas verticais de conteúdo têm impacto muito maior que outras
                - **Diferença entre rating e share:** Indica se a categoria atrai novos espectadores ou apenas redistribui a audiência
                - **Impacto proporcional ao número de eventos:** Revela consistência do efeito dentro da categoria
                
                Esta análise por categoria é fundamental para decisões estratégicas sobre aquisição de direitos
                e investimento em diferentes verticais de conteúdo.
                """)
                
                # Análise automática
                st.subheader("Conclusões sobre impacto de categorias de eventos:")
                
                for i, row in df_categories.iterrows():
                    categoria = row['Categoria']
                    rating_impact = row['Impacto Rating (%)']
                    num_events = row['Número de Eventos']
                    
                    if rating_impact > 15:
                        st.success(f"**{categoria}: Impacto muito alto na audiência (+{rating_impact:.1f}%)**")
                        st.markdown(f"""
                        **Análise detalhada - {categoria}:**
                        
                        Eventos da categoria **{categoria}** têm um impacto excepcional de **+{rating_impact:.1f}%** 
                        na audiência da TV Linear, baseado em {num_events} eventos analisados.
                        
                        **Características potenciais de sucesso:**
                        - Alto valor de "exclusividade" e temporalidade
                        - Forte componente social/cultural compartilhado
                        - Conteúdo que beneficia da experiência "ao vivo"
                        
                        **Recomendação:** Priorizar aquisição de direitos e investimento em eventos desta categoria,
                        com foco em maximizar seu impacto através de promoção antecipada e cobertura extensiva.
                        """)
                    elif rating_impact > 5:
                        st.info(f"**{categoria}: Impacto moderado na audiência (+{rating_impact:.1f}%)**")
                        st.markdown(f"""
                        **Análise detalhada - {categoria}:**
                        
                        Eventos da categoria **{categoria}** têm um impacto positivo de **+{rating_impact:.1f}%** 
                        na audiência da TV Linear, baseado em {num_events} eventos analisados.
                        
                        **Características potenciais:**
                        - Valor moderado de "compromisso" para assistir em tempo real
                        - Algum componente social/cultural compartilhado
                        - Conteúdo que beneficia da experiência conjunta
                        
                        **Recomendação:** Avaliar o custo-benefício destes eventos, identificando características específicas
                        que possam amplificar seu impacto através de estratégias de programação e promoção otimizadas.
                        """)
                    elif rating_impact < -5:
                        st.error(f"**{categoria}: Impacto negativo na audiência ({rating_impact:.1f}%)**")
                        st.markdown(f"""
                        **Análise detalhada - {categoria}:**
                        
                        Eventos da categoria **{categoria}** têm um impacto negativo de **{rating_impact:.1f}%** 
                        na audiência da TV Linear, baseado em {num_events} eventos analisados.
                        
                        **Possíveis razões:**
                        - Competição direta de outros canais transmitindo eventos similares
                        - Público-alvo diferente do perfil habitual da audiência do canal
                        - Falta de estratégia eficaz de programação durante estes eventos
                        
                        **Recomendação:** Desenvolver estratégia específica de contra-programação durante eventos
                        desta categoria, ou considerar mudança de abordagem na cobertura destes eventos.
                        """)
                    else:
                        st.warning(f"**{categoria}: Impacto limitado na audiência ({rating_impact:.1f}%)**")
                        st.markdown(f"""
                        **Análise detalhada - {categoria}:**
                        
                        Eventos da categoria **{categoria}** têm impacto limitado de **{rating_impact:.1f}%** 
                        na audiência da TV Linear, baseado em {num_events} eventos analisados.
                        
                        **Possíveis razões:**
                        - Menor apelo de "assistir ao vivo" comparado a outras categorias
                        - Competição com outras formas de consumo deste tipo de conteúdo
                        - Menor capacidade de criar "momentos culturais compartilhados"
                        
                        **Recomendação:** Avaliar o investimento nesta categoria com base em outros benefícios
                        além do impacto direto em audiência, como posicionamento de marca ou completude da oferta.
                        """)
    else:
        st.warning("Dados insuficientes para realizar a análise. É necessário ter informações sobre eventos externos.")

@st.cache_data
def analise_fatores_externos(df):
    """
    Análise para responder perguntas sobre como fatores externos impactam a audiência.
    """
    st.header("Fatores externos impactam a nossa audiência da TV")
    
    st.markdown("""
    Esta análise investiga como fatores econômicos e externos influenciam o comportamento da audiência da TV.
    O objetivo é entender como o ambiente macro afeta os padrões de consumo de mídia, permitindo
    antecipar tendências e adaptar estratégias.
    
    Exploramos duas questões principais:
    1. O impacto de indicadores macroeconômicos (inflação, desemprego, etc.) no consumo de entretenimento
    2. Como fatores ambientais e de mobilidade afetam os hábitos de consumo de TV
    """)
    
    # Verificar se temos dados econômicos
    eco_cols = ['EXTERNO_dolar', 'EXTERNO_unemployment_rate', 'EXTERNO_inflation_ipca', 
                'EXTERNO_selic_rate', 'EXTERNO_indice_cond_economicas']
    
    eco_available = [col for col in eco_cols if col in df.columns]
    
    if eco_available:
        # 4.1 Dados macro-econômicos podem explicar comportamentos de entretenimento?
        st.subheader("4.1 Dados macro-econômicos podem explicar comportamentos de compra e gastos com entretenimento?")
        
        st.markdown("""
        Esta análise examina como indicadores econômicos (como desemprego, inflação e taxa de juros)
        influenciam o consumo de mídia, tanto em TV Linear quanto em streaming. A premissa é que
        mudanças econômicas podem alterar os hábitos de entretenimento e gastos domésticos.
        
        **Metodologia:**
        - Analisamos a correlação entre indicadores econômicos e métricas de audiência
        - Comparamos o efeito em TV Linear vs. streaming (quando disponível)
        - Examinamos tendências temporais de indicadores econômicos e audiência
        
        **Como interpretar:**
        - Correlações fortes: indicam que fatores econômicos impactam significativamente o consumo de mídia
        - Diferenças entre TV e streaming: revelam possíveis mudanças de comportamento baseadas em custo
        - Tendências temporais: mostram como mudanças econômicas graduais afetam a audiência ao longo do tempo
        """)
        
        # Agrupar dados por dia para análise econômica
        df['data_apenas'] = pd.to_datetime(df['data_hora']).dt.date
        df_diario = df.groupby('data_apenas').mean().reset_index()
        
        # Correlação entre variáveis econômicas e audiência/streaming
        correlation_vars = eco_available + ['LINEAR_GLOBO_rat%', 'LINEAR_GLOBO_shr%']
        
        # Adicionar variáveis de streaming se disponíveis
        if 'GP_usuários_assinantes_' in df.columns:
            correlation_vars.append('GP_usuários_assinantes_')
        
        correlation_matrix = df_diario[correlation_vars].corr()
        
        # Visualizar matriz de correlação
        fig = px.imshow(correlation_matrix,
                       text_auto=True,
                       color_continuous_scale='RdBu_r',
                       title='Correlação entre Fatores Econômicos e Audiência')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Interpretação da matriz de correlação:**
        
        O heatmap acima mostra as correlações entre indicadores econômicos e métricas de audiência.
        Cores mais intensas indicam correlações mais fortes, sendo azul para positivas e vermelho para negativas.
        
        **O que observar:**
        - **Correlações com inflação/desemprego:** Podem indicar como o custo de vida afeta o consumo de entretenimento
        - **Efeito da taxa de juros:** Pode estar relacionado à disposição para gastos com assinaturas
        - **Variações entre TV Linear e streaming:** Revelam diferenças na sensibilidade a fatores econômicos
        
        Uma mudança nos padrões de correlação pode sinalizar alterações no comportamento do consumidor
        em resposta a condições econômicas específicas.
        """)
        
        # Análise detalhada por indicador econômico
        st.subheader("Impacto de indicadores econômicos específicos:")
        
        # Para cada indicador econômico, mostrar sua relação com audiência
        eco_analysis = []
        for eco_col in eco_available:
            # Extrair nome amigável do indicador
            eco_name = eco_col.replace('EXTERNO_', '').replace('_', ' ').title()
            
            # Calcular correlação com audiência
            corr_rating = df_diario[eco_col].corr(df_diario['LINEAR_GLOBO_rat%'])
            
            # Calcular correlação com streaming (se disponível)
            corr_streaming = None
            if 'GP_usuários_assinantes_' in df_diario.columns:
                corr_streaming = df_diario[eco_col].corr(df_diario['GP_usuários_assinantes_'])
            
            eco_analysis.append({
                'Indicador': eco_name,
                'Correlação TV': corr_rating,
                'Correlação Streaming': corr_streaming
            })
        
        df_eco = pd.DataFrame(eco_analysis)
        
        # Visualizar impacto dos indicadores
        fig = px.bar(df_eco, x='Indicador', y=['Correlação TV', 'Correlação Streaming'],
                   barmode='group',
                   title='Impacto de Indicadores Econômicos na Audiência',
                   labels={'value': 'Correlação', 'variable': 'Tipo de Audiência'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        **Análise comparativa dos indicadores econômicos:**
        
        O gráfico acima compara como diferentes indicadores econômicos se correlacionam com a audiência
        de TV Linear e streaming (quando disponível).
        
        **Interpretações importantes:**
        - **Correlações positivas:** Um aumento no indicador está associado a maior audiência
        - **Correlações negativas:** Um aumento no indicador está associado a menor audiência
        - **Diferenças entre TV e streaming:** Revelam possíveis tendências de substituição ou complementaridade
        
        Esta análise permite identificar quais indicadores econômicos são mais relevantes para prever
        comportamentos de audiência e adaptar estratégias de acordo com mudanças econômicas.
        """)
        
        # Análise específica para desemprego, inflação e taxa de juros
        if 'EXTERNO_unemployment_rate' in df_diario.columns and 'LINEAR_GLOBO_rat%' in df_diario.columns:
            fig = px.scatter(df_diario, x='EXTERNO_unemployment_rate', y='LINEAR_GLOBO_rat%',
                           trendline="ols",
                           title='Relação entre Taxa de Desemprego e Audiência TV',
                           labels={'EXTERNO_unemployment_rate': 'Taxa de Desemprego (%)',
                                  'LINEAR_GLOBO_rat%': 'Rating TV (%)'})
            
            st.plotly_chart(fig, use_container_width=True)
            
            corr = df_diario['EXTERNO_unemployment_rate'].corr(df_diario['LINEAR_GLOBO_rat%'])
            
            if corr > 0.3:
                st.success(f"**Há uma correlação positiva significativa ({corr:.2f}) entre desemprego e audiência TV, sugerindo que em períodos de maior desemprego as pessoas assistem mais TV**")
                st.markdown(f"""
                **Análise da relação desemprego-audiência:**
                
                Os dados mostram uma correlação positiva significativa (**{corr:.2f}**) entre a taxa de desemprego
                e a audiência de TV Linear. Isso confirma a hipótese de que, em períodos de maior desemprego,
                as pessoas tendem a consumir mais conteúdo de TV - possivelmente devido a:
                
                - Mais tempo disponível em casa
                - Busca por entretenimento de baixo custo
                - Redução em atividades de lazer fora de casa
                
                **Implicação estratégica:** Durante períodos de aumento no desemprego, pode ser vantajoso
                ajustar a programação para atender às necessidades deste público crescente, com conteúdos
                que ofereçam escapismo, desenvolvimento profissional ou entretenimento econômico.
                """)
            elif corr < -0.3:
                st.error(f"**Há uma correlação negativa significativa ({corr:.2f}) entre desemprego e audiência TV**")
                st.markdown(f"""
                **Análise da relação desemprego-audiência:**
                
                Surpreendentemente, os dados mostram uma correlação negativa significativa (**{corr:.2f}**) 
                entre a taxa de desemprego e a audiência de TV Linear, contrariando a hipótese tradicional.
                Possíveis explicações incluem:
                
                - Mudanças para alternativas de entretenimento mais econômicas (como streaming compartilhado)
                - Cortes de serviços pagos durante períodos de restrição financeira
                - Impacto psicológico que reduz o interesse em conteúdo de entretenimento
                
                **Implicação estratégica:** Em períodos de aumento do desemprego, pode ser necessário
                reconsiderar modelos de precificação, oferecendo pacotes mais acessíveis ou flexíveis
                para reter audiência.
                """)
            else:
                st.info(f"**Não há correlação significativa ({corr:.2f}) entre desemprego e audiência TV**")
                st.markdown(f"""
                **Análise da relação desemprego-audiência:**
                
                Os dados indicam uma correlação fraca (**{corr:.2f}**) entre a taxa de desemprego
                e a audiência de TV Linear. Isso sugere que a audiência de TV não é significativamente
                afetada por variações na taxa de desemprego, possivelmente porque:
                
                - Outros fatores têm maior influência nos hábitos de consumo de mídia
                - Os efeitos do desemprego na audiência variam entre diferentes segmentos, neutralizando o efeito global
                - A TV se tornou um bem de consumo essencial, menos sujeito a flutuações econômicas
                
                **Implicação estratégica:** As decisões de programação e investimento podem ser baseadas
                em outros fatores além das tendências de desemprego, como qualidade de conteúdo ou preferências
                demográficas específicas.
                """)
        
        # Tendências de audiência vs. economia ao longo do tempo
        if eco_available and 'LINEAR_GLOBO_rat%' in df_diario.columns:
            st.subheader("Tendências de audiência vs. economia ao longo do tempo:")
            
            # Selecionar um indicador econômico representativo
            if 'EXTERNO_indice_cond_economicas' in df_diario.columns:
                eco_indicator = 'EXTERNO_indice_cond_economicas'
                eco_name = 'Índice de Condições Econômicas'
            elif 'EXTERNO_unemployment_rate' in df_diario.columns:
                eco_indicator = 'EXTERNO_unemployment_rate'
                eco_name = 'Taxa de Desemprego'
            else:
                eco_indicator = eco_available[0]
                eco_name = eco_indicator.replace('EXTERNO_', '').replace('_', ' ').title()
            
            # Normalizar valores para comparação
            df_diario['rating_norm'] = df_diario['LINEAR_GLOBO_rat%'] / df_diario['LINEAR_GLOBO_rat%'].max()
            df_diario['eco_norm'] = df_diario[eco_indicator] / df_diario[eco_indicator].max()
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=df_diario['data_apenas'], y=df_diario['rating_norm'], 
                                    name='Rating TV (Normalizado)',
                                    line=dict(color='#1f77b4')))
            
            fig.add_trace(go.Scatter(x=df_diario['data_apenas'], y=df_diario['eco_norm'], 
                                    name=f'{eco_name} (Normalizado)',
                                    line=dict(color='#ff7f0e')))
            
            fig.update_layout(title=f'Tendência: Rating TV vs. {eco_name}',
                            xaxis_title='Data',
                            yaxis_title='Valor Normalizado (0-1)')
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown(f"""
            **Interpretação da tendência temporal:**
            
            O gráfico acima mostra a evolução ao longo do tempo do rating da TV Linear e do indicador econômico
            "{eco_name}", com ambos os valores normalizados para facilitar a comparação.
            
            **O que observar:**
            - **Movimentos paralelos:** Indicam que o indicador econômico e a audiência TV estão se movendo juntos
            - **Movimentos contrários:** Sugerem uma relação inversa entre o indicador e a audiência
            - **Defasagens (lags):** Podem indicar que mudanças econômicas afetam a audiência com atraso
            - **Pontos de inflexão:** Momentos onde a relação entre os dois fatores muda
            
            Esta análise temporal complementa a correlação estática, revelando como a relação entre
            economia e audiência evolui ao longo do tempo e ajudando a identificar padrões sazonais ou cíclicos.
            """)
        
        # 4.2 Fatores ambientais e dados de mobilidade
        st.subheader("4.2 Fatores ambientais e dados de mobilidade definem o comportamento e a rotina das pessoas?")
        
        st.markdown("🚧 **WORK IN PROGRESS** 🚧")
        st.info("Esta análise requer dados de fatores ambientais e mobilidade que não estão disponíveis no conjunto de dados atual.")
        
        st.markdown("""
        **O que pretendemos analisar aqui:**
        
        Esta seção investigaria como fatores ambientais (clima, temperatura, eventos naturais) e padrões
        de mobilidade (deslocamento urbano, tráfego, home office) influenciam o consumo de TV Linear.
        
        **Metodologia que seria utilizada:**
        - Correlação entre condições climáticas e picos de audiência
        - Análise de como períodos de restrição de mobilidade afetam o consumo de mídia
        - Comparação de padrões de audiência em dias úteis vs. finais de semana
        - Efeito de eventos climáticos extremos na audiência
        
        **Dados necessários para esta análise:**
        - Informações meteorológicas (temperatura, precipitação, etc.)
        - Dados de mobilidade urbana (índices de congestionamento, uso de transporte público)
        - Informações sobre padrões de trabalho (dias de home office, feriados)
        
        Esta análise será implementada quando os dados ambientais e de mobilidade estiverem disponíveis.
        """)
        
    else:
        st.warning("Dados insuficientes para realizar a análise. É necessário ter informações sobre indicadores econômicos externos.")

@st.cache_data
def analise_percepcao_marca(df):
    """
    Análise para responder perguntas sobre como a percepção da marca influencia a audiência.
    """
    st.header("A percepção da marca influencia na audiência")
    
    st.markdown("""
    Esta análise investiga como a percepção da marca Globo e a viralização de conteúdos
    influenciam a audiência da TV Linear, especialmente entre as novas gerações.
    
    Exploramos duas questões principais:
    1. O impacto da percepção da audiência em relação à marca Globo na audiência da TV Linear
    2. Como a viralização de conteúdos nas redes sociais pode atrair novas gerações para a TV
    """)
    
    # 5.1 A percepção da nossa audiência em relação à marca Globo afeta a audiência?
    st.subheader("5.1 A percepção da nossa audiência em relação à marca Globo afeta a audiência da TV linear?")
    
    st.markdown("🚧 **WORK IN PROGRESS** 🚧")
    st.info("Esta análise requer dados de percepção de marca (como pesquisas de opinião, sentimento em redes sociais) que não estão disponíveis no conjunto de dados atual.")
    
    st.markdown("""
    **O que pretendemos analisar aqui:**
    
    Esta seção investigaria como a percepção da marca Globo (sua imagem, reputação, sentimento associado)
    influencia os níveis de audiência da TV Linear, especialmente entre diferentes segmentos demográficos.
    
    **Metodologia que seria utilizada:**
    - Análise de sentimento sobre a marca em redes sociais correlacionada com audiência
    - Segmentação por características demográficas (idade, região, etc.)
    - Estudo de como eventos relacionados à marca afetam a audiência de curto e longo prazo
    - Correlação entre métricas de saúde da marca e audiência
    
    **Dados necessários para esta análise:**
    - Pesquisas de opinião sobre a marca
    - Dados de sentimento em redes sociais
    - Métricas de reputação de marca
    - Segmentação demográfica dos dados de audiência
    
    Esta análise será implementada quando os dados de percepção de marca estiverem disponíveis.
    """)
    
    # 5.2 A viralização de conteúdos da TV linear nas redes sociais
    st.subheader("5.2 A viralização de conteúdos da TV linear nas redes sociais pode atrair novas gerações?")
    
    st.markdown("""
    Esta análise examina como a viralização de conteúdos da TV Linear nas redes sociais
    influencia a audiência futura, com foco especial nas novas gerações (que tradicionalmente
    consomem menos TV Linear).
    
    **Metodologia:**
    - Avaliamos o efeito de compartilhamentos nas redes sociais na audiência futura
    - Analisamos como esse efeito varia ao longo de diferentes períodos de tempo (1, 2, 3 e 7 dias)
    - Comparamos o impacto de diferentes plataformas sociais
    - Examinamos se o efeito é mais forte em horários com público mais jovem
    
    **Como interpretar:**
    - Correlações positivas fortes: indicam que a viralização atrai audiência futura
    - Diferenças entre plataformas: revelam quais redes têm maior poder de conversão
    - Variações entre públicos: mostram se o efeito é mais significativo para novas gerações
    """)
    
    # Verificar se temos métricas de viralização
    viral_metrics = ['RS_GLOBO_INSTAGRAM_nr_shares', 'RS_GLOBO_FACEBOOK_nr_shares', 
                     'RS_GLOBO_TIKTOK_nr_shares', 'RS_GLOBO_YOUTUBE_nr_shares']
    
    available_viral = [col for col in viral_metrics if col in df.columns]
    
    if available_viral:
        # Agrupar dados por dia
        df['data_apenas'] = pd.to_datetime(df['data_hora']).dt.date
        
        # Selecionar apenas colunas numéricas para agregação
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        # Garantir que temos apenas as colunas que realmente existem nos dados
        cols_to_group = []
        for col in numeric_cols:
            if col in df.columns:
                cols_to_group.append(col)
        
        # Agrupar apenas colunas numéricas
        df_diario = df.groupby('data_apenas')[cols_to_group].mean().reset_index()
        
        # Verificar se conseguimos criar a métrica de viralização total
        viral_cols_available = [col for col in available_viral if col in df_diario.columns]
        
        if len(viral_cols_available) > 0:
            # Criar uma métrica de viralização total
            df_diario['total_shares'] = df_diario[viral_cols_available].sum(axis=1)
            
            # Criar lags para medir efeito posterior da viralização
            for lag in [1, 2, 3, 7]:  # 1, 2, 3 e 7 dias após
                df_diario[f'total_shares_lag{lag}'] = df_diario['total_shares'].shift(lag)
            
            # Remover linhas com NaN devido aos lags
            df_diario = df_diario.dropna()
            
            # Calcular correlação entre compartilhamentos e audiência futura
            if 'LINEAR_GLOBO_rat%' in df_diario.columns:
                correlations = []
                for lag in [1, 2, 3, 7]:
                    corr = df_diario[f'total_shares_lag{lag}'].corr(df_diario['LINEAR_GLOBO_rat%'])
                    correlations.append({
                        'Dias após viralização': lag,
                        'Correlação com audiência': corr
                    })
                
                df_correlations = pd.DataFrame(correlations)
                
                # Visualizar o efeito temporal da viralização
                fig = px.line(df_correlations, x='Dias após viralização', y='Correlação com audiência',
                             markers=True,
                             title='Efeito temporal da viralização na audiência TV',
                             labels={'Correlação com audiência': 'Correlação com Rating TV'})
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("""
                **Interpretação do efeito temporal da viralização:**
                
                O gráfico acima mostra como a correlação entre compartilhamentos nas redes sociais
                e a audiência da TV Linear varia ao longo de diferentes períodos após a viralização
                (1, 2, 3 e 7 dias).
                
                **O que observar:**
                - **Pico de correlação:** Indica o momento em que a viralização tem maior impacto na audiência
                - **Persistência do efeito:** Mostra quanto tempo o efeito da viralização dura
                - **Declínio ou aumento com o tempo:** Revela se o impacto é imediato ou tem efeito cumulativo
                
                Esta análise temporal é crucial para entender não apenas se a viralização funciona,
                mas também quando seu efeito é maximizado, permitindo estratégias de programação que
                aproveitem esses padrões.
                """)
                
                # Análise de viralização por plataforma
                platform_corrs = []
                for platform in ['INSTAGRAM', 'FACEBOOK', 'TIKTOK', 'YOUTUBE']:
                    share_col = f'RS_GLOBO_{platform}_nr_shares'
                    
                    # Verificar se a coluna específica existe no df_diario
                    if share_col in df_diario.columns:
                        # Criar uma nova coluna para o lag
                        lag_col_name = f'{platform}_lag1'
                        # Criar o lag para esta plataforma específica
                        df_diario[lag_col_name] = df_diario[share_col].shift(1)
                        
                        # Calcular correlação
                        corr = df_diario[lag_col_name].corr(df_diario['LINEAR_GLOBO_rat%'])
                        platform_corrs.append({
                            'Plataforma': platform,
                            'Correlação com audiência': corr
                        })
                
                if platform_corrs:
                    df_platforms = pd.DataFrame(platform_corrs)
                    
                    # Ordenar por correlação
                    df_platforms = df_platforms.sort_values('Correlação com audiência', ascending=False)
                    
                    # Visualizar impacto por plataforma
                    fig = px.bar(df_platforms, x='Plataforma', y='Correlação com audiência',
                               title='Impacto da viralização por plataforma na audiência TV futura',
                               labels={'Correlação com audiência': 'Correlação com Rating TV'},
                               color='Correlação com audiência',
                               color_continuous_scale=['red', 'yellow', 'green'])
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Análise automática
                    if len(df_platforms) > 0:
                        best_platform = df_platforms.iloc[0]
                        if best_platform['Correlação com audiência'] > 0.3:
                            st.success(f"**{best_platform['Plataforma']} mostra forte influência positiva da viralização na audiência futura (r = {best_platform['Correlação com audiência']:.2f})**")
                            st.markdown(f"""
                            **Análise por plataforma:**
                            
                            Entre todas as plataformas analisadas, **{best_platform['Plataforma']}** mostra a
                            correlação mais forte (**{best_platform['Correlação com audiência']:.2f}**) entre
                            compartilhamentos e audiência futura da TV Linear.
                            
                            **Características potenciais do sucesso:**
                            - Perfil demográfico dos usuários mais alinhado com potenciais espectadores de TV
                            - Formato de compartilhamento que gera maior interesse e conversão
                            - Conteúdos viralizados nesta plataforma podem ter maior poder de atração
                            
                            **Recomendação:** Priorizar estratégias de viralização no {best_platform['Plataforma']},
                            desenvolver conteúdos especificamente formatados para esta plataforma, e criar
                            chamadas para ação mais diretas que levem os usuários para a TV Linear.
                            """)
                        elif best_platform['Correlação com audiência'] > 0.15:
                            st.info(f"**{best_platform['Plataforma']} mostra influência moderada da viralização na audiência futura (r = {best_platform['Correlação com audiência']:.2f})**")
                            st.markdown(f"""
                            **Análise por plataforma:**
                            
                            A plataforma **{best_platform['Plataforma']}** mostra uma correlação moderada
                            (**{best_platform['Correlação com audiência']:.2f}**) entre compartilhamentos e
                            audiência futura da TV Linear.
                            
                            **Características potenciais:**
                            - Alguma capacidade de conversão, mas com limitações
                            - Público parcialmente alinhado com potenciais espectadores de TV
                            - Conteúdos viralizados geram interesse, mas nem sempre conversão efetiva
                            
                            **Recomendação:** Avaliar quais tipos específicos de conteúdo nesta plataforma
                            têm maior taxa de conversão, e focar em formatos e chamadas para ação que
                            maximizem a transferência de audiência para a TV Linear.
                            """)
                        else:
                            st.warning("**Nenhuma plataforma mostra forte evidência de que a viralização aumenta a audiência futura**")
                            st.markdown("""
                            **Análise por plataforma:**
                            
                            Nenhuma plataforma social mostra uma correlação forte entre compartilhamentos
                            e audiência futura da TV Linear, sugerindo que a viralização de conteúdos
                            atualmente não está sendo eficaz em converter usuários de redes sociais em
                            espectadores de TV.
                            
                            **Possíveis razões:**
                            - Desconexão entre o conteúdo viralizado e a programação de TV
                            - Ausência de chamadas para ação efetivas direcionando para a TV
                            - Diferença fundamental entre o público das redes sociais e potenciais espectadores de TV
                            
                            **Recomendação:** Reconsiderar a estratégia de viralização, com foco em criar
                            conteúdos que despertem curiosidade específica sobre a programação de TV e incluam
                            chamadas para ação mais claras e atraentes.
                            """)
            
            # Análise para novas gerações (por proxy de horário)
            df['hora'] = pd.to_datetime(df['data_hora']).dt.hour
            
            # Horários com maior presença de público jovem (19h-23h)
            df['horario_jovem'] = df['hora'].between(19, 23)
            
            # Selecionar apenas as colunas que realmente existem
            cols_for_analysis = ['data_apenas', 'horario_jovem']
            
            # Adicionar colunas numéricas necessárias se existirem
            if 'LINEAR_GLOBO_rat%' in df.columns:
                cols_for_analysis.append('LINEAR_GLOBO_rat%')
            
            # Adicionar colunas de viralização específicas
            target_viral_cols = ['RS_GLOBO_INSTAGRAM_nr_shares', 'RS_GLOBO_TIKTOK_nr_shares']
            for col in target_viral_cols:
                if col in df.columns:
                    cols_for_analysis.append(col)
            
            # Verificar se temos pelo menos uma coluna de viralização
            has_viral_cols = any(col in cols_for_analysis for col in target_viral_cols)
            
            if has_viral_cols and 'LINEAR_GLOBO_rat%' in cols_for_analysis:
                # Criar um dataframe apenas com as colunas que precisamos
                df_horario = df[cols_for_analysis].copy()
                
                # Calcular a média de cada métrica por dia e tipo de horário
                df_horario = df_horario.groupby(['data_apenas', 'horario_jovem']).mean().reset_index()
                
                # Criar a métrica de viralização jovem
                viral_jovem_cols = [col for col in target_viral_cols if col in df_horario.columns]
                
                if viral_jovem_cols:
                    # Se tivermos colunas de viralização, criar a soma
                    df_horario['viral_jovem'] = df_horario[viral_jovem_cols].sum(axis=1)
                    
                    # Separar dados para horários jovens e não-jovens
                    df_jovem = df_horario[df_horario['horario_jovem'] == True]
                    df_nao_jovem = df_horario[df_horario['horario_jovem'] == False]
                    
                    # Calcular correlação para cada grupo se tivermos dados suficientes
                    if len(df_jovem) > 5 and len(df_nao_jovem) > 5:
                        corr_jovem = df_jovem['viral_jovem'].corr(df_jovem['LINEAR_GLOBO_rat%'])
                        corr_nao_jovem = df_nao_jovem['viral_jovem'].corr(df_nao_jovem['LINEAR_GLOBO_rat%'])
                        
                        # Visualizar comparação
                        corr_data = pd.DataFrame([
                            {'Público': 'Horário Jovem (19h-23h)', 'Correlação': corr_jovem},
                            {'Público': 'Outros Horários', 'Correlação': corr_nao_jovem}
                        ])
                        
                        fig = px.bar(corr_data, x='Público', y='Correlação',
                                   title='Impacto da viralização na audiência por tipo de público',
                                   labels={'Correlação': 'Correlação Viralização vs. Rating TV'},
                                   color='Correlação',
                                   color_continuous_scale=['red', 'yellow', 'green'])
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown("""
                        **Interpretação do impacto por tipo de público:**
                        
                        O gráfico acima compara a correlação entre viralização e audiência TV para
                        horários com maior presença de público jovem (19h-23h) versus outros horários.
                        
                        **O que observar:**
                        - **Diferença entre os grupos:** Indica se a viralização tem efeito diferente em públicos jovens
                        - **Correlação mais forte em horário jovem:** Sugere que jovens são mais influenciados pela viralização
                        - **Correlação mais forte em outros horários:** Indica que públicos tradicionais são mais sensíveis à viralização
                        
                        Esta comparação é crucial para entender se a viralização é uma estratégia eficaz
                        especificamente para atrair públicos mais jovens para a TV Linear.
                        """)
                        
                        # Análise automática
                        if corr_jovem > corr_nao_jovem and corr_jovem > 0.2:
                            st.success(f"**A viralização tem um impacto mais forte na audiência em horários com público mais jovem (r = {corr_jovem:.2f} vs. {corr_nao_jovem:.2f})**")
                            st.markdown(f"""
                            **Conclusão sobre impacto nas novas gerações:**
                            
                            A análise mostra que a viralização tem um impacto significativamente maior
                            na audiência durante horários com maior presença de público jovem (correlação de **{corr_jovem:.2f}**)
                            comparado a outros horários ({corr_nao_jovem:.2f}).
                            
                            **Implicações importantes:**
                            - Novas gerações são mais suscetíveis à influência de conteúdos virais
                            - A viralização é uma estratégia eficaz para atrair público jovem para a TV Linear
                            - Existe uma ponte natural entre redes sociais e TV para este público
                            
                            **Recomendação:** Priorizar estratégias de viralização especificamente direcionadas ao público jovem,
                            com foco em conteúdos exibidos no horário nobre, que é quando este público está mais disponível
                            e receptivo à transferência entre plataformas.
                            """)
                        elif corr_nao_jovem > corr_jovem and corr_nao_jovem > 0.2:
                            st.info(f"**A viralização tem um impacto mais forte na audiência em horários com público mais velho (r = {corr_nao_jovem:.2f} vs. {corr_jovem:.2f})**")
                            st.markdown(f"""
                            **Conclusão sobre impacto nas diferentes gerações:**
                            
                            Surpreendentemente, a análise mostra que a viralização tem um impacto significativamente maior
                            na audiência durante horários com presença de público mais velho (correlação de **{corr_nao_jovem:.2f}**)
                            comparado a horários jovens ({corr_jovem:.2f}).
                            
                            **Possíveis explicações:**
                            - Públicos mais velhos podem ser mais fiéis à TV e mais fáceis de reconverter
                            - Conteúdos que viralizam atualmente podem ressoar mais com audiências tradicionais
                            - Jovens podem consumir conteúdo viral sem necessariamente migrar para a TV
                            
                            **Recomendação:** Considerar adaptar a estratégia de viralização para atrair novos espectadores
                            de faixas etárias intermediárias, que são mais propensas a converter engajamento social em
                            consumo de TV, enquanto desenvolve abordagens específicas para atrair jovens.
                            """)
                        else:
                            st.warning(f"**Não há diferença significativa no impacto da viralização entre públicos jovens e mais velhos (r = {corr_jovem:.2f} vs. {corr_nao_jovem:.2f})**")
                            st.markdown(f"""
                            **Conclusão sobre impacto nas diferentes gerações:**
                            
                            A análise mostra que não há diferença significativa no impacto da viralização
                            entre horários com público jovem ({corr_jovem:.2f}) e outros horários ({corr_nao_jovem:.2f}).
                            Isso sugere que a idade pode não ser o fator determinante na resposta à viralização.
                            
                            **Implicações:**
                            - Outros fatores além da idade podem ser mais importantes (como interesse no conteúdo)
                            - A conversão de engajamento social para TV pode seguir padrões similares independente da faixa etária
                            - A eficácia da viralização pode depender mais da qualidade e relevância do conteúdo do que do público-alvo
                            
                            **Recomendação:** Focar em estratégias de viralização baseadas na qualidade e relevância do conteúdo,
                            com mensagens que atraiam espectadores de todas as idades, em vez de segmentar especificamente por geração.
                            """)
                    else:
                        st.warning("Dados insuficientes para análise de correlação por faixa horária.")
                else:
                    st.warning("Métricas de viralização para Instagram e TikTok não estão disponíveis para análise.")
            else:
                st.warning("Dados insuficientes para análise de públicos jovens.")
        else:
            st.warning("Não foi possível criar a métrica de viralização total. Verificar disponibilidade de dados.")
    else:
        st.warning("Dados insuficientes para realizar a análise. É necessário ter métricas de compartilhamento em redes sociais.")