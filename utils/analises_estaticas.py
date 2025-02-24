# analises_estaticas.py
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

def analise_redes_sociais(df):
    """
    Realiza análises estáticas de redes sociais utilizando o dataframe tratado,
    onde as colunas seguem o padrão 'PLATAFORMA_metrica'.
    """
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
    else:
        st.error(f"**Vocês têm mais engajamento em {best_platform} e investem mais em {most_posts_platform}.**")

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
