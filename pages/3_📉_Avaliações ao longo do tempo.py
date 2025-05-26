import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import load_reviews_df, set_companies_raking_to_session

if __name__ == "__main__":
    st.set_page_config(
        page_title="Avaliações ao longo do tempo",
        page_icon=":chart_with_downwards_trend:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.header(
        "Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá"
    )

    st.subheader("Avaliações ao longo do tempo")

    st.markdown(
        """
    Esse gráfico apresenta uma análise de sentimentos das **avaliações entre 05
    de outubro de 2014 e 16 de março de 2024**, agrupadas por ano. Ao filtrar
    as avaliações por empresa, pode-se ter uma visão do comportamento das
    avaliações ao longo do tempo. Isso é importante para constatar se empresas
    anteriormente mal avaliadas adotaram alguma estratégia para melhorar suas
    avaliações.
"""
    )

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(reviews_df)

    company_options = ["Todas"] + sorted(reviews_df["company"].unique().tolist())
    company = st.selectbox(
        label="Empresa",
        options=company_options,
        key="reviews_along_time_company_input",
        index=0,
    )

    filtered_df = reviews_df[(reviews_df["company"] == company) | (company == "Todas")][
        [
            "company",
            "review_date",
            "sentiment",
        ]
    ]

    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df["review_date"] = pd.to_datetime(
        filtered_df["review_date"], format="%Y-%m-%d"
    )
    filtered_df["year"] = filtered_df["review_date"].dt.year

    min_year = filtered_df["review_date"].min().year
    max_year = filtered_df["review_date"].max().year

    sentiment_counts = (
        filtered_df.groupby(["year", "sentiment"]).size().reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(12, 6))
    sns.lineplot(
        data=sentiment_counts,
        x="year",
        y="count",
        hue="sentiment",
        palette=ReportConfig.SENTIMENT_PALETTE,
        ax=ax,
    )

    # Annotations for number of reviews per sentiment
    years_unique = sentiment_counts["year"].unique()
    for year in years_unique:
        year_counts = sentiment_counts[sentiment_counts["year"] == year][
            "count"
        ].tolist()

        neutral_counts, positive_counts, negative_counts = (year_counts + [None] * 3)[
            :3
        ]

        if neutral_counts:
            ax.text(
                x=year,
                y=neutral_counts,
                s=f"{neutral_counts}",
                ha="center",
                color=ReportConfig.NEUTRAL_SENTIMENT_COLOR,
            )

        if positive_counts:
            ax.text(
                x=year,
                y=positive_counts,
                s=f"{positive_counts}",
                ha="center",
                color=ReportConfig.POSITIVE_SENTIMENT_COLOR,
            )

        if negative_counts:
            ax.text(
                x=year,
                y=negative_counts,
                s=f"{negative_counts}",
                ha="center",
                color=ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    ax.set_title(
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.1,
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("Quantidade")

    ax.set_xticks([x for x in range(min_year, max_year + 1)])

    handles, labels = ax.get_legend_handles_labels()
    order_map = {label: handle for handle, label in zip(handles, labels)}
    handles = [order_map[sentiment] for sentiment in ReportConfig.PLOT_SENTIMENT_VALUES]

    plt.legend(
        # title="Sentimento",
        handles=handles,
        labels=ReportConfig.PLOT_SENTIMENT_LABELS,
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        edgecolor="1",
        ncols=3,
    )

    st.pyplot(fig)
    plt.close(fig)
