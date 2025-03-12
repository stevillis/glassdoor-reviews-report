import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import (
    get_bad_rating_companies,
    get_good_rating_companies,
    get_neutral_rating_companies,
    load_reviews_df,
    set_companies_raking_to_session,
)


def general_reviews_ranking():
    st.markdown(
        """
    Este gráfico mostra a quantidade de avaliações e o sentimento
    associado para cada empresa, ordenadas pela diferença entre avaliações
    positivas e negativas. Essa visualização das predições do modelo mostra
    que:

    - 18 das 22 empresas analisadas têm mais avaliações positivas do que
    negativas.
    - 2 empresas apresentam um número maior de avaliações negativas.
    - 2 empresas têm um número igual de avaliações positivas e negativas.
"""
    )

    reviews_df = load_reviews_df()

    reviews_df["company"] = reviews_df["company"].apply(
        lambda x: (
            x[: ReportConfig.COMPANY_NAME_MAX_LENGTH] + ""
            if len(x) > ReportConfig.COMPANY_NAME_MAX_LENGTH
            else x
        )
    )

    sentiment_plot_by_company_df = (
        reviews_df.groupby(["company", "sentiment_plot"]).size().unstack(fill_value=0)
    )

    sentiment_plot_by_company_df_reset = sentiment_plot_by_company_df.reset_index()
    sentiment_plot_by_company_df_reset.columns = [
        "company",
        "1positive",
        "2negative",
        "3neutral",
    ]

    sentiment_plot_by_company_df_reset["sentiment_diff"] = (
        sentiment_plot_by_company_df_reset["1positive"]
        - sentiment_plot_by_company_df_reset["2negative"]
    )

    bad_rating_companies = get_bad_rating_companies(sentiment_plot_by_company_df_reset)
    good_rating_companies = get_good_rating_companies(
        sentiment_plot_by_company_df_reset
    )
    neutral_rating_companies = get_neutral_rating_companies(
        sentiment_plot_by_company_df_reset
    )

    fig, ax = plt.subplots(1, figsize=(8, 10))

    # Plot

    sorted_companies_df = sentiment_plot_by_company_df_reset.sort_values(
        by="sentiment_diff", ascending=False
    )["company"]

    sns.countplot(
        data=reviews_df,
        y="company",
        hue="sentiment_plot",
        order=sorted_companies_df,
        ax=ax,
        palette=[
            ReportConfig.POSITIVE_SENTIMENT_COLOR,
            ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            ReportConfig.NEUTRAL_SENTIMENT_COLOR,
        ],
        width=0.9,
    )

    # Highlight Companies
    for i, label in enumerate(ax.get_yticklabels()):
        company_name = label.get_text()
        if company_name in bad_rating_companies:
            label.set_color(ReportConfig.NEGATIVE_SENTIMENT_COLOR)

    for i, label in enumerate(ax.get_yticklabels()):
        company_name = label.get_text()
        if company_name in good_rating_companies:
            label.set_color(ReportConfig.POSITIVE_SENTIMENT_COLOR)

    for i, label in enumerate(ax.get_yticklabels()):
        company_name = label.get_text()
        if company_name in neutral_rating_companies:
            label.set_color(ReportConfig.NEUTRAL_SENTIMENT_COLOR)

    # Plot Annotates
    for p in ax.patches:
        ax.annotate(
            text=f"{p.get_width():.0f}",
            xy=(p.get_width(), (p.get_y() + p.get_height() / 2)),
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            xytext=(10, -1),
            textcoords="offset points",
        )

    sns.despine(bottom=True)

    ax.set_xlabel("")
    ax.set_xticks([])

    ax.set_ylabel("")

    ax.set_title(
        "Ranking geral de avaliações",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.1,
    )

    positive_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=ReportConfig.POSITIVE_SENTIMENT_COLOR
    )
    negative_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=ReportConfig.NEGATIVE_SENTIMENT_COLOR
    )
    neutral_patch = plt.Rectangle((0, 0), 1, 1, fc=ReportConfig.NEUTRAL_SENTIMENT_COLOR)

    ax.legend(
        # title="Sentimento",
        handles=[positive_patch, negative_patch, neutral_patch],
        labels=["Positivo", "Negativo", "Neutro"],
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        edgecolor="1",
        ncols=3,
    )

    st.pyplot(fig)

    # plt.savefig(
    #     "general_sentiment_reviews_rank.png",
    #     transparent=False,
    #     dpi=300,
    #     bbox_inches="tight",
    # )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Ranking geral de avaliações",
        page_icon="	:first_place_medal:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.header(
        "Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá"
    )

    st.subheader("Ranking geral de avaliações")

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(reviews_df)

    general_reviews_ranking()
