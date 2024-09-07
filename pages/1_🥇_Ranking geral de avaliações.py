import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig
from utils import (
    create_predicted_sentiment_plot,
    get_bad_rating_companies,
    get_good_rating_companies,
    get_neutral_rating_companies,
    get_ranking_positive_negative_companies,
)


def general_reviews_ranking():
    st.markdown(
        """Este gráfico mostra a quantidade de avaliações e o sentimento associado para cada empresa, ordenadas pela totalidade de avaliações.

**Metodologia**

As avaliações neutras não foram consideradas na relação entre avaliações positivas e negativas.

**Resultados**

- 15 das 22 empresas analisadas têm mais avaliações positivas do que negativas.
- 4 empresas apresentam um número maior de avaliações negativas.
- 3 empresas têm um número igual de avaliações positivas e negativas.

"""
    )

    reviews_df = st.session_state.get("reviews_df")

    reviews_df = create_predicted_sentiment_plot(reviews_df)

    predicted_sentiment_plot_by_company_df = (
        reviews_df.groupby(["company", "predicted_sentiment_plot"])
        .size()
        .unstack(fill_value=0)
    )

    predicted_sentiment_plot_by_company_df_reset = (
        predicted_sentiment_plot_by_company_df.reset_index()
    )
    predicted_sentiment_plot_by_company_df_reset.columns = [
        "company",
        "1positive",
        "2negative",
        "3neutral",
    ]

    bad_rating_companies = get_bad_rating_companies(
        predicted_sentiment_plot_by_company_df_reset
    )
    good_rating_companies = get_good_rating_companies(
        predicted_sentiment_plot_by_company_df_reset
    )
    neutral_rating_companies = get_neutral_rating_companies(
        predicted_sentiment_plot_by_company_df_reset
    )

    fig, ax = plt.subplots(1, figsize=(10, 8))

    # Plot
    sns.countplot(
        data=reviews_df,
        y="company",
        hue="predicted_sentiment_plot",
        order=reviews_df["company"].value_counts().index,
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
            xy=(p.get_width() + 10, (p.get_y() + p.get_height() / 2) + 0.02),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            xytext=(0, 0),
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


if __name__ == "__main__":
    st.set_page_config(
        page_title="Ranking geral de avaliações",
        page_icon="	:first_place_medal:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
    )

    st.subheader("Ranking geral de avaliações")

    if "reviews_df" not in st.session_state:
        # Reviews DF
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")

        reviews_df["sentiment"] = reviews_df["sentiment"].apply(
            lambda x: 2 if x == -1 else x
        )

        reviews_df["sentiment_label"] = reviews_df["predicted_sentiment"].map(
            ReportConfig.SENTIMENT_DICT
        )

        reviews_df["company"] = reviews_df["company"].apply(
            lambda x: (
                x[: ReportConfig.COMPANY_NAME_MAX_LENGTH] + ""
                if len(x) > ReportConfig.COMPANY_NAME_MAX_LENGTH
                else x
            )
        )

        st.session_state["reviews_df"] = reviews_df

        # Top Companies Reviews DF
        top_positive_companies_df, top_negative_companies_df = (
            get_ranking_positive_negative_companies(reviews_df)
        )

        st.session_state["top_positive_companies_df"] = top_positive_companies_df
        st.session_state["top_negative_companies_df"] = top_negative_companies_df

        # Neutral Reviews DF
        neutral_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 0]
        st.session_state["neutral_reviews_df"] = neutral_reviews_df

        # Positive Reviews DF
        positive_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 1]
        st.session_state["positive_reviews_df"] = positive_reviews_df

        # Negative Reviews DF
        negative_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 2]
        st.session_state["negative_reviews_df"] = negative_reviews_df

    general_reviews_ranking()
