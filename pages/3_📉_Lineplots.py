import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig

if __name__ == "__main__":
    st.set_page_config(
        page_title="Lineplots",
        page_icon=":chart_with_downwards_trend:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
    )

    st.subheader("Sentimento de avaliações por empresa ao longo do tempo")

    st.markdown(
        """
    O gráfico apresenta uma análise de sentimentos das avaliações entre 05 de outubro de 2014 e 16 de março de 2024.
    Além disso, permite a filtragem das avaliações por empresa, proporcionando uma visão mais clara e específica sobre
    a percepção dos funcionários em relação a cada organização.
"""
    )

    if "reviews_df" not in st.session_state:
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")
        reviews_df["sentiment_label"] = reviews_df["predicted_sentiment"].map(
            ReportConfig.SENTIMENT_DICT
        )
        st.session_state["reviews_df"] = reviews_df
    else:
        reviews_df = st.session_state.get("reviews_df")

    company = st.selectbox(
        label="Empresa",
        options=reviews_df["company"].unique().tolist(),
        key="reviews_along_time_company_input",
    )

    filtered_df = reviews_df[reviews_df["company"] == company][
        [
            "company",
            "review_date",
            "predicted_sentiment",
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
        filtered_df.groupby(["year", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(12, 6))
    sns.lineplot(
        data=sentiment_counts,
        x="year",
        y="count",
        hue="predicted_sentiment",
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
        "Sentimento de avaliações por empresa ao longo do tempo",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.1,
    )

    ax.set_xlabel("Ano")
    ax.set_ylabel("Número de Avaliações")

    ax.set_xticks([x for x in range(min_year, max_year + 1)])

    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(ReportConfig.SENTIMENT_DICT)):
        handles[i]._label = ReportConfig.SENTIMENT_DICT[int(labels[i])]

    ax.legend(
        # title="Sentimento",
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        edgecolor="1",
        ncols=3,
    )

    st.pyplot(fig)
