import warnings
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig
from utils import (
    STOPWORDS,
    TRANSLATION_TABLE_SPECIAL_CHARACTERS,
    get_ranking_positive_negative_companies,
    get_sentiment_key_from_value,
)


def top_10_most_common_words_analysis():
    st.subheader("Top 10 palavras mais usadas por empresa")

    reviews_df = st.session_state.get("reviews_df")

    col1, col2 = st.columns(2)
    with col1:
        company_options = ["Todas"] + sorted(reviews_df["company"].unique().tolist())
        company = st.selectbox(
            label="Empresa",
            options=company_options,
            key="wordcloud_company_selectbox",
            index=0,
        )

    filtered_df = reviews_df[(reviews_df["company"] == company) | (company == "Todas")]

    with col2:
        sentiment = st.selectbox(
            label="Sentimento das Avaliações",
            options=(
                "Todos",
                "Positivo",
                "Negativo",
                "Neutro",
            ),
            key="sentiment_selectbox",
        )

    if sentiment != "Todos":
        sentiment_key = get_sentiment_key_from_value(sentiment)
        filtered_df = filtered_df[filtered_df["predicted_sentiment"] == sentiment_key]

    review_text = filtered_df["review_text"].str.split().values.tolist()
    corpus = [word for i in review_text for word in i]

    non_stopwords_corpus = []
    for word in corpus:
        word_lower = word.lower()
        cleaned_word = word_lower.translate(TRANSLATION_TABLE_SPECIAL_CHARACTERS)
        if cleaned_word and cleaned_word not in STOPWORDS:
            non_stopwords_corpus.append(cleaned_word)

    counter = Counter(non_stopwords_corpus)
    most_common_words = counter.most_common(n=10)

    words, counts = zip(*most_common_words)  # Unzip the words and counts
    most_common_words_df = pd.DataFrame({"words": words, "counts": counts})

    fig, ax = plt.subplots(1, figsize=(10, 8))

    sns.barplot(
        data=most_common_words_df,
        x="counts",
        y="words",
        ax=ax,
        width=0.9,
        orient="h",
    )

    # Annotates
    for p in ax.patches:
        ax.annotate(
            text=f"{p.get_width():.0f}",
            xy=(p.get_width(), (p.get_y() + p.get_height() / 2)),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            xytext=(-15, 0),
            textcoords="offset points",
        )

    # Axes config
    ax.set_xlabel("")

    ax.set_xticks([])

    ax.set_ylabel("")

    ax.set_title(
        "Top 10 palavras mais usadas por empresa",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    st.pyplot(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "use_inf_as_na")

    st.set_page_config(
        page_title="Top 10 palavras mais usadas por empresa",
        page_icon=":bar_chart:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        "Análise de sentimentos nas avaliações do Glassdoor: Um estudo sobre empresas de Tecnologia em Cuiabá"
    )

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

    top_10_most_common_words_analysis()
