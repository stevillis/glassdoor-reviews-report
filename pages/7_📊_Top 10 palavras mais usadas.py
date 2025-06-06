import warnings
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import (
    STOPWORDS,
    TRANSLATION_TABLE_SPECIAL_CHARACTERS,
    get_sentiment_key_from_value,
    load_reviews_df,
    set_companies_raking_to_session,
)


def top_10_most_common_words_analysis():
    st.subheader("Top 10 palavras mais frequentes por empresa")

    st.write(
        """
    Essa visualização permite uma análise interativa das palavras mais
    frequentes encontradas nas avaliações de forma quantitativa. Ela permite
    que você personalize a análise filtrando por empresa e tipo de sentimento.
"""
    )

    reviews_df = load_reviews_df()

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
            label="Sentimento",
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
        filtered_df = filtered_df[filtered_df["sentiment"] == sentiment_key]

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
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    st.pyplot(fig)
    plt.close(fig)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "use_inf_as_na")

    st.set_page_config(
        page_title="Top 10 palavras mais frequentes por empresa",
        page_icon=":bar_chart:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.header(
        "Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá"
    )

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(reviews_df)

    top_10_most_common_words_analysis()
