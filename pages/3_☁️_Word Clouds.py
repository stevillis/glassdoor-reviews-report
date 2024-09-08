from string import punctuation

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from report_config import ReportConfig
from utils import get_ranking_positive_negative_companies, get_sentiment_key_from_value


def print_wordcloud(corpus, title=None, max_words: int = 150):
    portuguese_stop_words = nltk.corpus.stopwords.words("portuguese")

    non_stopwords_corpus = []
    for word in corpus:
        word_lower = word.lower()
        if word_lower not in portuguese_stop_words and word_lower not in punctuation:
            non_stopwords_corpus.append(word_lower)

    non_stopwords_corpus_str = " ".join(non_stopwords_corpus)

    wordcloud = WordCloud(
        background_color="white",
        random_state=ReportConfig.RANDOM_SEED,
        max_words=max_words,
        width=1024,
        height=768,
    )

    fig = plt.figure(1, figsize=(10, 6))
    plt.axis("off")

    plt.imshow(wordcloud.generate(str(non_stopwords_corpus_str)))
    plt.title(
        title,
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
    )

    st.pyplot(fig)


def positive_wordcloud():
    st.subheader("Word Cloud de avaliações positivas")

    positive_reviews_df = st.session_state.get("positive_reviews_df")

    if "positive_corpus" not in st.session_state:
        review_text = positive_reviews_df["review_text"].str.split().values.tolist()
        positive_corpus = [word for i in review_text for word in i]

        st.session_state["positive_corpus"] = positive_corpus
    else:
        positive_corpus = st.session_state.get("positive_corpus")

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=50,
        value=10,
        key="max_words_positive_slider",
    )

    print_wordcloud(corpus=positive_corpus, max_words=max_words)


def negative_wordcloud():
    st.subheader("Word Cloud de avaliações negativas")

    negative_reviews_df = st.session_state.get("negative_reviews_df")
    if "negative_corpus" not in st.session_state:
        review_text = negative_reviews_df["review_text"].str.split().values.tolist()
        negative_corpus = [word for i in review_text for word in i]

        st.session_state["negative_corpus"] = negative_corpus
    else:
        negative_corpus = st.session_state.get("negative_corpus")

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=50,
        value=10,
        key="max_words_negative_slider",
    )

    print_wordcloud(corpus=negative_corpus, max_words=max_words)


def neutral_wordcloud():
    st.subheader("Word Cloud de avaliações neutras")

    neutral_reviews_df = st.session_state.get("neutral_reviews_df")

    if "neutral_corpus" not in st.session_state:
        review_text = neutral_reviews_df["review_text"].str.split().values.tolist()
        neutral_corpus = [word for i in review_text for word in i]

        st.session_state["neutral_corpus"] = neutral_corpus
    else:
        neutral_corpus = st.session_state.get("neutral_corpus")

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=50,
        value=10,
        key="max_words_neutral_slider",
    )

    print_wordcloud(corpus=neutral_corpus, max_words=max_words)


def wordcloud_by_company():
    st.subheader("Avaliações por empresa")

    reviews_df = st.session_state.get("reviews_df")

    col1, col2 = st.columns(2)

    with col1:
        company = st.selectbox(
            label="Empresa",
            options=reviews_df["company"].unique().tolist(),
            key="wordcloud_company_selectbox",
        )

    filtered_df = reviews_df[reviews_df["company"] == company]

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

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=50,
        value=10,
        key="max_words_by_company_slider",
    )

    print_wordcloud(corpus, max_words=max_words)


if __name__ == "__main__":
    st.set_page_config(
        page_title="Word Clouds",
        page_icon=":cloud:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        """Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"""
    )

    st.subheader("WordClouds")

    st.markdown(
        """
        As WordClouds destacam as palavras mais frequentemente associadas a
        cada tipo de sentimento nas avaliações.
"""
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

        # Neutral Reviews DF
        neutral_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 0]
        st.session_state["neutral_reviews_df"] = neutral_reviews_df

        # Positive Reviews DF
        positive_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 1]
        st.session_state["positive_reviews_df"] = positive_reviews_df

        # Negative Reviews DF
        negative_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 2]
        st.session_state["negative_reviews_df"] = negative_reviews_df

        nltk.download("stopwords")

    positive_wordcloud()
    st.markdown("---")
    negative_wordcloud()
    st.markdown("---")

    neutral_wordcloud()
    st.markdown("---")

    wordcloud_by_company()
