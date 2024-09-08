from string import punctuation

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from report_config import ReportConfig
from utils import get_ranking_positive_negative_companies


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
    st.subheader("Avaliações Positivas")

    positive_reviews_df = st.session_state.get("positive_reviews_df")
    review_text = positive_reviews_df["review_text"].str.split().values.tolist()
    corpus = [word for i in review_text for word in i]

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=150,
        value=100,
        key="slider_max_words_positive",
    )

    print_wordcloud(corpus, max_words=max_words)


def negative_wordcloud():
    st.subheader("Avaliações Negativas")

    negative_reviews_df = st.session_state.get("negative_reviews_df")
    review_text = negative_reviews_df["review_text"].str.split().values.tolist()
    corpus = [word for i in review_text for word in i]

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=150,
        value=100,
        key="slider_max_words_negative",
    )

    print_wordcloud(corpus, max_words=max_words)


def neutral_wordcloud():
    st.subheader("Avaliações Neutras")

    neutral_reviews_df = st.session_state.get("neutral_reviews_df")
    review_text = neutral_reviews_df["review_text"].str.split().values.tolist()
    corpus = [word for i in review_text for word in i]

    max_words = st.slider(
        label="Quantidade de palavras",
        min_value=10,
        max_value=150,
        value=100,
        key="slider_max_words_neutral",
    )

    print_wordcloud(corpus, max_words=max_words)


if __name__ == "__main__":
    st.set_page_config(
        page_title="WordClouds",
        page_icon=":cloud:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        """Desvendando emoções nas avaliações do Glassdoor de empresas de
        Tecnologia de Cuiabá"""
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
    negative_wordcloud()
    neutral_wordcloud()

    # TODO: create wordclouds by company
