from string import punctuation

import matplotlib.pyplot as plt
import nltk
import pandas as pd
import streamlit as st
from wordcloud import WordCloud

from report_config import ReportConfig

if "download_stopwords" not in st.session_state:
    nltk.download("stopwords")
    st.session_state["download_stopwords"] = 1

    portuguese_stop_words = nltk.corpus.stopwords.words("portuguese")

CUSTOM_CSS = """
<style>
    [data-testid="stMarkdown"] {
        text-align: justify;
    }
</style>
"""


def _print_wordcloud(corpus, title=None, max_words: int = 150):
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


def _positive_wordcloud():
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

    _print_wordcloud(corpus, max_words=max_words)


def _negative_wordcloud():
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

    _print_wordcloud(corpus, max_words=max_words)


def _neutral_wordcloud():
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

    _print_wordcloud(corpus, max_words=max_words)


if __name__ == "__main__":
    st.set_page_config(
        page_title="WordClouds",
        page_icon=":cloud:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.title(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
    )

    st.header("WordClouds")

    st.markdown(
        """
        As WordClouds destacam as palavras mais frequentemente associadas a cada tipo de sentimento nas avaliações.
"""
    )

    if "reviews_df" not in st.session_state:
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")
        st.session_state["reviews_df"] = reviews_df

        neutral_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 0]
        st.session_state["neutral_reviews_df"] = neutral_reviews_df

        positive_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 1]
        st.session_state["positive_reviews_df"] = positive_reviews_df

        negative_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 2]
        st.session_state["negative_reviews_df"] = negative_reviews_df

        nltk.download("stopwords")

    _positive_wordcloud()
    _negative_wordcloud()
    _neutral_wordcloud()
