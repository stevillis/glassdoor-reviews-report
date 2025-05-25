from collections import Counter

import matplotlib.pyplot as plt
import streamlit as st
from wordcloud import WordCloud

from app_messages import AppMessages
from report_config import ReportConfig
from utils import (
    STOPWORDS,
    TRANSLATION_TABLE_SPECIAL_CHARACTERS,
    get_sentiment_key_from_value,
    load_reviews_df,
    set_companies_raking_to_session,
)


def print_wordcloud(corpus, title=None, max_words: int = 150):
    non_stopwords_corpus = []
    for word in corpus:
        word_lower = word.lower()
        cleaned_word = word_lower.translate(TRANSLATION_TABLE_SPECIAL_CHARACTERS)
        if cleaned_word and cleaned_word not in STOPWORDS:
            non_stopwords_corpus.append(cleaned_word)

    counter = Counter(non_stopwords_corpus)
    most_common_words = counter.most_common(n=max_words)

    wordcloud = WordCloud(
        background_color="white",
        random_state=ReportConfig.RANDOM_SEED,
        # max_words=max_words,
        width=1024,
        height=768,
    )

    fig = plt.figure(1, figsize=(10, 6))
    plt.axis("off")

    plt.imshow(wordcloud.generate_from_frequencies(dict(most_common_words)))

    plt.title(
        title,
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
    )

    st.pyplot(fig)


def wordcloud_by_company():
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
        page_title="Nuvem de Palavras por empresa",
        page_icon=":cloud:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.header(
        """
        Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá
    """
    )

    st.subheader("Nuvem de Palavras por empresa")

    st.markdown(
        """
        Essa visualização permite uma análise interativa das palavras mais
        frequentes encontradas nas avaliações. Ela permite que você
        personalize a análise filtrando por empresa, tipo de sentimento e
        ajustando a quantidade de palavras mais frequentes que deseja
        visualizar.
"""
    )

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(reviews_df)

    wordcloud_by_company()
