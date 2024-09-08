import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from app_messages import AppMessages
from report_config import ReportConfig
from utils import get_sentiment_key_from_value


def plot_top_ngrams_barchart(review_text, n_grams=2, top=10, title=None):
    vec = CountVectorizer(ngram_range=(n_grams, n_grams)).fit(review_text)
    bag_of_words = vec.transform(review_text)
    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

    top_n_grams = words_freq[:top]
    x, y = map(list, zip(*top_n_grams))

    fig, ax = plt.subplots(1, figsize=(10, 6))

    sns.barplot(x=y, y=x, ax=ax)

    plt.title(
        title,
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
    )

    st.pyplot(fig)


def positive_ngrams():
    st.subheader("Avaliações Positivas")

    col1, _, _ = st.columns(3)

    with col1:
        positive_n_gram_input = st.number_input(
            label="Quantidade de palavras do N-Gram",
            step=1,
            value=3,
            min_value=2,
            max_value=5,
            key="positive_n_gram_input",
        )

    positive_reviews_df = st.session_state.get("positive_reviews_df")
    plot_top_ngrams_barchart(
        positive_reviews_df["review_text"],
        n_grams=positive_n_gram_input,
        top=10,
    )


def negative_ngrams():
    st.subheader("Avaliações Negativas")

    col1, _, _ = st.columns(3)

    with col1:
        negative_n_gram_input = st.number_input(
            label="Quantidade de palavras do N-Gram",
            step=1,
            value=3,
            min_value=2,
            max_value=5,
            key="negative_n_gram_input",
        )

    negative_reviews_df = st.session_state.get("negative_reviews_df")
    plot_top_ngrams_barchart(
        negative_reviews_df["review_text"],
        n_grams=negative_n_gram_input,
        top=10,
    )


def neutral_ngrams():
    st.subheader("Avaliações Neutras")

    col1, _, _ = st.columns(3)

    with col1:
        neutral_n_gram_input = st.number_input(
            label="Quantidade de palavras do N-Gram",
            step=1,
            value=3,
            min_value=2,
            max_value=5,
            key="neutral_n_gram_input",
        )

    neutral_reviews_df = st.session_state.get("neutral_reviews_df")
    plot_top_ngrams_barchart(
        neutral_reviews_df["review_text"],
        n_grams=neutral_n_gram_input,
        top=10,
    )


def n_gram_by_company():
    st.subheader("N-Gram por Empresa, Cargo e Sentimento de Avaliações")

    st.markdown(
        """
        Ao analisar os N-Grams em conjunto com informações sobre a empresa, cargo e sentimento expresso nas avaliações, é possível identificar padrões específicos e insights valiosos.
        Isso pode ajudar as empresas a entenderem melhor as preocupações e as percepções dos funcionários em diferentes níveis organizacionais e a direcionarem estratégias de melhoria de forma mais precisa e eficaz.
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    col1, col2 = st.columns(2)
    with col1:
        company = st.selectbox(
            label="Empresa",
            options=reviews_df["company"].unique().tolist(),
            key="company_input",
        )

        filtered_df = reviews_df[reviews_df["company"] == company]

    with col2:
        employee_role = st.selectbox(
            label="Cargo",
            options=filtered_df["employee_role"].unique().tolist(),
            key="employee_role_input",
        )

        filtered_df = filtered_df[filtered_df["employee_role"] == employee_role]

    col3, col4 = st.columns(2)
    with col3:
        sentiment = st.selectbox(
            label="Sentimento das Avaliações",
            options=("Positivo", "Negativo", "Neutro"),
            key="sentiment_input",
        )

        sentiment_key = get_sentiment_key_from_value(sentiment)
        filtered_df = filtered_df[filtered_df["predicted_sentiment"] == sentiment_key]

    with col4:
        n_gram_input = st.number_input(
            label="Quantidade de palavras",
            step=1,
            value=3,
            min_value=2,
            max_value=5,
            key="n_gram_input",
        )

    filtered_df = filtered_df[
        [
            "employee_role",
            "employee_detail",
            "review_text",
            "review_date",
            "star_rating",
            "sentiment_label",
        ]
    ]
    filtered_df.reset_index(drop=True, inplace=True)

    if len(filtered_df) > 0:
        plot_top_ngrams_barchart(
            filtered_df["review_text"],
            n_grams=n_gram_input,
            top=10,
        )

        st.write("Avaliações filtradas")
        st.dataframe(filtered_df)
    else:
        st.error(
            AppMessages.ERROR_EMPTY_DATAFRAME,
            icon="🚨",
        )


if __name__ == "__main__":
    st.set_page_config(
        page_title="N-Grams",
        page_icon=":capital_abcd:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
    )

    st.subheader("Explorando as características comuns entre avaliações com N-Grams")

    st.markdown(
        """
        Ao utilizar N-Grams, que são combinações de palavras que ocorrem juntas, é possível explorar as nuances das avaliações.
        Para uma análise mais detalhada, pode-se ajustar a quantidade de palavras em cada N-Gram.
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

    positive_ngrams()
    negative_ngrams()
    neutral_ngrams()

    n_gram_by_company()
