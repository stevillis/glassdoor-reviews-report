import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer

from app_messages import AppMessages
from report_config import ReportConfig
from utils import get_ranking_positive_negative_companies, get_sentiment_key_from_value


def n_gram_by_company():
    reviews_df = st.session_state.get("reviews_df")

    col1, col2, col3 = st.columns(3)
    with col1:
        company_options = ["Todas"] + sorted(reviews_df["company"].unique().tolist())
        company = st.selectbox(
            label="Empresa",
            options=company_options,
            key="company_input",
            index=0,
        )

        filtered_df = reviews_df[
            (reviews_df["company"] == company) | (company == "Todas")
        ]

    with col2:
        sentiment = st.selectbox(
            label="Sentimento das Avaliações",
            options=("Todos", "Positivo", "Negativo", "Neutro"),
            key="sentiment_input",
        )

        if sentiment != "Todos":
            sentiment_key = get_sentiment_key_from_value(sentiment)
            filtered_df = filtered_df[
                filtered_df["predicted_sentiment"] == sentiment_key
            ]

    with col3:
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
        review_text = filtered_df["review_text"]

        vec = CountVectorizer(ngram_range=(n_gram_input, n_gram_input)).fit(review_text)
        bag_of_words = vec.transform(review_text)
        sum_words = bag_of_words.sum(axis=0)

        words_freq = [
            (word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()
        ]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)

        top_n_grams = words_freq[:10]
        x, y = map(list, zip(*top_n_grams))

        fig, ax = plt.subplots(1, figsize=(10, 8))

        sns.barplot(
            x=y,
            y=x,
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
            "Top 10 NGrams por empresa",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
            y=1.0,
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

        st.pyplot(fig)

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
        "Análise de sentimentos nas avaliações do Glassdoor: Um estudo sobre empresas de Tecnologia em Cuiabá"
    )

    st.subheader("Top 10 NGrams por empresa")

    st.markdown(
        """
    A análise de NGrams nas avaliações positivas revela os tópicos mais
    recorrentes, que incluem:
    - Ambiente de trabalho
    - Plano de saúde
    - Oportunidade de crescimento

    Por outro lado, os NGrams nas avaliações negativas destacam os seguintes
    temas:
    - Plano de carreira
    - Plano de saúde
    - Salário abaixo do mercado

    As análises de NGrams nas avaliações neutras indicam que os avaliadores
    não conseguiram identificar aspectos negativos a serem destacados na seção
    "Contras" do Glassdoor, mesmo quando obrigados a fornecer uma resposta.
    Isso sugere que, para muitos colaboradores, a experiência de trabalho é
    suficientemente positiva, resultando em uma falta de críticas
    significativas.
"""
    )

    if "reviews_df" not in st.session_state:
        # Reviews DF
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")
        st.session_state["reviews_df"] = reviews_df

        # Top Companies Reviews DF
        if "top_positive_companies_df" not in st.session_state:
            top_positive_companies_df, top_negative_companies_df = (
                get_ranking_positive_negative_companies(reviews_df)
            )

            st.session_state["top_positive_companies_df"] = top_positive_companies_df
            st.session_state["top_negative_companies_df"] = top_negative_companies_df

    n_gram_by_company()
