import math
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from PIL import Image
from wordcloud import WordCloud

from app_messages import AppMessages
from report_config import ReportConfig
from utils import (
    ROLE_GROUPS,
    STOPWORDS,
    TRANSLATION_TABLE_SPECIAL_CHARACTERS,
    get_top_ngrams,
    load_reviews_df,
    set_companies_raking_to_session,
)


def calculate_group_metrics(df):
    total = len(df)
    pos = (df["sentiment"] == 1).sum() / total * 100 if total else 0
    neg = (df["sentiment"] == 2).sum() / total * 100 if total else 0
    neu = (df["sentiment"] == 0).sum() / total * 100 if total else 0
    return total, pos, neg, neu


def render_group_metrics(title, total, pos, neg, neu):
    st.caption(f"{title}")
    cols = st.columns(4)
    cols[0].metric("Total", total)

    cols[1].metric("% Positivas", f"{pos:.1f}%")
    cols[2].metric("% Negativas", f"{neg:.1f}%")
    cols[3].metric("% Neutras", f"{neu:.1f}%")


def metrics():
    reviews_df = load_reviews_df()

    st.markdown("###### Avaliações Gerais")
    total, pos, neg, neu = calculate_group_metrics(reviews_df)
    render_group_metrics("", total, pos, neg, neu)

    st.markdown("###### Avaliações por Área de Atuação")
    it_df = reviews_df[reviews_df["role_group"] == 1]
    total_it, it_pos, it_neg, it_neu = calculate_group_metrics(it_df)
    render_group_metrics("Profissionais de TI", total_it, it_pos, it_neg, it_neu)

    confidencial_df = reviews_df[reviews_df["role_group"] == 2]
    total_confidencial, confidencial_pos, confidencial_neg, confidencial_neu = (
        calculate_group_metrics(confidencial_df)
    )
    render_group_metrics(
        "Funcionário confidencial",
        total_confidencial,
        confidencial_pos,
        confidencial_neg,
        confidencial_neu,
    )

    other_df = reviews_df[reviews_df["role_group"] == 0]
    total_other, other_pos, other_neg, other_neu = calculate_group_metrics(other_df)
    render_group_metrics("Outros", total_other, other_pos, other_neg, other_neu)


@st.cache_data
def positive_reviews_ranking():
    top_positive_companies_df = st.session_state.get("top_positive_companies_df")

    fig, ax = plt.subplots(1, figsize=(5, 7))

    sns.barplot(
        data=top_positive_companies_df,
        x="sentiment_count",
        y="company",
        hue="sentiment_plot",
        palette=[
            ReportConfig.POSITIVE_SENTIMENT_COLOR,
            ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            ReportConfig.NEUTRAL_SENTIMENT_COLOR,
        ],
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
            fontsize=10,
            color="black",
            xytext=(10, 0),
            textcoords="offset points",
        )

    # Axes config
    ax.set_xlabel("")

    ax.set_xticks([])

    ax.set_ylabel("")

    ax.set_title(
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

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
        labels=ReportConfig.PLOT_SENTIMENT_LABELS,
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        edgecolor="1",
        ncols=3,
    )

    return fig


@st.cache_data
def negative_reviews_ranking():
    top_negative_companies_df = st.session_state.get("top_negative_companies_df")

    fig, ax = plt.subplots(1, figsize=(6, 5))

    sns.barplot(
        data=top_negative_companies_df,
        x="sentiment_count",
        y="company",
        hue="sentiment_plot",
        palette=[
            ReportConfig.POSITIVE_SENTIMENT_COLOR,
            ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            ReportConfig.NEUTRAL_SENTIMENT_COLOR,
        ],
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
            fontsize=10,
            color="black",
            xytext=(10, 0),
            textcoords="offset points",
        )

    # Axes config
    ax.set_xlabel("")
    ax.set_xticks([])

    ax.set_ylabel("")

    # Get unique companies and create labels
    unique_companies = top_negative_companies_df["company"].unique()
    num_companies = len(unique_companies)

    # Always set 5 y-ticks
    num_ticks = 5
    y_positions = range(num_ticks)

    # Create labels with company names followed by "-" for empty slots
    labels = list(unique_companies) + [""] * (num_ticks - num_companies)
    labels = labels[:num_ticks]  # Ensure we don't exceed num_ticks

    # Set the y-ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)

    ax.set_title(
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.1,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

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
        labels=ReportConfig.PLOT_SENTIMENT_LABELS,
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        edgecolor="1",
        ncols=3,
    )

    return fig


@st.cache_data
def company_analisys():
    reviews_df = load_reviews_df()

    fig, ax = plt.subplots(1, figsize=(14, 6))
    sns.countplot(
        data=reviews_df,
        x="company",
        hue="sentiment",
        order=reviews_df["company"].value_counts().index,
        ax=ax,
        palette=ReportConfig.SENTIMENT_PALETTE,
    )

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    sns.despine(bottom=False, left=True)

    plt.xlabel("")
    plt.ylabel("")

    plt.xticks(rotation=45, ha="right")
    plt.yticks([])

    plt.legend(title="Sentimento", labels=ReportConfig.SENTIMENT_DICT.values())

    return fig


@st.cache_data
def sentiment_reviews_along_time():
    reviews_df = load_reviews_df()

    reviews_df["review_date"] = pd.to_datetime(
        reviews_df["review_date"], format="%Y-%m-%d"
    )
    reviews_df["year"] = reviews_df["review_date"].dt.year

    sentiment_counts = (
        reviews_df.groupby(["year", "sentiment"]).size().reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(14, 6))
    sns.lineplot(
        data=sentiment_counts,
        x="year",
        y="count",
        hue="sentiment",
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
                y=neutral_counts - 8,
                s=f"{neutral_counts}",
                ha="center",
                color=ReportConfig.NEUTRAL_SENTIMENT_COLOR,
            )

        if positive_counts:
            ax.text(
                x=year,
                y=positive_counts + 8,
                s=f"{positive_counts}",
                ha="center",
                color=ReportConfig.POSITIVE_SENTIMENT_COLOR,
            )

        if negative_counts:
            ax.text(
                x=year,
                y=negative_counts - 8,
                s=f"{negative_counts}",
                ha="center",
                color=ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.spines["bottom"].set_visible(False)

    ax.set_xlabel("Ano")
    ax.set_ylabel("Quantidade de avaliações")

    ax.set_title(
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.1,
    )

    handles, labels = ax.get_legend_handles_labels()
    order_map = {label: handle for handle, label in zip(handles, labels)}
    handles = [order_map[sentiment] for sentiment in ReportConfig.PLOT_SENTIMENT_VALUES]

    plt.legend(
        # title="Sentimento",
        handles=handles,
        labels=ReportConfig.PLOT_SENTIMENT_LABELS,
        bbox_to_anchor=(0.5, 1.1),
        loc="upper center",
        edgecolor="1",
        ncols=3,
    )

    return fig


@st.cache_data
def rating_star_analysis():
    reviews_df = load_reviews_df()

    filtered_df = reviews_df[
        [
            "company",
            "employee_role",
            "employee_detail",
            "review_text",
            "review_date",
            "star_rating",
            "sentiment_plot",
            "sentiment_label",
        ]
    ]

    filtered_df.reset_index(drop=True, inplace=True)

    if len(filtered_df) > 0:
        sentiment_counts = (
            filtered_df.groupby(["star_rating", "sentiment_plot"])
            .size()
            .reset_index(name="count")
        )

        fig, ax = plt.subplots(1, figsize=(14, 6))

        bars = sns.barplot(
            data=sentiment_counts,
            x="star_rating",
            y="count",
            hue="sentiment_plot",
            ax=ax,
            palette=[
                ReportConfig.POSITIVE_SENTIMENT_COLOR,
                ReportConfig.NEGATIVE_SENTIMENT_COLOR,
                ReportConfig.NEUTRAL_SENTIMENT_COLOR,
            ],
        )

        for p in bars.patches:
            height = p.get_height()
            if math.isnan(height):
                height = 0.0

            bars.annotate(
                text=f"{int(height)}",
                xy=(p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax.set_title(
            label="",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
            y=1.1,
        )

        ax.set_xlabel("")
        ax.set_xticklabels(
            ["★" * int(x) for x in sentiment_counts["star_rating"].unique()]
        )

        ax.set_yticks([])
        ax.set_ylabel("")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

        positive_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=ReportConfig.POSITIVE_SENTIMENT_COLOR
        )
        negative_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=ReportConfig.NEGATIVE_SENTIMENT_COLOR
        )
        neutral_patch = plt.Rectangle(
            (0, 0), 1, 1, fc=ReportConfig.NEUTRAL_SENTIMENT_COLOR
        )

        ax.legend(
            # title="Sentimento",
            handles=[positive_patch, negative_patch, neutral_patch],
            labels=ReportConfig.PLOT_SENTIMENT_LABELS,
            bbox_to_anchor=(0.5, 1.1),
            loc="upper center",
            edgecolor="1",
            ncols=3,
        )

        return fig
    else:
        st.error(
            AppMessages.ERROR_EMPTY_DATAFRAME,
            icon="🚨",
        )

        return None


@st.cache_data
def wordcloud_analysis():
    reviews_df = load_reviews_df()
    review_text = reviews_df["review_text"].str.split().values.tolist()
    corpus = [word for i in review_text for word in i]

    non_stopwords_corpus = []
    for word in corpus:
        word_lower = word.lower()
        cleaned_word = word_lower.translate(TRANSLATION_TABLE_SPECIAL_CHARACTERS)
        if cleaned_word and cleaned_word not in STOPWORDS:
            non_stopwords_corpus.append(cleaned_word)

    counter = Counter(non_stopwords_corpus)
    most_common_words = counter.most_common(n=50)

    mask = np.array(Image.open("./img/black_jaguar.jpg"))
    wordcloud = WordCloud(
        background_color="white",
        mask=mask,
        random_state=ReportConfig.RANDOM_SEED,
        # max_words=50,
        width=1024,
        height=768,
        contour_color="black",
        contour_width=1,
    )

    fig, ax = plt.subplots(1, figsize=(10, 6))
    plt.axis("off")

    ax.imshow(wordcloud.generate_from_frequencies(dict(most_common_words)))

    return fig


@st.cache_data
def most_common_words_analysis():
    reviews_df = load_reviews_df()
    review_text = reviews_df["review_text"].str.split().values.tolist()
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

    fig, ax = plt.subplots(1)

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
            fontsize=10,
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

    return fig


@st.cache_data
def ngram_analysis():
    reviews_df = load_reviews_df()

    review_text = reviews_df["review_text"]

    ngrams = get_top_ngrams(review_text, ngram_range=(3, 3), top_n=10)

    x, y = map(list, zip(*ngrams))

    fig, ax = plt.subplots(1)

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
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return fig


def setup_page():
    """Configure the Streamlit page settings and custom CSS."""
    st.set_page_config(
        page_title="Análise de sentimento em avaliações no Glassdoor",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better styling
    st.markdown(
        """
    <style>
        /* Positive Metrics */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2),
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) p {
            color: #2ca02c !important;
        }

        /* Negative Metrics */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(3),
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(3) p {
            color: #ff7f0e !important;
        }

        /* Neutral Metrics */
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(4),
        [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(4) p {
            color: #1f77b4 !important;
        }

    </style>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "use_inf_as_na")
    setup_page()

    st.title("📈 Análise de sentimento em avaliações no Glassdoor")
    st.markdown("Um estudo sobre empresas de Tecnologia da Informação em Cuiabá")

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(reviews_df)

    tab1, tab2 = st.tabs(["📊 Visão agregada", "📈 Visão detalhada"])

    with tab1:
        st.markdown("### 📊 Métricas gerais")
        with st.container():
            metrics()

        with st.container():
            st.markdown("---")
            st.markdown("### 🏆 Ranking de empresas")

            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("##### 🏅 Ranking de empresas com melhor avaliação")

                positive_reviews_ranking_plot = positive_reviews_ranking()
                st.pyplot(positive_reviews_ranking_plot)
                plt.close(positive_reviews_ranking_plot)

            with col2:
                st.markdown("##### ⚠️ Ranking de empresas com pior avaliação")

                negative_reviews_ranking_plot = negative_reviews_ranking()
                st.pyplot(negative_reviews_ranking_plot)
                plt.close(negative_reviews_ranking_plot)

        with st.container():
            st.markdown("---")

            st.markdown("### 📊 Análise de texto")
            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("#### 🔠 Top 10 palavras mais frequentes")

                most_common_words_analysis_plot = most_common_words_analysis()
                st.pyplot(most_common_words_analysis_plot)
                plt.close(most_common_words_analysis_plot)

            with col2:
                st.markdown("#### 🔤 Top 10 trigramas mais frequentes")

                ngram_analysis_plot = ngram_analysis()
                st.pyplot(ngram_analysis_plot)
                plt.close(ngram_analysis_plot)

    with tab2:
        with st.container():
            st.markdown("### 🏢 Distribuição de sentimentos por empresa")

            company_analisys_plot = company_analisys()
            st.pyplot(company_analisys_plot)
            plt.close(company_analisys_plot)

            st.markdown("---")

        with st.container():
            st.markdown("### 📅 Distribuição de sentimentos ao longo do tempo")

            sentiment_reviews_along_time_plot = sentiment_reviews_along_time()
            st.pyplot(sentiment_reviews_along_time_plot)
            plt.close(sentiment_reviews_along_time_plot)

            st.markdown("---")

        with st.container():
            st.markdown("### ⭐ Relação entre quantidade de estrelas e sentimento")

            rating_star_analysis_plot = rating_star_analysis()
            st.pyplot(rating_star_analysis_plot)
            plt.close(rating_star_analysis_plot)

            st.markdown("---")

        with st.container():
            st.markdown("### ☁️ Nuvem de palavras - Top 50")

            wordcloud_analysis_plot = wordcloud_analysis()
            st.pyplot(wordcloud_analysis_plot)
            plt.close(wordcloud_analysis_plot)

    st.markdown("---")
    with st.container():
        st.markdown("### ℹ️ Sobre os dados")
        st.markdown(
            """
        **Fonte:** Avaliações públicas do Glassdoor
        <br>
        **Período analisado:** 05/10/2014 a 16/03/2024
        <br>
        **Última atualização:** Março de 2024

        **Nota**: *Este projeto foi desenvolvido como parte do Trabalho de
        Conclusão de Curso da Pós-Graduação em Gestão e Ciência de Dados da
        Universidade Federal de Mato Grosso - UFMT. A classificação das
        avaliações como positivas, negativas ou neutras pode conter viés de
        interpretação subjetiva do autor.*
        """,
            unsafe_allow_html=True,
        )
