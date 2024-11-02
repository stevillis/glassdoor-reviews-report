import math
import warnings
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from app_messages import AppMessages
from report_config import ReportConfig
from utils import (
    STOPWORDS,
    TRANSLATION_TABLE_SPECIAL_CHARACTERS,
    create_predicted_sentiment_plot,
    get_ranking_positive_negative_companies,
)


def introduction():
    st.markdown(
        """
   As avaliações de funcionários no Glassdoor proporcionam uma visão valiosa
   sobre a cultura e o ambiente de trabalho de uma empresa. No contexto do
   setor de Tecnologia, entender as emoções expressas nessas avaliações é
   crucial para atrair e reter talentos.

   Além disso, essas percepções podem  servir como base para reestruturações
   necessárias, especialmente em resposta a comentários negativos. Ao
   valorizar o feedback dos colaboradores, as organizações podem não apenas
   melhorar seu ambiente interno, mas também garantir um caminho sólido para o
   sucesso a longo prazo.

    Esse trabalho mostra uma **análise das emoções expressas nas avaliações
    no Glassdoor de 22 empresas de Tecnologia de Cuiabá**, com dados de **05 de
    outubro de 2014 a 16 de março de 2024** e um total de **2532 avaliações**.
    Para isso, foi treinado um Modelo de [Inteligência Artificial (IA)](https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial)
    baseado na técnica de [Transfer Learning](https://pt.wikipedia.org/wiki/Aprendizado_por_transfer%C3%AAncia)
    com [BERTimbau](https://neuralmind.ai/bert/).

    Os detalhes de treinamento e avaliação do Modelo podem ser acessados no
    menu
    <a target="_self" href="./Treinamento_do_Modelo">🧠Treinamento do Modelo</a>.

    #### Tecnologias e ferramentas usadas

    | **Categoria**                     | **Ferramentas e Tecnologias**                                                                                                                                           |
    |-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
    | Extração de Dados                 | ![Selenium](https://img.shields.io/badge/-selenium-%43B02A?style=for-the-badge&logo=selenium&logoColor=white) BeautifulSoup                           |
    | Manipulação e Análise de Dados    | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
    | Treinamento e Avaliação do Modelo | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) |
    | Visualização de Dados             | ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) Seaborn |
    | Versionamento                     | ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) |
    | Ambiente de Desenvolvimento       | ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white) Google Colab |

    <br/>
    As seções a seguir apresentam as predições realizadas pelo modelo treinado
    para todas as avaliações. Nelas, é possível comparar os dados reais com
    as previsões geradas pelo Modelo, permitindo uma avaliação clara da
    eficácia do modelo na tarefa proposta.

    <br/>
    """,
        unsafe_allow_html=True,
    )


def positive_reviews_ranking():
    st.subheader("Ranking de avaliações positivas por empresa")

    st.markdown(
        """
    Este gráfico ilustra as cinco empresas que apresentam um número de
    avaliações positivas superior ao de avaliações negativas. Para garantir
    a relevância dos dados, foram consideradas apenas as empresas que
    possuem pelo menos 21 avaliações, um critério que representa a metade da
    mediana de avaliações de todas as empresas analisadas.
"""
    )

    top_positive_companies_df = st.session_state.get("top_positive_companies_df")

    fig, ax = plt.subplots(1, figsize=(10, 8))

    sns.barplot(
        data=top_positive_companies_df,
        x="sentiment_count",
        y="company",
        hue="predicted_sentiment_plot",
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
        "Ranking de avaliações positivas por empresa",
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

    # plt.savefig(
    #     "positive_reviews_by_company.png",
    #     transparent=False,
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    st.pyplot(fig)


def negative_reviews_ranking():
    st.subheader("Ranking de avaliações negativas por empresa")

    st.markdown(
        """
    Este gráfico mostra as empresas que apresentam um número de
    avaliações negativas superior ao de avaliações positivas, seguindo
    os mesmos critérios do gráfico anterior.
"""
    )

    top_negative_companies_df = st.session_state.get("top_negative_companies_df")

    fig, ax = plt.subplots(1, figsize=(10, 8))

    sns.barplot(
        data=top_negative_companies_df,
        x="sentiment_count",
        y="company",
        hue="predicted_sentiment_plot",
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
        "Ranking de avaliações negativas por empresa",
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

    # plt.savefig(
    #     "negative_reviews_by_company.png",
    #     transparent=True,
    #     dpi=300,
    #     bbox_inches="tight",
    # )

    st.pyplot(fig)

    st.markdown(
        """
        O ranking completo de avaliações por empresa pode ser visualizado no
        menu <a target="_self" href="./Ranking_geral_de_avaliações">🥇Ranking
        geral de avaliações</a>.

        <br/>
    """,
        unsafe_allow_html=True,
    )


def company_analisys():
    st.subheader("Sentimentos das Avaliações por Empresa")

    st.markdown(
        """
    A visualização da distribuição de avaliações e emoções em todas as empresas permite uma comparação rápida e uma visão abrangente do panorama geral.
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    fig, ax = plt.subplots(1, figsize=(12, 6))
    sns.countplot(
        data=reviews_df,
        x="company",
        hue="predicted_sentiment",
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

    plt.title(
        "Sentimentos das Avaliações por Empresa",
        fontdict={
            "weight": "bold",
            "size": ReportConfig.CHART_TITLE_FONT_SIZE,
        },
    )

    plt.xlabel("")
    plt.ylabel("")

    plt.xticks(rotation=45, ha="right")
    plt.yticks([])

    plt.legend(title="Sentimento", labels=ReportConfig.SENTIMENT_DICT.values())

    st.pyplot(fig)


def sentiment_reviews_along_time():
    st.subheader("Sentimento das avaliações ao longo do tempo")

    st.markdown(
        """
     Este gráfico revela que as avaliações positivas sempre foram mais
     frequentes do que as negativas e neutras. O ano de 2022 destacou-se como
     o período com o maior número total de avaliações, apresentando também a
     maior disparidade entre as avaliações positivas e negativas.

    <br/>
""",
        unsafe_allow_html=True,
    )

    reviews_df = st.session_state.get("reviews_df")

    reviews_df["review_date"] = pd.to_datetime(
        reviews_df["review_date"], format="%Y-%m-%d"
    )
    reviews_df["year"] = reviews_df["review_date"].dt.year

    sentiment_counts = (
        reviews_df.groupby(["year", "predicted_sentiment"])
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
        "Sentimento das avaliações ao longo do tempo",
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

    st.pyplot(fig)

    st.markdown(
        """
        As avaliações ao longo do tempo por empresa podem ser visualizadas no
        menu <a target="_self" href="./Avaliações_ao_longo_do_tempo">
        📉Avaliações ao longo do tempo</a>.

        <br/>
    """,
        unsafe_allow_html=True,
    )

    # plt.savefig(
    #     "sentiments_reviews_along_time.png",
    #     transparent=True,
    #     dpi=300,
    #     bbox_inches="tight",
    # )


def rating_star_analysis():
    warnings.filterwarnings("ignore", "use_inf_as_na")
    reviews_df = st.session_state.get("reviews_df")

    st.subheader(
        "Distribuição de Sentimentos por Quantidade de Estrelas em Avaliações, por Empresa"
    )

    st.markdown(
        """
    Esta análise mostra padrões interessantes na relação entre as avaliações e o número de estrelas atribuídas, revelando correlações intrigantes entre a satisfação dos funcionários e a classificação geral.
"""
    )

    g = sns.FacetGrid(
        reviews_df,
        col="company",
        hue="predicted_sentiment",
        col_wrap=4,
    )
    g.map(sns.histplot, "star_rating")

    g.set_titles("{col_name}")
    g.set_axis_labels("Star Rating", "Count")
    g.add_legend(
        title="Predicted Sentiment",
        labels=list(ReportConfig.SENTIMENT_DICT.values()),
    )

    st.pyplot(g)


def rating_star_analysis2():
    reviews_df = st.session_state.get("reviews_df")

    st.subheader(
        "Distribuição de Sentimentos por Quantidade de Estrelas em Avaliações, por Empresa"
    )

    st.markdown(
        """
        Esta análise mostra padrões interessantes na relação entre as avaliações e o número de estrelas atribuídas, revelando correlações intrigantes entre a satisfação dos funcionários e a classificação geral.
    """
    )

    company = st.selectbox(
        label="Empresa",
        options=reviews_df["company"].unique().tolist(),
        key="rating_star_company_input",
    )

    filtered_df = reviews_df[reviews_df["company"] == company][
        [
            "company",
            "employee_role",
            "employee_detail",
            "review_text",
            "review_date",
            "star_rating",
            "predicted_sentiment",
            "sentiment_label",
        ]
    ]
    filtered_df.reset_index(drop=True, inplace=True)

    sentiment_counts = (
        filtered_df.groupby(["company", "star_rating", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(12, 8))

    sns.scatterplot(
        data=sentiment_counts,
        x="star_rating",
        y="count",
        hue="predicted_sentiment",
        # style="company",
        ax=ax,
        palette=sns.color_palette(),
    )

    plt.title(
        "Sentiment Counts by Star Rating and Company",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
    )
    plt.xlabel("Star Rating")
    plt.ylabel("Count")

    plt.xticks(ticks=sorted(reviews_df["star_rating"].unique()))

    handles, labels = ax.get_legend_handles_labels()
    handles[0].set_color([ReportConfig.NEGATIVE_SENTIMENT_COLOR])
    handles[1].set_color([ReportConfig.POSITIVE_SENTIMENT_COLOR])
    handles[2].set_color([ReportConfig.NEUTRAL_SENTIMENT_COLOR])

    legend_labels = [
        ReportConfig.SENTIMENT_DICT[i] for i in range(len(ReportConfig.SENTIMENT_DICT))
    ]

    plt.legend(
        title="Sentiment",
        labels=legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    st.pyplot(fig)

    st.write("Avaliações filtradas")
    filtered_df = filtered_df.drop(labels="predicted_sentiment", axis=1)
    st.dataframe(filtered_df)


def rating_star_analysis3():
    reviews_df = st.session_state.get("reviews_df")
    reviews_df = create_predicted_sentiment_plot(reviews_df)

    st.subheader("Distribuição de sentimentos por quantidade de estrelas")

    st.markdown(
        """
        Este gráfico ilustra que:
        - As avaliações de 1 a 3 estrelas apresentam um sentimento
        predominantemente negativo.
        - Por outro lado, as avaliações de 4 estrelas mostram uma distribuição
        equilibrada entre sentimentos positivos e negativos.
        - Já as avaliações de 5 estrelas são majoritariamente positivas,
        destacando-se também um número significativo de avaliações neutras.

        Essa predominância de avaliações neutras em avaliações de 5 estrelas
        pode ser atribuída à exigência no Glassdoor de preencher as seções
        *Prós* e *Contras*. Em diversas avaliações, os usuários não encontram
        aspectos negativos a serem mencionados na seção *Contras*, resultando
        em comentários neutros como `Não há nada a ser apontado` ou `Não tenho
        nada a reclamar`.
    """
    )

    filtered_df = reviews_df[
        [
            "company",
            "employee_role",
            "employee_detail",
            "review_text",
            "review_date",
            "star_rating",
            "predicted_sentiment_plot",
            "sentiment_label",
        ]
    ]

    filtered_df.reset_index(drop=True, inplace=True)

    if len(filtered_df) > 0:
        sentiment_counts = (
            filtered_df.groupby(["star_rating", "predicted_sentiment_plot"])
            .size()
            .reset_index(name="count")
        )

        fig, ax = plt.subplots(1, figsize=(10, 6))

        bars = sns.barplot(
            data=sentiment_counts,
            x="star_rating",
            y="count",
            hue="predicted_sentiment_plot",
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
            "Distribuição de sentimentos por quantidade de estrelas",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
            y=1.1,
        )

        ax.set_xlabel("")
        ax.set_xticklabels(
            ["\u2605" * int(x) for x in sentiment_counts["star_rating"].unique()]
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

        st.pyplot(fig)

        # plt.savefig(
        #     "sentiment_by_rating_star.png",
        #     transparent=True,
        #     dpi=300,
        #     bbox_inches="tight",
        # )
    else:
        st.error(
            AppMessages.ERROR_EMPTY_DATAFRAME,
            icon="🚨",
        )

    st.markdown(
        """
        A distribuição de sentimentos por quantidades de estrelas para cada
        empresa pode ser visualizada no menu
        <a target="_self" href="./Avaliações_por_quantidade_de_estrelas">
        📊Avaliações por quantidade de estrelas</a>.

        <br/>
    """,
        unsafe_allow_html=True,
    )


def wordcloud_analysis():
    st.subheader("Word Cloud de todas as avaliações")

    st.markdown(
        """
    A Word Cloud (Nuvem de Palavras) é uma representação visual que ilustra as
    palavras mais frequentemente utilizadas no conjunto de avaliações. Neste
    gráfico, as palavras aparecem em tamanhos variados, refletindo sua
    frequência de uso: quanto maior a palavra, mais vezes ela foi mencionada
    nas avaliações.

    É importante ressaltar que as stopwords, que são palavras comuns e
    geralmente sem significado relevante para a análise (como "e", "a", "o",
    "de"), foram excluídas desta visualização.

    Essa abordagem permite uma análise mais clara e focada, facilitando a
    identificação rápida dos tópicos mais relevantes e das percepções
    predominantes dos usuários.
"""
    )

    reviews_df = st.session_state.get("reviews_df")
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

    wordcloud = WordCloud(
        background_color="white",
        random_state=ReportConfig.RANDOM_SEED,
        # max_words=50,
        width=1024,
        height=768,
    )

    fig, ax = plt.subplots(1, figsize=(10, 6))
    plt.axis("off")

    ax.imshow(wordcloud.generate_from_frequencies(dict(most_common_words)))

    st.pyplot(fig)

    st.markdown(
        """
        A Word Cloud de avaliações por sentimento e por empresa pode ser
        visualizada no menu
        <a target="_self" href="./Word_Cloud">☁️Word Cloud</a>.
    """,
        unsafe_allow_html=True,
    )


def most_common_words_analysis():
    st.subheader("Top 10 palavras mais frequentes nas avaliações")

    st.markdown(
        """
        Embora a Word Cloud ofereça uma visão geral interessante das
        palavras mais utilizadas nas avaliações, ela pode não ser a melhor
        opção para destacar de forma clara e precisa a palavra mais frequente.
        Para complementar essa análise, é mostrado o gráfico de barras que
        apresenta as 10 palavras mais frequentemente utilizadas nas avaliações
        analisadas.

        Este gráfico segue os mesmos critérios da Word Cloud, garantindo que
        as palavras selecionadas sejam relevantes e significativas. Com a
        disposição em barras, é possível visualizar facilmente a frequência de
        cada palavra, permitindo uma comparação direta entre elas.

        Essa abordagem torna a interpretação dos dados mais intuitiva e
        acessível, facilitando a identificação dos temas mais recorrentes nas
        avaliações.
"""
    )

    reviews_df = st.session_state.get("reviews_df")
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
        "Top 10 palavras mais frequentes nas avaliações",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    st.pyplot(fig)

    st.markdown(
        """
        As Top 10 palavras mais frequentes nas avaliações por empresa e por
        sentimento podem ser visualizadas no menu <a target="_self" href="./Top_10_palavras_mais_usadas">📊Top 10 palavras mais frequentes</a>.

        <br/>
    """,
        unsafe_allow_html=True,
    )


def ngram_analysis():
    st.subheader("Top 10 NGrams mais frequentes nas avaliações")

    st.markdown(
        """
    Embora o gráfico de palavras mais frequentes forneça uma visão inicial
    sobre os termos mais utilizados nas avaliações, ele não captura a riqueza
    dos contextos em que essas palavras aparecem. Palavras isoladas podem ter
    significados variados e não revelam como elas se combinam para formar
    ideias ou sentimentos mais complexos. Por exemplo, a palavra `crescimento`
    pode aparecer frequentemente, mas sem o contexto, como em `oportunidade de
    crescimento`, seu significado pode ser ambíguo.

    Os N-Gramas são sequências contíguas de "n" itens (palavras ou
    caracteres) e são essenciais para uma análise mais profunda, pois permitem
    identificar padrões e temas recorrentes nas avaliações.

    Ao considerar as combinações de palavras, é possível entender melhor as
    percepções dos funcionários e os aspectos mais relevantes de suas
    experiências. Essa análise revelou que as combinações de palavras mais
    frequentes, considerando todas as avaliações, foram: `ambiente de
    trabalho`, `plano de carreira`, `plano de saúde` e `oportunidade de
    crescimento`.
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    review_text = reviews_df["review_text"]

    vec = CountVectorizer(ngram_range=(3, 3)).fit(review_text)
    bag_of_words = vec.transform(review_text)
    sum_words = bag_of_words.sum(axis=0)

    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
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
        "Top 10 NGrams mais frequentes nas avaliações",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    st.pyplot(fig)

    st.markdown(
        """
        Os Top 10 NGrams mais frequentes nas avaliações de cada empresa pode
        ser visualizado no menu <a target="_self" href="./NGrams">🔠NGrams</a>.

        <br/>
    """,
        unsafe_allow_html=True,
    )


def conclusion():
    # TODO: fix the final description
    st.subheader("Conclusão")

    st.markdown(
        """
    **A análise de sentimentos das avaliações no Glassdoor de 22 empresas de
    Tecnologia em Cuiabá** revelou que o modelo de IA, baseado na técnica de
    Transfer Learning com BERTimbau, **demonstrou uma alta acurácia de 95% na
    classificação das avaliações**, evidenciando a eficácia metodologia aplicada.

    Os resultados indicam que **15 das 22 empresas analisadas possuem mais
    avaliações positivas do que negativas**, refletindo um ambiente de trabalho
    predominantemente favorável.

    As **avaliações positivas** frequentemente mencionam temas como **ambiente
    de trabalho**, **plano de saúde** e **oportunidade de crescimento**,
    enquanto as **avaliações negativas** destacam preocupações com **plano de
    carreira**, **salário abaixo do mercado** e, em alguns casos, o **plano de
    saúde**. As **avaliações neutras**, embora menos frequentes, sugerem que
    muitos colaboradores **não encontraram aspectos negativos a serem
    destacados**, indicando uma satisfação geral com suas experiências.

    Além disso, a análise temporal revelou que **as avaliações positivas sempre
    foram predominantes em relação as demais**. Esta análise também mostrou que
    houve um grande aumento no número de avaliações entre 2020 e 2022, período
    da Pandemia de Covid-19, onde as empresas contrataram mais.

    Esses insights são fundamentais para as empresas, pois proporcionam uma
    visão clara das áreas que precisam de melhorias e das que já estão
    apresentando resultados positivos. Com base nessas informações, as
    organizações podem desenvolver estratégias eficazes para aprimorar o
    ambiente de trabalho, focar em benefícios que realmente importam para os
    colaboradores e, assim, não apenas aumentar a retenção de talentos, mas
    também atrair profissionais que buscam ambientes com melhores avaliações.
    A reputação positiva, refletida nas avaliações, pode ser um diferencial
    decisivo na escolha de uma empresa por candidatos qualificados, impactando
    diretamente o sucesso e a competitividade no mercado.
"""
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "use_inf_as_na")

    st.set_page_config(
        page_title="Home",
        page_icon=":house:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.header(
        "Análise de sentimentos nas avaliações do Glassdoor: Um estudo sobre empresas de Tecnologia em Cuiabá"
    )

    if "reviews_df" not in st.session_state:
        # Reviews DF
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")

        # TODO:check where the "sentiment" column is used and if it is being
        # used instead of "predicted_sentiment"

        st.session_state["reviews_df"] = reviews_df

        # Top Companies Reviews DF
        if "top_positive_companies_df" not in st.session_state:
            top_positive_companies_df, top_negative_companies_df = (
                get_ranking_positive_negative_companies(reviews_df)
            )

            st.session_state["top_positive_companies_df"] = top_positive_companies_df
            st.session_state["top_negative_companies_df"] = top_negative_companies_df

    introduction()

    # company_analisys()

    # TODO: create positive reviews ranking for `sentiment` column.
    positive_reviews_ranking()
    # TODO: create negative reviews ranking for `sentiment` column.
    negative_reviews_ranking()

    # TODO: create sentiment reviews along time for `sentiment` column.
    sentiment_reviews_along_time()

    # rating_star_analysis()
    # rating_star_analysis2()

    # TODO: create rating star analysis for `sentiment` column.
    rating_star_analysis3()

    # TODO: create wordcloud for `sentiment` column.
    wordcloud_analysis()

    # TODO: create most common words plot for `sentiment` column.
    most_common_words_analysis()

    # TODO: create ngram plot for `sentiment` column.
    ngram_analysis()

    # TODO: refact conclusion, explaining that the model is efficient for sentiment classification
    conclusion()
