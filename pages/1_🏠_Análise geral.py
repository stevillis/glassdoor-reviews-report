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


def introduction():
    st.subheader("Introdução")

    st.markdown(
        """
    Se você é um profissional de Tecnologia da Informação (TI) e está buscando
    a primeira experiência ou mesmo uma recolocação, já deve ter analisado o
    perfil de algumas empresas para entender se estas seriam um bom fit para
    você. Plataformas como o **Glassdoor** são uma boa fonte para entender como
    ex-funcionários e funcionários atuais avaliam as empresas. Mas analisar
    centenas ou milhares de avaliações nessas plataformas pode ser
    um desafio enorme.

    Pensando em resolver um problema pessoal de entender como empresas de TI
    de Cuiabá eram avaliadas no Glassdoor e também ajudar profissionais com o
    mesmo desafio, treinei um Modelo de [Inteligência Artificial (IA)](https://pt.wikipedia.org/wiki/Intelig%C3%AAncia_artificial)
    baseado na técnica de [Transfer Learning](https://pt.wikipedia.org/wiki/Aprendizado_por_transfer%C3%AAncia)
    com [BERTimbau](https://neuralmind.ai/bert/). O Modelo foi treinado para
    classificar as avaliações da plataforma em três classes: **Positivo**,
    **Negativo** e **Neutro**. Os detalhes de **treinamento e avaliação do
    Modelo** estão disponíveis no menu [🧠 Treinamento do Modelo](./Treinamento_do_Modelo).
    Você pode utilizar o modelo acessando o [Hugging Face Spaces](https://huggingface.co/spaces/stevillis/bertimbau-finetuned-glassdoor-reviews).

    Ao explorar as seções subsequentes, você poderá entender melhor como as
    empresas de TI em Cuiabá são percebidas pelos funcionários e
    ex-funcionários. As análises dos dados gerais busca responder as seguintes
    questões:
    - Quais as empresas com melhor relação entre avaliações positivas e
    negativas? E quais seriam as piores nesse critério?
    - Ao longo do tempo, houve alguma mudança na quantidade de avaliações por
    sentimento?
    - Qual a distribuição de sentimentos de acordo com a nota atribuída
    para cada avaliação?
    - Avaliações de profissionais de TI são predominantes?
    - Quais as palavras mais frequentes nas avaliações?
    - Quais palavras aparecem com mais frequência juntas?
    """,
        unsafe_allow_html=True,
    )


@st.cache_data
def positive_reviews_ranking():
    st.subheader("Ranking de avaliações positivas por empresa")

    st.markdown(
        """
    Este gráfico ilustra as 5 empresas que apresentam **número de
    avaliações positivas superior ao de avaliações negativas**. Para garantir
    a relevância dos dados, **foram consideradas apenas as empresas com pelo
    menos 21 avaliações**, que representa a metade da mediana de avaliações de
    todas as empresas analisadas.
    """
    )

    top_positive_companies_df = st.session_state.get("top_positive_companies_df")

    fig, ax = plt.subplots(1, figsize=(10, 8))

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
    st.subheader("Ranking de avaliações negativas por empresa")

    st.markdown(
        """
    Este gráfico mostra as empresas que apresentam **número de
    avaliações negativas superior ao de avaliações positivas**, seguindo
    os mesmos critérios do gráfico anterior.
    """
    )

    top_negative_companies_df = st.session_state.get("top_negative_companies_df")

    fig, ax = plt.subplots(1, figsize=(10, 8))

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


def company_analisys():
    st.subheader("Sentimentos das Avaliações por Empresa")

    st.markdown(
        """
    A visualização da distribuição de avaliações e emoções em todas as
    empresas permite uma comparação rápida e uma visão abrangente do panorama
    geral.
    """
    )

    reviews_df = load_reviews_df()

    fig, ax = plt.subplots(1, figsize=(12, 6))
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


@st.cache_data
def sentiment_reviews_along_time():
    st.subheader("Sentimento das avaliações ao longo do tempo")

    st.markdown(
        """
     Este gráfico revela que as avaliações positivas sempre foram mais
     frequentes do que as negativas e neutras. O ano de 2022 destacou-se como
     o período com o maior número total de avaliações, apresentando também a
     maior disparidade entre as avaliações positivas e negativas.
    """,
        unsafe_allow_html=True,
    )

    reviews_df = load_reviews_df()

    reviews_df["review_date"] = pd.to_datetime(
        reviews_df["review_date"], format="%Y-%m-%d"
    )
    reviews_df["year"] = reviews_df["review_date"].dt.year

    sentiment_counts = (
        reviews_df.groupby(["year", "sentiment"]).size().reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(12, 6))
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

    st.subheader("Distribuição de sentimentos por quantidade de estrelas")

    st.markdown(
        """
        - As **avaliações de 1 a 3 estrelas** apresentam **sentimento
        predominantemente negativo**.
        - Por outro lado, as **avaliações de 4 estrelas** mostram uma
        **distribuição equilibrada entre sentimentos positivos e negativos**.
        - Já as **avaliações de 5 estrelas** são
        **majoritariamente positivas**, destacando-se também um
        **número significativo de avaliações neutras**.

        Essa predominância de avaliações neutras em avaliações de 5 estrelas
        pode ser atribuída à exigência ao usuário do Glassdoor de preencher as
        seções *Prós* e *Contras* ao avaliarem uma empresa. Em diversas
        avaliações, os usuários não encontram aspectos negativos a serem
        mencionados na seção *Contras*, resultando em comentários neutros como
        `Não há nada a ser apontado` ou `Não tenho nada a reclamar`.
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

        fig, ax = plt.subplots(1, figsize=(10, 6))

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

        return fig

    return None


@st.cache_data
def employee_role_analysis():
    reviews_df = load_reviews_df()

    st.subheader("Distribuição de sentimentos por grupo de funcionários")

    st.markdown(
        """
        Este gráfico revela que as **avaliações positivas são predominantes**,
        independentemente do grupo de funcionários. A maioria das
        avaliações provém de profissionais de outras áreas, com destaque para
        os seguintes dados:

        - Cerca de **64% das avaliações** são provenientes de profissionais de
        áreas não relacionadas à TI.
        - Os **profissionais de TI representam cerca de 25%** do total de
        avaliações.
        - Aproximadamente **11% das avaliações** foram emitidas por
        profissionais que optaram por não revelar seus cargos.
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
            "sentiment_plot",
            "sentiment_label",
            "role_group",
        ]
    ]

    filtered_df.reset_index(drop=True, inplace=True)

    if len(filtered_df) > 0:
        sentiment_counts = (
            reviews_df.groupby(["role_group", "sentiment_plot"])
            .size()
            .reset_index(name="sentiment_count")
        )

        fig, ax = plt.subplots(1, figsize=(10, 6))

        sns.barplot(
            data=sentiment_counts,
            x="sentiment_count",
            y="role_group",
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

        ax.set_title(
            label="",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
            y=1.1,
        )

        ax.set_xlabel("")
        ax.set_xticks([])

        ax.set_ylabel("")
        ax.set_yticklabels(
            [ROLE_GROUPS[group] for group in sentiment_counts["role_group"].unique()]
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

    return None


@st.cache_data
def wordcloud_analysis():
    st.subheader("Nuvem de Palavras")

    st.markdown(
        """
    A Nuvem de Palavras ([Word Cloud](https://techner.com.br/glossario/o-que-e-word-cloud/ "Word Cloud"))
    **é uma representação visual que ilustra as palavras mais frequentemente
    utilizadas no conjunto de avaliações**. Neste gráfico, as palavras
    aparecem em tamanhos variados, refletindo sua frequência de uso: quanto
    maior a palavra, mais vezes ela foi mencionada nas avaliações. É
    importante ressaltar que as *stopwords*, que são palavras comuns e
    geralmente sem significado relevante para a análise (como "e", "a", "o",
    "de") foram excluídas desta visualização.

    A Nuvem de Palavras a seguir mostra as 50 palavras mais frequentes nas
    avaliações e permite a identificação rápida dos tópicos mais relevantes,
    onde `empresa` e `trabalho` são visivelmente as palavras mais comuns.
    """
    )

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
    st.subheader("Top 10 palavras mais frequentes nas avaliações")

    st.markdown(
        """
        Embora a Nuvem de Palavras ofereça uma visão geral interessante das
        palavras mais utilizadas nas avaliações, ela pode não ser a melhor
        opção para destacar de forma clara e precisa a palavra mais frequente.
        Para complementar essa análise e oferecer uma visão mais quantitativa,
        é apresentado um gráfico com as 10 palavras mais utilizadas nas
        avaliações analisadas, junto como suas respectivas frequências.

        Este gráfico segue os mesmos critérios da Nuvem de Palavras,
        garantindo que as palavras selecionadas sejam relevantes e
        significativas.
    """
    )

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
    st.subheader("Top 10 N-Grams mais frequentes nas avaliações")

    st.markdown(
        """
    Embora o gráfico de palavras mais frequentes forneça uma visão inicial
    sobre os termos mais utilizados nas avaliações, ele não captura a riqueza
    dos contextos em que essas palavras aparecem. Palavras isoladas podem ter
    significados variados e não revelam como elas se combinam para formar
    ideias ou sentimentos mais complexos. Por exemplo, a palavra `crescimento`
    pode aparecer frequentemente, mas sem o contexto, como em `oportunidade de
    crescimento`, seu significado pode ser ambíguo.

    Os [N-Gramas](https://pt.wikipedia.org/wiki/N-grama) são sequências
    contíguas de "n" itens (palavras ou caracteres) e são essenciais para uma
    análise mais profunda, pois permitem identificar padrões e temas
    recorrentes nas avaliações. Ao considerar as combinações de palavras, é
    possível entender melhor as percepções dos funcionários e os aspectos mais
    relevantes de suas experiências.

    Ao analisar os Top 10 Trigramas mais frequentes, conclui-se que as
    combinações de palavras mais frequentes foram: `ambiente de trabalho`,
    `plano de carreira` e `plano de saúde`.
    """
    )

    reviews_df = load_reviews_df()

    review_text = reviews_df["review_text"]

    ngrams = get_top_ngrams(review_text, ngram_range=(3, 3), top_n=10)

    x, y = map(list, zip(*ngrams))

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
        label="",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        y=1.0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    return fig


def conclusion():
    st.subheader("Conclusão")

    st.markdown(
        """
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

    **A reputação positiva, refletida nas avaliações, pode ser um diferencial
    decisivo na escolha de uma empresa por candidatos qualificados**,
    impactando diretamente o sucesso e a competitividade no mercado.

    #### Tecnologias e ferramentas usadas

    | **Categoria**                     | **Tecnologia/Ferramenta**                                                                                                                                           |
    |-----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
    | Extração de Dados                 | ![Selenium](https://img.shields.io/badge/-selenium-%43B02A?style=for-the-badge&logo=selenium&logoColor=white) BeautifulSoup                           |
    | Manipulação e Análise de Dados    | ![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) |
    | Treinamento e Avaliação do Modelo | ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) |
    | Visualização de Dados             | ![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black) ![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=for-the-badge&logo=streamlit&logoColor=white) Seaborn |
    | Versionamento                     | ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white) |
    | Ambiente de Desenvolvimento       | ![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle-035a7d?style=for-the-badge&logo=kaggle&logoColor=white) Google Colab |

    """,
        unsafe_allow_html=True,
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
        "Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá"
    )

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(
        reviews_df
    )  # TODO: maybe this can be done in load_reviews_df

    introduction()

    # company_analisys()

    with st.container():
        st.pyplot(positive_reviews_ranking())

    with st.container():
        st.pyplot(negative_reviews_ranking())
        st.markdown(
            """
        O ranking completo de avaliações por empresa pode ser visualizado no
        menu <a target="_self" href="./Ranking_geral_de_avaliações">🥇Ranking
        geral de avaliações</a>.
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.pyplot(sentiment_reviews_along_time())
        st.markdown(
            """
        As avaliações ao longo do tempo por empresa podem ser visualizadas no
        menu <a target="_self" href="./Avaliações_ao_longo_do_tempo">
        📉Avaliações ao longo do tempo</a>.

        <br/>
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        fig = rating_star_analysis()
        if fig:
            st.pyplot(fig)
        else:
            st.error(
                AppMessages.ERROR_EMPTY_DATAFRAME,
                icon="🚨",
            )

        st.markdown(
            """
            A distribuição de sentimentos por quantidade de estrelas para cada
            empresa pode ser visualizada no menu
            <a target="_self" href="./Avaliações_por_quantidade_de_estrelas">
            📊Avaliações por quantidade de estrelas</a>.
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        fig = rating_star_analysis()
        if fig:
            st.pyplot(fig)
        else:
            st.error(
                AppMessages.ERROR_EMPTY_DATAFRAME,
                icon="🚨",
            )

    with st.container():
        fig = employee_role_analysis()
        if fig:
            st.pyplot(fig)
        else:
            st.error(
                AppMessages.ERROR_EMPTY_DATAFRAME,
                icon="🚨",
            )

        st.markdown(
            """
            A distribuição de sentimentos por grupo de funcionários para cada
            empresa pode ser visualizada no menu
            <a target="_self" href="./Avaliações_por_grupo_de_funcionários">
            📊Avaliações por grupo de funcionários</a>.
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.pyplot(wordcloud_analysis())
        st.markdown(
            """
        A Word Cloud de avaliações por sentimento e por empresa pode ser
        visualizada no menu
        <a target="_self" href="./Nuvem_de_Palavras_por_empresa">☁️Nuvem de Palavras por empresa</a>.
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.pyplot(most_common_words_analysis())
        st.markdown(
            """
        As Top 10 palavras mais frequentes nas avaliações por empresa e por
        sentimento podem ser visualizadas no menu
        <a target="_self" href="./Top_10_palavras_mais_usadas">📊Top 10 palavras mais frequentes</a>.
        """,
            unsafe_allow_html=True,
        )

    with st.container():
        st.pyplot(ngram_analysis())
        st.markdown(
            """
        Os Top 10 N-Grams mais frequentes nas avaliações de cada empresa pode
        ser visualizado no menu <a target="_self" href="./NGrams">🔠NGrams</a>.
    """,
            unsafe_allow_html=True,
        )

    with st.container():
        conclusion()
