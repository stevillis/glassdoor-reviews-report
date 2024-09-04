import math
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import (
    create_predicted_sentiment_plot,
    get_bad_rating_companies,
    get_good_rating_companies,
    get_neutral_rating_companies,
    get_sentiment_key_from_value,
)


def introduction():
    st.markdown(
        """
       As avaliações dos funcionários no Glassdoor são como janelas para o coração de uma empresa.
       Elas revelam não apenas a cultura e o ambiente de trabalho, mas também o pulso emocional dos colaboradores.
       Em setores altamente competitivos, como o de Tecnologia, entender essas emoções pode ser a chave para atrair talentos, encantar clientes e impulsionar o sucesso empresarial.

       A fim de identificar as emoções nas avaliações do Glassdoor de 22 empresas de Tecnologia de Cuiabá, foi criado um Modelo de IA utilizando a técnica de Transfer Learning com BERT (Bidirectional Encoder Representations from Transformers),
       um Modelo de linguagem pré-treinado que utiliza a representação bidirecional de texto para entender o contexto das palavras em uma frase ou texto.

       O Modelo pré-treinado utilizado como base para a criação do Modelo de identificação de sentimentos foi o BERTimbau, um Modelo que consiste no BERT, mas treinado com a língua portuguesa.
       Os insights que surgiram da análise das avaliações do Glassdoor são apresentados a seguir.
"""
    )


def general_analysis():
    st.subheader("Distribuição de sentimentos das avaliações")

    st.markdown(
        """
       **Metodologia**

        Antes de treinar o modelo de aprendizado de máquina para classificar os sentimentos das avaliações extraídas do Glassdoor, foi necessário preparar os dados. Essa preparação envolveu:

        *Classificação manual de uma amostra das avaliações*

        Uma parte das avaliações foi classificada manualmente, utilizando uma ferramenta de anotação criada pelo próprio autor.
        Esse conjunto de dados extraídos e classificados manualmente é chamado de "sentimentos anotados" e serviu como base de treinamento e validação para o modelo.

        *Tratamento do desequilíbrio de classes*

        Ao analisar o conjunto de dados anotados, observou-se um desequilíbrio significativo entre as classes de sentimento. Avaliações classificadas como Neutro representavam quase 5 vezes
        menos do que as demais classes (Positivo e Negativo). Para lidar com esse problema, foi aplicada a técnica de oversampling na classe Neutro, replicando aleatoriamente algumas amostras
        dessa classe durante o treinamento. Isso ajudou a balancear a distribuição das classes e melhorar o desempenho do modelo na identificação correta de avaliações Neutras.

        **Resultados**

        *Comparação entre dados anotados e classificados pelo modelo*

        O gráfico a seguir mostra a comparação entre a distribuição dos sentimentos nos dados anotados manualmente e a distribuição dos sentimentos classificados pelo modelo treinado:
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    fig, ax = plt.subplots(2, 1, figsize=(8, 4))

    # Annotated sentiment
    sentiment_counts = reviews_df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["sentiment", "count"]

    sentiment_counts["sentiment"] = sentiment_counts["sentiment"].map(
        lambda x: ReportConfig.SENTIMENT_DICT[x]
    )

    sns.barplot(
        data=sentiment_counts,
        y="sentiment",
        x="count",
        palette=[
            ReportConfig.POSITIVE_SENTIMENT_COLOR,
            ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            ReportConfig.NEUTRAL_SENTIMENT_COLOR,
        ],
        ax=ax[0],
    )

    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    ax[0].spines["bottom"].set_visible(False)

    for p in ax[0].patches:
        ax[0].annotate(
            f"{int(p.get_width())}",
            (p.get_width() - 50, p.get_y() + p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            xytext=(0, 0),
            textcoords="offset points",
        )

    ax[0].set_xticks([])
    ax[0].set_xlabel("")
    ax[0].set_ylabel("")
    ax[0].set_title(
        "Distribuição de sentimentos anotados",
        loc="center",
        fontsize=14,
    )

    # Predicted sentiment
    predicted_sentiment_counts = (
        reviews_df["predicted_sentiment"].value_counts().reset_index()
    )
    predicted_sentiment_counts.columns = ["predicted_sentiment", "count"]

    predicted_sentiment_counts["predicted_sentiment"] = predicted_sentiment_counts[
        "predicted_sentiment"
    ].map(lambda x: ReportConfig.SENTIMENT_DICT[x])

    sns.barplot(
        data=predicted_sentiment_counts,
        y="predicted_sentiment",
        x="count",
        palette=[
            ReportConfig.POSITIVE_SENTIMENT_COLOR,
            ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            ReportConfig.NEUTRAL_SENTIMENT_COLOR,
        ],
        ax=ax[1],
    )

    for p in ax[1].patches:
        ax[1].annotate(
            f"{int(p.get_width())}",
            (p.get_width() - 50, p.get_y() + p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            xytext=(0, 0),
            textcoords="offset points",
        )

    ax[1].set_xticks([])
    ax[1].set_xlabel("")
    ax[1].set_ylabel("")
    ax[1].set_title(
        "Distribuição de sentimentos classificados pelo Modelo",
        loc="center",
        fontsize=14,
    )

    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)


def company_analisys():
    st.subheader("Sentimentos das Avaliações por Empresa")

    st.markdown(
        """
    A visualização da distribuição de avaliações e emoções em todas as empresas permite uma comparação rápida e uma visão abrangente do panorama geral.
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    LIMIT_COMPANY_NAME = 30
    reviews_df["company"] = reviews_df["company"].apply(
        lambda x: x[:LIMIT_COMPANY_NAME] + "" if len(x) > LIMIT_COMPANY_NAME else x
    )

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


def company_analisys2():
    st.subheader("Distribuição de Sentimentos das Avaliações por Empresa")

    st.markdown(
        """Este gráfico mostra a quantidade de avaliações e o sentimento associado para cada empresa, ordenadas pela totalidade de avaliações.

**Metodologia**

As avaliações neutras não foram consideradas na relação entre avaliações positivas e negativas.

**Resultados**

- 15 das 22 empresas analisadas têm mais avaliações positivas do que negativas.
- 4 empresas apresentam um número maior de avaliações negativas.
- 3 empresas têm um número igual de avaliações positivas e negativas.

"""
    )

    reviews_df = st.session_state.get("reviews_df")

    reviews_df = create_predicted_sentiment_plot(reviews_df)

    predicted_sentiment_plot_by_company_df = (
        reviews_df.groupby(["company", "predicted_sentiment_plot"])
        .size()
        .unstack(fill_value=0)
    )

    predicted_sentiment_plot_by_company_df_reset = (
        predicted_sentiment_plot_by_company_df.reset_index()
    )
    predicted_sentiment_plot_by_company_df_reset.columns = [
        "company",
        "1positive",
        "2negative",
        "3neutral",
    ]

    bad_rating_companies = get_bad_rating_companies(
        predicted_sentiment_plot_by_company_df_reset
    )
    good_rating_companies = get_good_rating_companies(
        predicted_sentiment_plot_by_company_df_reset
    )
    neutral_rating_companies = get_neutral_rating_companies(
        predicted_sentiment_plot_by_company_df_reset
    )

    fig, ax = plt.subplots(1, figsize=(10, 8))

    # Plot
    sns.countplot(
        data=reviews_df,
        y="company",
        hue="predicted_sentiment_plot",
        order=reviews_df["company"].value_counts().index,
        ax=ax,
        palette=[
            ReportConfig.POSITIVE_SENTIMENT_COLOR,
            ReportConfig.NEGATIVE_SENTIMENT_COLOR,
            ReportConfig.NEUTRAL_SENTIMENT_COLOR,
        ],
        width=0.9,
    )

    # Highlight Companies
    for i, label in enumerate(ax.get_yticklabels()):
        company_name = label.get_text()
        if company_name in bad_rating_companies:
            label.set_color(ReportConfig.NEGATIVE_SENTIMENT_COLOR)

    for i, label in enumerate(ax.get_yticklabels()):
        company_name = label.get_text()
        if company_name in good_rating_companies:
            label.set_color(ReportConfig.POSITIVE_SENTIMENT_COLOR)

    for i, label in enumerate(ax.get_yticklabels()):
        company_name = label.get_text()
        if company_name in neutral_rating_companies:
            label.set_color(ReportConfig.NEUTRAL_SENTIMENT_COLOR)

    # Plot Annotates
    for p in ax.patches:
        ax.annotate(
            text=f"{p.get_width():.0f}",
            xy=(p.get_width() + 10, (p.get_y() + p.get_height() / 2) + 0.02),
            ha="center",
            va="center",
            fontsize=6,
            color="black",
            xytext=(0, 0),
            textcoords="offset points",
        )

    sns.despine(bottom=True)

    plt.xlabel("")
    plt.xticks([])

    plt.ylabel("")

    # plt.title(
    #     "Distribuição de Sentimentos de Avaliações no Glassdoor por Empresa",
    #     fontdict={
    #         # "weight": "bold",
    #         "size": ReportConfig.CHART_TITLE_FONT_SIZE,
    #     },
    #     loc="center",
    # )

    # plt.legend(
    #     title="Sentimento", labels=["Positivo", "Negativo", "Neutro"], loc="lower right"
    # )

    positive_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=ReportConfig.POSITIVE_SENTIMENT_COLOR
    )
    negative_patch = plt.Rectangle(
        (0, 0), 1, 1, fc=ReportConfig.NEGATIVE_SENTIMENT_COLOR
    )
    neutral_patch = plt.Rectangle((0, 0), 1, 1, fc=ReportConfig.NEUTRAL_SENTIMENT_COLOR)

    ax.legend(
        handles=[positive_patch, negative_patch, neutral_patch],
        labels=["Positivo", "Negativo", "Neutro"],
        title="Sentimento",
        bbox_to_anchor=(-0.3, 1),
        # loc="upper left",
    )

    st.pyplot(fig)


def sentiment_reviews_along_time():
    st.subheader("Número de Avaliações por Sentimento ao longo do tempo")

    st.markdown(
        """
    O gráfico destaca que as avaliações positivas superam consistentemente as negativas, enquanto as avaliações neutras são menos comuns.

    Observa-se um padrão notável de flutuações nos sentimentos ao longo do tempo. De 2014 a 2017, há uma tendência ascendente seguida por
    um declínio, repetindo-se de 2017 a 2020.

    Um aumento significativo na quantidade de avaliações de todas as emoções ocorre de 2020 a 2022, com pico em 2022. Enquanto ss avaliações
    neutras  passam por um declínio a partir de 2022, assim como as positivas, as avaliações negativas continuam a aumentar a partir deste período.
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    reviews_df["review_date"] = pd.to_datetime(reviews_df["review_date"])
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

    plt.xlabel("Year")
    plt.ylabel("Number of Reviews")
    plt.title("Number of Reviews by Sentiment over time")

    handles, labels = ax.get_legend_handles_labels()
    for i in range(len(ReportConfig.SENTIMENT_DICT)):
        handles[i]._label = ReportConfig.SENTIMENT_DICT[int(labels[i])]

    plt.legend(
        title="Sentiment",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    st.pyplot(fig)


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
        key="rating_star_company_input2",
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

    if len(filtered_df) > 0:
        sentiment_counts = (
            filtered_df.groupby(["company", "star_rating", "predicted_sentiment"])
            .size()
            .reset_index(name="count")
        )

        fig, ax = plt.subplots(1, figsize=(12, 8))

        bars = sns.barplot(
            data=sentiment_counts,
            x="star_rating",
            y="count",
            hue="predicted_sentiment",
            ax=ax,
            palette=ReportConfig.SENTIMENT_PALETTE,
        )

        for p in bars.patches:
            height = p.get_height()
            if math.isnan(height):
                height = 0.0

            bars.annotate(
                f"{int(height)}",
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va="center",
                fontsize=11,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        plt.title(
            "Sentiment Counts by Star Rating and Company",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        )
        plt.xlabel("Star Rating")
        plt.ylabel("Count")

        legend_labels = [
            ReportConfig.SENTIMENT_DICT[i]
            for i in range(len(ReportConfig.SENTIMENT_DICT))
        ]
        plt.legend(
            title="Sentiment",
            labels=legend_labels,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )

        leg = ax.get_legend()
        leg.legend_handles[0].set_color(ReportConfig.NEUTRAL_SENTIMENT_COLOR)
        leg.legend_handles[1].set_color(ReportConfig.POSITIVE_SENTIMENT_COLOR)
        leg.legend_handles[2].set_color(ReportConfig.NEGATIVE_SENTIMENT_COLOR)

        st.pyplot(fig)

        st.write("Avaliações filtradas")
        filtered_df = filtered_df.drop(labels="predicted_sentiment", axis=1)
        st.dataframe(filtered_df)
    else:
        st.error(
            AppMessages.ERROR_EMPTY_DATAFRAME,
            icon="🚨",
        )


def employee_role_analysis():
    st.subheader("Sentimentos das Avaliações por Empresa e por Cargo")

    st.markdown(
        """
    A visualização da distribuição de avaliações e emoções em todas as empresas permite uma comparação rápida e uma visão abrangente do panorama geral.
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

    with col2:
        sentiment = st.selectbox(
            label="Sentimento das Avaliações",
            options=("Positivo", "Negativo", "Neutro"),
            key="sentiment_input",
        )

    sentiment_key = get_sentiment_key_from_value(sentiment)
    company_df = reviews_df[reviews_df["company"] == company]

    filtered_df = company_df[company_df["predicted_sentiment"] == sentiment_key]

    if len(filtered_df) > 0:
        top_10_roles = filtered_df["employee_role"].value_counts().index[:10]
        filtered_df = filtered_df[filtered_df["employee_role"].isin(top_10_roles)][
            [
                "employee_role",
                "employee_detail",
                "review_text",
                "review_date",
                "star_rating",
            ]
        ]
        filtered_df.reset_index(drop=True, inplace=True)

        fig, ax = plt.subplots(1, figsize=(10, 8))
        sns.countplot(data=filtered_df, y="employee_role", order=top_10_roles, ax=ax)

        plt.title(
            f"Top 10 Employee Roles with Predicted Sentiment {ReportConfig.SENTIMENT_DICT[sentiment_key]} for {company}",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
        )
        plt.xlabel("Count")
        plt.ylabel("Employee Role")

        st.pyplot(fig)

        st.write("Avaliações filtradas")
        st.dataframe(filtered_df)
    else:
        st.error(
            AppMessages.ERROR_EMPTY_DATAFRAME,
            icon="🚨",
        )


def conclusion():
    st.subheader("Conclusão")

    st.markdown(
        """
    Com base nos resultados da análise das avaliações do Glassdoor, as empresas podem fortalecer pontos positivos identificados, abordar áreas de melhoria,
    personalizar estratégias de engajamento e monitorar continuamente o clima organizacional.

    Além disso, a análise revelou que algumas empresas foram classificadas
    como pertencentes à área de Tecnologia da Informação no Glassdoor, embora apresentem cargos não relacionados a TI, como *Classificador de Grãos* e *Comprador*,
    especialmente evidente na empresa **Amaggi**.

    Para trabalhos futuros, uma possibilidade seria treinar o modelo com avaliações específicas de cargos de TI, visando torná-lo mais especializado na identificação de
    pontos relevantes para classificação de sentimentos nessa área.
"""
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Home",
        page_icon=":house:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.sidebar.info(AppMessages.INFO_MENU_NAVIGATION)
    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.title(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
    )

    if "reviews_df" not in st.session_state:
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")
        reviews_df["sentiment"] = reviews_df["sentiment"].apply(
            lambda x: 2 if x == -1 else x
        )
        reviews_df["sentiment_label"] = reviews_df["predicted_sentiment"].map(
            ReportConfig.SENTIMENT_DICT
        )
        st.session_state["reviews_df"] = reviews_df

        neutral_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 0]
        st.session_state["neutral_reviews_df"] = neutral_reviews_df

        positive_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 1]
        st.session_state["positive_reviews_df"] = positive_reviews_df

        negative_reviews_df = reviews_df[reviews_df["predicted_sentiment"] == 2]
        st.session_state["negative_reviews_df"] = negative_reviews_df

    introduction()
    general_analysis()
    # company_analisys()
    company_analisys2()
    sentiment_reviews_along_time()
    # rating_star_analysis()
    # rating_star_analysis2()
    rating_star_analysis3()
    employee_role_analysis()
    conclusion()
