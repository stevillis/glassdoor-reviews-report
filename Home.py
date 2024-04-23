import warnings

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig
from utils import get_sentiment_key_from_value


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
    st.subheader("Análise Geral das Avaliações")

    st.markdown(
        """
    Inicialmente, uma análise geral das avaliações foi realizada, totalizando um conjunto de 2532 análises.
    O Modelo de IA treinado identificou 1257 avaliações positivas, destacando os aspectos mais elogiados pelas equipes.
    Por outro lado, foram identificadas 1052 avaliações negativas, sugerindo áreas específicas que podem ser aprimoradas.
    Além disso, o Modelo classificou 223 avaliações como neutras, fornecendo uma perspectiva equilibrada e imparcial.
"""
    )

    reviews_df = st.session_state.get("reviews_df")

    sentiment_counts = reviews_df["predicted_sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["predicted_sentiment", "count"]

    fig, ax = plt.subplots(1, figsize=(10, 6))

    sns.barplot(
        data=sentiment_counts,
        x=sentiment_counts.index,
        y="count",
        palette=sns.color_palette(sns.color_palette(), n_colors=3),
        ax=ax,
    )

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="center",
            fontsize=11,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    plt.xlabel("Sentiment")
    plt.ylabel("Count")

    ax.set_xticklabels(
        [
            ReportConfig.SENTIMENT_DICT[sentiment]
            for sentiment in sentiment_counts["predicted_sentiment"]
        ]
    )

    plt.title("Sentiment Distribution")

    st.pyplot(fig)


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

    plt.title("Predicted Sentiment by Company")
    plt.xlabel("Company")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Predicted Sentiment", labels=ReportConfig.SENTIMENT_DICT.values())

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

    plt.title("Sentiment Counts by Star Rating and Company")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")

    plt.xticks(ticks=sorted(reviews_df["star_rating"].unique()))

    handles, labels = ax.get_legend_handles_labels()
    handles[0].set_color(["#ff7f0e"])
    handles[1].set_color(["#2ca02c"])
    handles[2].set_color(["#1f77b4"])

    legend_labels = [ReportConfig.SENTIMENT_DICT[i] for i in range(3)]

    plt.legend(
        title="Sentiment",
        labels=legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    st.pyplot(fig)

    st.write("Avaliações filtradas")
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
        ]
    ]
    filtered_df.reset_index(drop=True, inplace=True)

    sentiment_counts = (
        filtered_df.groupby(["company", "star_rating", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(12, 8))

    sns.barplot(
        data=sentiment_counts,
        x="star_rating",
        y="count",
        hue="predicted_sentiment",
        ax=ax,
        palette=sns.color_palette(),
    )

    plt.title("Sentiment Counts by Star Rating and Company")
    plt.xlabel("Star Rating")
    plt.ylabel("Count")

    legend_labels = [ReportConfig.SENTIMENT_DICT[i] for i in range(3)]
    plt.legend(
        title="Sentiment",
        labels=legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )

    leg = ax.get_legend()
    leg.legendHandles[0].set_color("#3274a1")
    leg.legendHandles[1].set_color("#e1812c")
    leg.legendHandles[2].set_color("#3a923a")

    st.pyplot(fig)

    st.write("Avaliações filtradas")
    st.dataframe(filtered_df)


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
            options=("Positive", "Negative", "Neutral"),
            key="sentiment_input",
        )

    sentiment_key = get_sentiment_key_from_value(sentiment)
    company_df = reviews_df[reviews_df["company"] == company]

    filtered_df = company_df[company_df["predicted_sentiment"] == sentiment_key]

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
        f"Top 10 Employee Roles with Predicted Sentiment {ReportConfig.SENTIMENT_DICT[sentiment_key]} for {company}"
    )
    plt.xlabel("Count")
    plt.ylabel("Employee Role")

    st.pyplot(fig)

    st.write("Avaliações filtradas")
    st.dataframe(filtered_df)


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
    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.title(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
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

    introduction()
    general_analysis()
    company_analisys()
    # rating_star_analysis()
    # rating_star_analysis2()
    rating_star_analysis3()
    employee_role_analysis()
    conclusion()
