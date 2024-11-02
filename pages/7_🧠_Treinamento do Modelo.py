import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig
from utils import get_ranking_positive_negative_companies


def model_implementation():
    st.subheader("Treinamento do Modelo")

    st.markdown(
        """
       **Metodologia**

        Antes de treinar o modelo de aprendizado de máquina para classificar
        os sentimentos das avaliações extraídas no Glassdoor, foi necessário
        preparar os dados. Essa preparação envolveu:


        *Extrair as avaliações do Glassdoor*

        Para isto, criou-se um **scraper** para baixar as páginas HTML de
        avaliações no Glassdoor das empresas pré-selecionadas e,
        posteriormente, extrair dados relevantes desssas páginas, como
        **texto da avaliação**, **cargo do avaliador**, **quantidade de
        estrelas da avaliação**, etc.

        A partir destes dados extraídos, criou-se um
        dataset com todas as informações extraídas, onde cada avaliação
        foi classificada como **Positiva** ou **Negativa**, de acordo com a
        seção onde esta se encontrava na página, onde avaliações da seção
        **Prós** foram classificadas como **Positivas** e avaliações da seção
        **Contras** foram classificadas como **Negativas**.

        *Classificação manual das avaliações*

        Durante a **Análise Exploratória dos Dados**, identificou-se a
        necessidade de criação de uma nova classe de sentimento para as
        avaliações, pois haviam casos onde nem o sentimento Positivo ou o
        Negativo eram apropriados. Assim, optou-se pela criação da classe
        **Neutro**.

        Com o auxílio do modelo pré-treinado
        [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned),
        que classifica sentimentos em texto de diversos idiomas, incluindo o
        Português, foi possível classificar as avaliações com sentimento
        Neutro. Posteriormente, essas avaliações Neutras precisaram ser
        revisadas  manualmente, o que foi feito com o auxílio de uma
        ferramenta de anotação de dados criada pelo próprio autor.

        *Tratamento do desequilíbrio de classes*

        Ao analisar o conjunto de dados anotados, observou-se um desequilíbrio
        significativo entre as classes de sentimento. Avaliações classificadas
        como Neutro representavam quase 5 vezes menos do que as demais classes
        (Positivo e Negativo). Para lidar com esse problema, foi aplicada a
        técnica de oversampling na classe Neutro, replicando aleatoriamente
        algumas amostras dessa classe durante o treinamento. Isso ajudou a
        balancear a distribuição das classes e melhorar o desempenho do modelo
        na identificação correta de avaliações neutras.

        **Resultados**

        *Arquitetura do Modelo*

        Para identificar a melhor configuração na classificação das três
        classes de sentimento (Neutro, Positivo e Negativo), diversas
        abordagens foram testadas. As configurações incluem:
        - Modelo sem congelamento das camadas do BERTimbau.
        - Modelo com congelamento das camadas do BERTimbau.
        - Oversampling sem congelamento do BERTimbau.
        - Oversampling com congelamento do BERTimbau.

        Dentre todas as configurações testadas, a combinação que apresentou a
        melhor acurácia foi a do modelo com Oversampling e congelamento das
        camadas do BERTimbau.

        A arquitetura do modelo consiste em:
        - Camada de Entrada: Conectada ao BERTimbau com suas camadas
        congeladas.
        - Camadas Ocultas:
            - Primeira camada oculta com 300 neurônios.
            - Segunda camada oculta com 100 neurônios.
            - Terceira camada oculta com 50 neurônios.

        A última camada oculta é conectada a uma função Softmax, que
        classifica a entrada em uma das três classes de sentimento: Neutro,
        Positivo ou Negativo.

        ![Arquitetura do modelo](https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/arquitetura_do_modelo.png?raw=true "Arquitetura do Modelo")

        *Treinamento do Modelo*

        O modelo foi treinado utilizando 80% dos dados disponíveis, enquanto
        os 20% restantes foram reservados para testes. A tabela a seguir
        apresenta as métricas de desempenho do modelo treinado.

        |              | precision | recall | f1-score | support |
        | ------------ | --------- | ------ | -------- | ------- |
        | Neutro       | 0.96      | 0.98   | 0.97     | 197     |
        | Positivo     | 0.92      | 0.98   | 0.95     | 256     |
        | Negativo     | 0.98      | 0.88   | 0.93     | 199     |
        | accuracy     |           |        | 0.95     | 652     |
        | macro avg    | 0.96      | 0.95   | 0.95     | 652     |
        | weighted avg | 0.95      | 0.95   | 0.95     | 652     |

        $~$

        *Comparação entre dados anotados e classificados pelo modelo*

        As barras do gráfico são divididas em duas categorias: uma
        representando os dados anotados manualmente e a outra representando as
        previsões do modelo.
        - Classificação Positiva: O modelo identificou 1257 avaliações como
        positivas, o que é apenas 12 a menos do que a anotação manual. Isso
        indica uma alta precisão na detecção de sentimentos positivos.
        - Classificação Negativa: O modelo classificou 1052 avaliações como
        negativas, superando a anotação manual em 31 casos. Essa leve
        discrepância sugere que o modelo pode estar identificando um número
        maior de sentimentos negativos do que realmente existe nos dados
        anotados.
        - Classificação Neutra: O modelo identificou 223 avaliações como
        neutras, o que representa uma diferença de 19 casos a menos em
        comparação com as anotações manuais. Essa discrepância evidencia a
        conhecida dificuldade do modelo em reconhecer sentimentos neutros,
        atribuída ao desbalanceamento em relação às classes positivas e
        negativas.

            Entretanto, a aplicação da técnica de Oversampling demonstrou ser
            eficaz, uma vez que, sem essa abordagem, o modelo apresentava
            dificuldades significativas em identificar as classes neutras
            durante o treinamento.
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
            (p.get_width(), p.get_y() + p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            xytext=(-15, 0),
            textcoords="offset points",
        )

    ax[0].set_xticks([])
    ax[0].set_xlabel("")
    ax[0].set_ylabel("")
    ax[0].set_title(
        "Distribuição de sentimentos anotados",
        loc="center",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
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
        "Distribuição de sentimentos classificados pelo modelo",
        loc="center",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
    )

    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Avaliação Geral do Modelo")

    st.write(
        """
        *Matriz de Confusão*

        A matriz de confusão é uma ferramenta fundamental na avaliação do
        desempenho de modelos de classificação, permitindo uma análise
        detalhada dos acertos e erros do modelo em relação às classes reais.

        ![Matriz de Confusão](https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/confusion_matrix.png?raw=true "Matriz de Confusão")

        A matriz de confusão apresentada mostra o desempenho geral do modelo,
        permitindo uma interpretação clara dos resultados obtidos em relação
        aos sentimentos classificados:

        - Sentimento Neutro (Linha 1):
            - O modelo **corretamente** previu **216** avaliações como neutras.
            - O modelo incorretamente previu 9 avaliações como positivas.
            - O modelo incorretamente previu 17 avaliações como negativas.

        - Sentimento Positivo (Linha 2):
            - O modelo incorretamente previu 5 avaliações como neutras.
            - O modelo **corretamente** previu **1246** avaliações como
            positivas.
            - O modelo incorretamente previu 18 avaliações como negativas.

        - Sentimento Negativo (Linha 3):
            - O modelo incorretamente previu 2 avaliações como neutras.
            - O modelo incorretamente previu 2 avaliações como positivas.
            - O modelo **corretamente** previu **1017** avaliações como
            negativas.

        À partir da Matriz de Confusão, foram calculadas as métricas do modelo,
        apresentadas as seguir, que evidenciam sua eficácia na classificação
        de sentimentos, destacando especialmente a acurácia, que superou os
        resultados obtidos durante a fase de treinamento. Essa melhoria indica
        um desempenho robusto e confiável do modelo em situações reais.
        |            | precision | recall | accuracy | f1-score |
        |------------|-----------|--------|----------|----------|
        | Neutro     | 0.97      | 0.89   | 0.99     | 0.93     |
        | Positivo   | 0.99      | 0.98   | 0.99     | 0.99     |
        | Negativo   | 0.97      | 1.00   | 0.98     | 0.98     |
    """
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Treinamento do Modelo",
        page_icon=":brain:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.header(
        """
        Análise de sentimentos nas avaliações do Glassdoor: Um estudo sobre empresas de Tecnologia em Cuiabá
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

    # TODO: refact model implementation description
    # TODO: add metrics for each class
    # TODO: replace general confusion matrix and to confusion matrix of each class
    # TODO: make a table comparing the four model configuration used in model development
    model_implementation()
