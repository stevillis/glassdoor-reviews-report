import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig
from utils import get_ranking_positive_negative_companies


def data_extraction():
    st.subheader("Extração de Dados")

    st.markdown(
        """
        Para obter os dados, foi utilizada a técnica de
        [Raspagem de dados](https://pt.wikipedia.org/wiki/Raspagem_de_dados)
        para baixar as páginas HTML de avaliações no Glassdoor das 22 empresas
        pré-selecionadas e, posteriormente, extrair dados relevantes desssas
        páginas, como **texto da avaliação**, **cargo do avaliador**,
        **quantidade de estrelas da avaliação**, etc. Ao final deste processo,
        **2532 avaliações** foram extraídas.
"""
    )


def data_preparation():
    st.subheader("Preparação dos Dados")

    st.markdown(
        """
    A partir dos dados extraídos, foi criado um **dataset** que consolidou
    todas as informações extraídas, onde cada avaliação foi classificada como
    **Positiva** ou **Negativa**. Essa classificação baseou-se na seção em que
    a avaliação estava localizada na página de avaliações do Glassdoor.
    Avaliações da seção *Prós* foram categorizadas como **Positivas**,
    enquanto aquelas da seção *Contras* foram consideradas **Negativas**.

    ##### Classificação das Avaliações

    Durante a [Análise Exploratória dos Dados](https://pt.wikipedia.org/wiki/An%C3%A1lise_explorat%C3%B3ria_de_dados),
    foi identificada a necessidade de criar uma nova classe de sentimento,
    pois alguns casos não se enquadravam nem como Positivos nem como Negativos.
    Assim, decidiu-se pela inclusão da classe **Neutra**.

    O dataset original foi, então, dividido em dois conjuntos de 1266
    avaliações: um contendo avaliações positivas e outro com avaliações
    negativas. O objetivo era aplicar um modelo pré-treinado de classificação
    de sentimentos para identificar as avaliações neutras. Para isso,
    utilizou-se o modelo pré-treinado [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned),
    que é capaz de classificar sentimentos em textos em diversos idiomas,
    incluindo o Português.

    Os resultados obtidos após a aplicação do modelo a cada um dos datasets
    mostraram que no conjunto de avaliações positivas, **650** foram
    classificadas de forma diferente da classificação original. No caso do
    conjunto de avaliações negativas, **821** receberam uma classificação
    distinta da original.

    ##### Anotação Manual de Dados
    Para corrigir as predições não positivas e não negativas, foi criada
    uma **ferramenta de anotação de sentimentos**. A ferramenta permite o
    carregamento do conjunto de dados a ser anotado, além de visualizar as
    avaliações com seu respectivo sentimento associado e o sentimento inferido
    pela predição, juntamente com o score associado. Além disso, a ferramenta
    permite realizar o download do conjunto de dados com as anotações
    corrigidas. A ferramenta de anotação criada é exibida a seguir:

    <img src="https://github.com/stevillis/glassdoor-reviews-analysis-nlp/raw/master/data_preparation/annotation_tool_preview.png" alt="Ferramenta de Anotação de Sentimentos" width="600"/>

    <br/>
    <br/>

    A correção manual das 650 avaliações não positivas preditas pelo modelo
    revelou que 33 avaliações eram neutras e 15 eram negativas. Já na correção
    das 821 avaliações não negativas inferidas, foram classificadas 247
    avaliações como neutras, 49 como positivas e 17 como negativas. Após a
    correção manual das predições, estas foram combinadas ao conjunto de dados
    original.

    O gráfico a seguir mostra a distribuição de sentimentos das avaliações após
    a anotação dos dados.
    """,
        unsafe_allow_html=True,
    )

    reviews_df = st.session_state.get("reviews_df")

    fig, ax = plt.subplots(figsize=(8, 4))

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
        ax=ax,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_width())}",
            (p.get_width(), p.get_y() + p.get_height() / 2.0),
            ha="center",
            va="center",
            fontsize=11,
            color="white",
            xytext=(-15, 0),
            textcoords="offset points",
        )

    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(
        "Distribuição de sentimentos anotados",
        loc="center",
        fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
    )

    plt.tight_layout()
    st.pyplot(fig)


def model_development():
    st.subheader("Desenvolvimento do Modelo")

    st.markdown(
        """
    ##### Tratamento do desequilíbrio de classes

    Ao analisar o conjunto de dados anotados, observou-se um desequilíbrio
    significativo entre as classes de sentimento. Avaliações classificadas
    como Neutro representavam quase 5 vezes menos do que as demais classes.
    Para lidar com esse problema, foi aplicada a técnica de [Oversampling](https://www.escoladnc.com.br/blog/entendendo-oversampling-tecnicas-e-metodos-para-balanceamento-de-dados/#:~:text=O%20que%20%C3%A9%20Oversampling%3F,cada%20classe%20durante%20o%20treinamento.),
    na classe Neutro, **replicando aleatoriamente 3 vezes o número de amostras**
    dessa classe durante o treinamento. Isso ajudou a balancear a distribuição
    das classes e melhorar o desempenho do modelo na identificação correta de
    avaliações neutras.

    ##### Arquitetura do Modelo

    Para identificar a melhor configuração na classificação das três
    classes de sentimento (Neutro, Positivo e Negativo), diversas
    abordagens foram testadas. As configurações incluíram:
    - Modelo sem congelamento das camadas do BERTimbau.
    - Modelo com congelamento das camadas do BERTimbau.
    - Oversampling sem congelamento do BERTimbau.
    - Oversampling com congelamento do BERTimbau.

    Dentre todas as configurações testadas, a combinação que apresentou
    melhor [F1-Score](https://pt.wikipedia.org/wiki/Precis%C3%A3o_e_revoca%C3%A7%C3%A3o#F-measure)
    em todas as classes foi a do modelo com **Oversampling e congelamento das
    camadas do BERTimbau**.

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

    ##### Treinamento Modelo

    O modelo foi treinado utilizando 80% dos dados, enquanto os 20% restantes
    foram reservados para testes. Os parâmetros de treinamento foram os
    seguintes:
    - **Épocas**: 5
    - **Batch Size**: 16
    - **Learning Ratio**: $2\\times 10^{-5}$
    - **Loss Function**: CrossEntropyLoss
    - **Otimizador**: Adam

    O gráfico de perda nos dados de treino e de teste mostram que na
    segunda época o modelo se saiu melhor nos dados de teste, mas no
    restante das épocas, essa perda aumentou levemente. Ao analisar
    a evolução de perda nos dados de treino, é posssível observar que
    o Modelo praticamente decorou os dados de treino.

    <img src="https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/train_and_test_loss.png?raw=true" alt="Loss de treinamento e teste ao longo das épocas" width="600"/>

    ##### Métricas do Modelo

    As métricas do modelo após as 5 épocas de treinamento foram:
    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | Neutro       | 0.97      | 0.98   | 0.98     | 197     |
    | Positivo     | 0.95      | 0.98   | 0.96     | 256     |
    | Negativo     | 0.98      | 0.93   | 0.96     | 199     |
    | accuracy     |           |        | 0.97     | 652     |
    | macro avg    | 0.97      | 0.97   | 0.97     | 652     |
    | weighted avg | 0.97      | 0.97   | 0.97     | 652     |

    <br/>

    A [Matriz de Confusão](https://pt.wikipedia.org/wiki/Matriz_de_confus%C3%A3o#:~:text=Em%20an%C3%A1lise%20preditiva%2C%20a%20matriz,verdadeiros%20positivos%20e%20verdadeiros%20negativos%20.)
    das predições do Modelo nos dados de teste após as 5 épocas é mostrada a 
    seguir:

    <img src="https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/confusion_matrix.png?raw=true" alt="Matriz de Confusão" width="600"/>
    """,
        unsafe_allow_html=True,
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

    data_extraction()
    data_preparation()
    model_development()
