import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import load_reviews_df, set_companies_raking_to_session


def data_extraction():
    st.subheader("Extração de Dados")

    st.markdown(
        """
        Para obter os dados, foi utilizada a técnica de
        [Raspagem de dados](https://pt.wikipedia.org/wiki/Raspagem_de_dados)
        para baixar as páginas HTML de avaliações no Glassdoor das 22 empresas
        pré-selecionadas e, posteriormente, extrair dados relevantes dessas
        páginas, como **texto da avaliação**, **cargo do avaliador**,
        **quantidade de estrelas da avaliação**, etc. Ao final deste processo,
        **2532 avaliações** com data entre **05 de outubro de
        2014 e 16 de março de 2024** foram extraídas.
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
    pois alguns casos não se enquadravam nem como Positivos, nem como
    Negativos. Assim, decidiu-se pela inclusão da classe **Neutra**.

    O dataset original foi então dividido em dois conjuntos de 1266
    avaliações: um composto somente por avaliações positivas e outro com
    apenas avaliações negativas. Essa divisão teve como objetivo utilizar um
    modelo pré-treinado de classificação de sentimentos para identificar as
    avaliações neutras.

    Para isso, utilizou-se o modelo pré-treinado [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned),
    que é capaz de classificar sentimentos em textos em diversos idiomas,
    incluindo o português.

    Os resultados obtidos após a aplicação do modelo a cada um dos datasets
    mostraram que no conjunto de avaliações positivas, **650 foram
    classificadas de forma diferente da classificação original**. No caso do
    conjunto de avaliações negativas, **821 receberam uma classificação
    distinta da original**.

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
    revelou que **33 avaliações eram neutras e 15 eram negativas**. Já na
    correção das 821 avaliações não negativas inferidas, foram classificadas
    **209 avaliações como neutras, 49 como positivas e 25 como negativas**.
    Após a correção manual das predições, estas foram combinadas ao conjunto
    de dados original.

    O gráfico a seguir mostra a distribuição de sentimentos das avaliações após
    a anotação dos dados.
    """,
        unsafe_allow_html=True,
    )

    reviews_df = load_reviews_df()

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

    st.write(
        """
    Ao final da anotação, obteve-se um dataset com as informações necessárias
    para treinar um Modelo para classificar o sentimento das avaliações. As 5
    primeiras linhas do dataset obtido são apresentadas a seguir:
    """
    )

    reviews_df = load_reviews_df()
    reviews_df.drop(
        labels=[
            "sentiment_label",
            "sentiment_plot",
            "role_group",
        ],
        axis=1,
        inplace=True,
    )

    st.dataframe(reviews_df.head())

    st.write(
        """
    Todas as colunas, exceto `annotated` e `predicted_sentiment` foram
    extraídas do Glassdoor.
    - `predicted_sentiment`: representa o sentimento inferido pelo modelo
    pré-treinado [citizenlab/twitter-xlm-roberta-base-sentiment-finetunned](https://huggingface.co/citizenlab/twitter-xlm-roberta-base-sentiment-finetunned).

    - `annotated`: indica que, durante a anotação manual, houve divergência
    entre a classificação do modelo pré-treinado e o sentimento original, e
    que este foi corrigido.
    """
    )


def model_development():
    st.subheader("Desenvolvimento dos Modelos")

    st.markdown(
        """
    ##### Tratamento do desequilíbrio de classes

    Ao analisar o conjunto de dados anotados, observou-se um desequilíbrio
    significativo entre as classes de sentimento. **Avaliações classificadas
    como Neutro representavam quase 5 vezes menos do que as demais classes**.
    Para lidar com esse problema, foi aplicada a técnica de [Oversampling](https://www.escoladnc.com.br/blog/entendendo-oversampling-tecnicas-e-metodos-para-balanceamento-de-dados/#:~:text=O%20que%20%C3%A9%20Oversampling%3F,cada%20classe%20durante%20o%20treinamento.),
    na classe Neutro, **replicando aleatoriamente 3 vezes o número de amostras**
    dessa classe durante o treinamento. Isso ajudou a balancear a distribuição
    das classes e melhorar o desempenho do modelo na identificação correta de
    avaliações neutras.

    ##### Arquitetura

    Para identificar a melhor arquitetura de modelo que alcançasse a maior
    média macro F1 nas três classes de sentimento (Positivo, Negativo e Neutro),
    foram treinados três modelos com diferentes configurações:
    - **Modelo A**: Este modelo possui três camadas ocultas com 300, 100 e 50
    neurônios, respectivamente, sem o uso de dropout.
    - **Modelo B**: Similar ao Modelo A, mas com dropout de 20% aplicado nas
    camadas ocultas.
    - **Modelo C**: Este modelo possui apenas uma camada oculta com 50
    neurônios, sem o uso de dropout.

    Cada um dos modelos tem a seguinte arquitetura:
    - **Camada de entrada**: Conectada ao BERTimbau.
    - **Camadas ocultas**: Varia conforme o modelo.
    - **Camada de saída**: A última camada oculta é conectada a uma função
    Softmax, que classifica a entrada em uma das três classes de sentimento.

    A figura a seguir mostra a Arquitetura do **Modelo A** como exemplo.

    ![Arquitetura do modelo](https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/arquitetura_do_modelo.png?raw=true "Arquitetura do Modelo")

    ##### Treinamento

    Os modelos A, B e C foram treinados utilizando as seguintes configurações:

    - **Divisão dos dados**: 80% para treino e 20% para teste.
    - **Estratégias de treinamento**:
    - Sem oversampling:
        - Com congelamento das camadas do BERTimbau.
        - Sem congelamento das camadas do BERTimbau.
    - Com oversampling:
        - Com congelamento das camadas do BERTimbau.
        - Sem congelamento das camadas do BERTimbau.
    - **Parâmetros de treinamento**:
    - Número de épocas: 5
    - Tamanho do batch: 16
    - Taxa de aprendizado: $2\\times 10^{-5}$
    - Função de perda: CrossEntropyLoss
    - Otimizador: Adam

    **Seleção dos modelos com melhor médica macro**

    Após o treinamento em todas as configurações mencionadas, foram
    selecionados os modelos que apresentaram a maior média macro F1 nos
    dados de teste:

    - **Modelo A**: com oversampling e congelamento das camadas do BERTimbau.
    - **Modelo B**: com oversampling e congelamento das camadas do BERTimbau.
    - **Modelo C**: com oversampling e sem congelamento das camadas do
    BERTimbau.

    O gráfico a seguir mostra a evolução de erro (loss) de treino e teste ao
    longo das 5 épocas para os três modelos selecionados. Esse gráfico mostra
    que o erro de treino diminuiu consistentemente, indicando que os modelos
    aprenderam os padrões dos dados. Os erros de teste, embora ligeiramente
    maiores que os de treino, indicam que os modelos conseguiram generalizar o
    aprendizado.
    """,
        unsafe_allow_html=True,
    )

    st.image(
        image="https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/train_and_test_loss_models.png?raw=true",
        caption="Loss de treinamento e teste ao longo das épocas",
    )

    st.write(
        """
    ##### Métricas dos Modelos nos dados de teste

    **Matriz de Confusão**

    A [Matriz de Confusão](https://pt.wikipedia.org/wiki/Matriz_de_confus%C3%A3o#:~:text=Em%20an%C3%A1lise%20preditiva%2C%20a%20matriz,verdadeiros%20positivos%20e%20verdadeiros%20negativos%20.)
    de cada modelo mostra que o Modelo A foi o que obteve melhor desempenho:
    - Das 197 avaliações Neutras, o modelo errou 4.
    - Das 256 avaliações Positivas, o modelo errou 5.
    - Das 199 avaliações Negativas, o modelo errou 13.
    """,
        unsafe_allow_html=True,
    )

    st.image(
        image="https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/confusion_matrices.png?raw=true",
        caption="Matriz de Confusão de cada modelo",
    )

    st.write(
        """
    **Métricas**

    Como mostrado a seguir, o Modelo A foi o que obteve melhor desempenho,
    alcançando 0.97 de acurácia, enquanto os modelos B e C tiveram desempenho
    similar, alcançando 0.96 de acurácia.

    *Modelo A*

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | Neutro       | 0.97      | 0.98   | 0.98     | 197     |
    | Positivo     | 0.95      | 0.98   | 0.96     | 256     |
    | Negativo     | 0.98      | 0.93   | 0.96     | 199     |
    | accuracy     |           |        | 0.97     | 652     |
    | macro avg    | 0.97      | 0.97   | 0.97     | 652     |
    | weighted avg | 0.97      | 0.97   | 0.97     | 652     |

    <br/>

    *Modelo B*

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | Neutro       | 0.95      | 0.98   | 0.96     | 197     |
    | Positivo     | 0.95      | 0.95   | 0.95     | 256     |
    | Negativo     | 0.97      | 0.93   | 0.95     | 199     |
    | accuracy     |           |        | 0.96     | 652     |
    | macro avg    | 0.96      | 0.96   | 0.96     | 652     |
    | weighted avg | 0.96      | 0.96   | 0.96     | 652     |

    <br/>

    *Modelo C*

    |              | precision | recall | f1-score | support |
    | ------------ | --------- | ------ | -------- | ------- |
    | Neutro       | 0.96      | 0.98   | 0.97     | 197     |
    | Positivo     | 0.96      | 0.95   | 0.95     | 256     |
    | Negativo     | 0.95      | 0.96   | 0.95     | 199     |
    | accuracy     |           |        | 0.96     | 652     |
    | macro avg    | 0.96      | 0.96   | 0.96     | 652     |
    | weighted avg | 0.96      | 0.96   | 0.96     | 652     |

    <br/>

    ##### Métricas dos modelos nos dados de validação

    Um conjunto de dados para validação foi elaborado com avaliações
    totalmente anotadas manualmente, selecionando aleatoriamente 246
    avaliações das empresas *TOTVS* e *Sankhya Gestão de Negócios*,
    distribuídas igualmente em 82 avaliações para cada classe de sentimento.

    Os modelos A, B e C foram aplicados a esse conjunto de validação para
    comparar seu desempenho com o do modelo
    citizenlab/Twitter-XLM-RoBERTa-base-sentiment-finetuned, utilizado para
    auxiliar na anotação de avaliações neutras e denominado aqui como Modelo X.

    Conforme ilustrado no gráfico a seguir, todos os modelos treinados
    superaram o desempenho do Modelo X, o que era esperado, uma vez que
    [estudos](https://mapmeld.medium.com/a-whole-world-of-bert-f20d6bd47b2f#8a7b)
    indicam que modelos BERT treinados em uma língua específica
    apresentam melhor performance do que modelos multilíngues em algumas
    tarefas de NLP. Além disso, observa-se que o Modelo C obteve o segundo
    melhor desempenho nos dados de validação, mesmo utilizando apenas uma
    camada oculta com 50 neurônios.
    Isso evidencia que o uso do BERTimbau como base para análise de
    sentimentos é eficiente, dispensando a necessidade de camadas ocultas
    volumosas.
    """,
        unsafe_allow_html=True,
    )

    st.image(
        image="https://github.com/stevillis/glassdoor-reviews-report/blob/master/img/models_performance_comparison.png?raw=true",
        caption="Performance dos modelos nos dados de validação",
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

    st.sidebar.warning(AppMessages.WARNING_PLOT_NOT_WORKING)

    st.header(
        """
        Análise de sentimento em avaliações no Glassdoor: Um estudo sobre empresas de Tecnologia da Informação em Cuiabá
    """
    )

    reviews_df = load_reviews_df()
    set_companies_raking_to_session(reviews_df)

    data_extraction()
    data_preparation()
    model_development()
