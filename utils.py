from string import punctuation
from typing import List

import nltk
import numpy as np
import pandas as pd
import streamlit as st

from report_config import ReportConfig

nltk.download("stopwords")

REVIEWS_DF_PATH = "./glassdoor_reviews_predicted.csv"

MIN_REVIEWS = 42 / 2  # (reviews median) / 2

STOPWORDS = [word.lower() for word in nltk.corpus.stopwords.words("portuguese")]
STOPWORDS.extend(
    [
        # "empresa",
    ]
)

IT_ROLES = [
    "ANALISTA SAP",
    "Administrador De Redes",
    "Analista De BI",
    "Analista De Conversão De Dados",
    "Analista De Desenvolvimento Júnior",
    "Analista De Desenvolvimento Pleno",
    "Analista De Implantação Pleno",
    "Analista De Implantação Sênior",
    "Analista De Infraestrutura Pleno",
    "Analista De Negócios",
    "Analista De Negócios De TI Pleno",
    "Analista De Negócios Júnior",
    "Analista De Negócios Pleno",
    "Analista De Negócios Sênior",
    "Analista De Projetos II",
    "Analista De Projetos Pleno",
    "Analista De Qualidade Pleno",
    "Analista De Redes",
    "Analista De Redes Júnior",
    "Analista De Requisitos",
    "Analista De Requisitos Pleno",
    "Analista De Requisitos Sênior",
    "Analista De Sistemas",
    "Analista De Sistemas Desenvolvedor",
    "Analista De Sistemas Júnior",
    "Analista De Sistemas N2",
    "Analista De Sistemas Pleno",
    "Analista De Sistemas Sênior",
    "Analista De Sistemas Sênior I",
    "Analista De Suporte",
    "Analista De Suporte De Sistemas",
    "Analista De Suporte E Infraestrutura Em TI",
    "Analista De Suporte I",
    "Analista De Suporte Júnior",
    "Analista De Suporte Pleno",
    "Analista De Suporte Técnico",
    "Analista De TI II",
    "Analista De TI Pleno",
    "Analista De TI Sênior",
    "Analista De Tecnologia Da Informação (TI)",
    "Analista De Teste",
    "Analista De Teste Junior",
    "Analista De Testes",
    "Analista De Testes Pleno",
    "Analista De Testes Sênior",
    "Analista Desenvolvedor",
    "Analista Desenvolvedor .NET",
    "Analista Desenvolvedor .NET Sênior",
    "Analista Desenvolvedor De Sistemas",
    "Analista Desenvolvedor Pleno",
    "Analista Desenvolvedor Sênior",
    "Analista Programador",
    "Analista Service Desk",
    "Analista de Infraestrutura e Suporte",
    "Analista de Negócios",
    "Analista de Negócios Pleno",
    "Analista de Sistemas Júnior",
    "Analista de Sistemas Sênior",
    "Analista de Suporte Técnico",
    "Arquiteto Desenvolvedor Java Sênior",
    "Auxiliar De Analista De Sistemas",
    "Business Analyst",
    "Cientista De Dados",
    "Consultor De Implantação De Sistemas",
    "Consultor De Implantação Júnior",
    "Consultor De Implementação Pleno",
    "Consultor De Suporte Técnico",
    "Consultor SAP",
    "Consultor SAP B1",
    "Consultor SAP Júnior",
    "Consultor TI",
    "Coordenador De Desenvolvimento De Software",
    "Coordenador De Service Desk",
    "Desenvolvedor",
    "Desenvolvedor C#",
    "Desenvolvedor C++ Pleno",
    "Desenvolvedor De Programas",
    "Desenvolvedor De Sistemas",
    "Desenvolvedor De Software",
    "Desenvolvedor De Software Pleno",
    "Desenvolvedor Front End Angular Junior",
    "Desenvolvedor Genexus",
    "Desenvolvedor Júnior",
    "Desenvolvedor Júnior I",
    "Desenvolvedor Pleno",
    "Desenvolvedor React Native",
    "Desenvolvedor Sênior",
    "Desenvolvedor de Java Senior - PJ",
    "Desenvolvedor.NET",
    "Especialista De Sistemas",
    "Especialista Em Sistemas",
    "Estagiário De Desenvolvimento",
    "Estagiário De TI",
    "Estágio Desenvolvimento Java",
    "Gerente De Projetos De TI",
    "Gerente De Serviços De TI",
    "Gerente De TI",
    "IT Consultant",
    "Information Technology Analyst",
    "Product Owner",
    "Programador",
    "Programador De Sistemas Pleno",
    "Programador Delphi",
    "Programador I Júnior",
    "Programador Júnior",
    "Programador Pleno",
    "Programador Pleno V",
    "Programador Sênior",
    "Programador Trainee",
    "Programador treinee",
    "QA Analyst",
    "QAssurance",
    "Quality Assurance",
    "Senior Analyst Developer",
    "Software Developer",
    "Sofware Developer",
    "Supervisor De Infraestrutura De TI",
    "Suporte Técnico",
    "Suporte técnico",
    "System Analyst",
    "Technical Support Engineer",
    "Técnico De Informática",
    "Técnico De Informática II",
    "Técnico De Suporte",
    "Técnico De Suporte II",
    "Técnico De Suporte Júnior",
    "Técnico Em Informática",
    "Técnico Em Sistemas Da Informação",
    "Técnico Em Suporte Técnico",
    "Técnico Em Suporte Técnico N2",
    "Técnico Em Suporte Técnico Pleno",
    "Técnico Suporte N2",
    "Técnico suporte informática",
    "UX Designer",
    "Web Designer",
    "dev junior 2",
]

ROLE_GROUPS = {0: "Outros", 1: "Profissionais de TI", 2: "Funcionário confidencial"}

# Translation table for replacing any special character from a word.
TRANSLATION_TABLE_SPECIAL_CHARACTERS = str.maketrans("", "", punctuation)


@st.cache_data
def load_reviews_df() -> pd.DataFrame:
    reviews_df = pd.read_csv(REVIEWS_DF_PATH)
    reviews_df = create_role_group(reviews_df)

    return reviews_df


def set_companies_raking_to_session(reviews_df):
    # Top Companies Reviews DF
    if "top_positive_companies_df" not in st.session_state:
        top_positive_companies_df, top_negative_companies_df = (
            get_ranking_positive_negative_companies(reviews_df)
        )

        st.session_state["top_positive_companies_df"] = top_positive_companies_df
        st.session_state["top_negative_companies_df"] = top_negative_companies_df


def get_sentiment_key_from_value(value):
    key_list = list(ReportConfig.SENTIMENT_DICT.keys())
    val_list = list(ReportConfig.SENTIMENT_DICT.values())

    position = val_list.index(value)
    return key_list[position]


def get_role_group_keys_from_values(values):
    key_list = list(ROLE_GROUPS.keys())
    val_list = list(ROLE_GROUPS.values())

    print(values)
    print(key_list)
    print(val_list)
    role_group_keys = []
    for value in values:
        position = val_list.index(value)
        role_group_keys.append(key_list[position])

    return role_group_keys


def get_good_rating_companies(df: pd.DataFrame) -> list:
    """
    Get companies with more positive ratings than negative ratings.

    Parameters:
    df (pandas.DataFrame): Input DataFrame containing ratings data.

    Returns:
    list: List of company names with more positive ratings than negative ratings.
    """
    positive_greater_than_negative = df[df["1positive"] > df["2negative"]]

    positive_greater_than_negative = positive_greater_than_negative.sort_values(
        by="sentiment_diff", ascending=False
    ).reset_index()

    return list(positive_greater_than_negative["company"].values)


def get_bad_rating_companies(df: pd.DataFrame) -> list:
    """
    Get companies with more negative ratings than positive ratings.

    Parameters:
    - df (DataFrame): Input DataFrame containing ratings data.

    Returns:
    - List: List of company names with more negative ratings than positive ratings.
    """
    negative_greater_than_positive = df[df["2negative"] > df["1positive"]]

    negative_greater_than_positive = negative_greater_than_positive.sort_values(
        by="sentiment_diff", ascending=False
    ).reset_index()

    return list(negative_greater_than_positive["company"].values)


def get_neutral_rating_companies(df: pd.DataFrame) -> list:
    """
    Get companies where positive and negative rantings are the same.

    Parameters:
    - df (DataFrame): Input DataFrame containing ratings.

    Returns:
    - List: List of companies  where positive and negative rantings are the same.
    """
    positive_equal_to_negative = df[df["1positive"] == df["2negative"]]
    return list(positive_equal_to_negative["company"].values)


def categorize_role(role: str) -> int:
    """
    Categorizes the given role into a specific group.

    The function assigns roles to one of three groups:
    - Other roles (returns 0)
    - IT roles (returns 1)
    - Confidential roles (returns 2)

    Parameters:
    role (str): The role to be categorized.

    Returns:
    int: The group number corresponding to the role.
    """
    if role in IT_ROLES:
        return 1
    elif role == "Funcionário confidencial":
        return 2
    else:
        return 0


def create_role_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a new column 'role_group' to the DataFrame based on the 'employee_role' column.

    The function categorizes each role into one of three groups using the categorize_role function:
    - Other roles (0)
    - IT roles (1)
    - Confidential roles (2)

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'employee_role' column.

    Returns:

    pd.DataFrame: A DataFrame with an additional column 'role_group' with categorized roles.
    """
    df["role_group"] = df["employee_role"].apply(categorize_role)
    return df


def get_ranking_positive_negative_companies(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Calculate the ranking of companies based on the difference between
    positive and negative sentiment counts.

    Parameters:
    - df (DataFrame): Input DataFrame containing sentiment analysis data.

    Returns:
    - Tuple: Two DataFrames representing the top positive and top negative
    companies based on sentiment difference and review counts.
    """
    reviews_count_df = df.groupby(["company"])["review_text"].count()

    reviews_count_df = reviews_count_df.reset_index()
    reviews_count_df.columns = [
        "company",
        "reviews_count",
    ]

    df = pd.merge(
        left=df,
        right=reviews_count_df,
        on="company",
        how="left",
    )

    sentiment_plot_by_company_df = (
        df.groupby(["company", "sentiment_plot"])["review_text"]
        .count()
        .unstack(fill_value=0)
    )

    sentiment_plot_by_company_df = sentiment_plot_by_company_df.reset_index()

    sentiment_plot_by_company_df.columns = [
        "company",
        "1positive",
        "2negative",
        "3neutral",
    ]

    sentiment_plot_by_company_df["sentiment_diff"] = (
        sentiment_plot_by_company_df["1positive"]
        - sentiment_plot_by_company_df["2negative"]
    )

    sentiment_plot_by_company_df = sentiment_plot_by_company_df.sort_values(
        by="sentiment_diff", ascending=False
    ).reset_index()

    sentiment_plot_by_company_df = sentiment_plot_by_company_df.drop(
        labels="index", axis=1
    )

    bad_rating_companies = get_bad_rating_companies(sentiment_plot_by_company_df)

    good_rating_companies = get_good_rating_companies(sentiment_plot_by_company_df)

    reviews_30_plus_df = pd.merge(
        left=df,
        right=sentiment_plot_by_company_df,
        on="company",
        how="left",
    )

    reviews_30_plus_df = reviews_30_plus_df.sort_values(
        by="sentiment_diff", ascending=False
    ).reset_index()

    reviews_30_plus_df = reviews_30_plus_df[
        reviews_30_plus_df["reviews_count"] >= MIN_REVIEWS
    ]

    reviews_30_plus_df["sentiment_count"] = np.select(
        condlist=[
            (reviews_30_plus_df["sentiment_plot"] == 1),  # Positive
            (reviews_30_plus_df["sentiment_plot"] == 2),  # Negative
            (reviews_30_plus_df["sentiment_plot"] == 3),  # Neutral
        ],
        choicelist=[
            reviews_30_plus_df["1positive"],
            reviews_30_plus_df["2negative"],
            reviews_30_plus_df["3neutral"],
        ],
        default=0,
    )

    top_good_bad_companies_by_sentiment_diff = reviews_30_plus_df[
        (
            reviews_30_plus_df["company"].isin(good_rating_companies[:5])
            | reviews_30_plus_df["company"].isin(bad_rating_companies[:5])
        )
    ]

    top_positive_companies_df = top_good_bad_companies_by_sentiment_diff[
        top_good_bad_companies_by_sentiment_diff["sentiment_diff"] > 0
    ]

    top_positive_companies_df = top_positive_companies_df.sort_values(
        by=["sentiment_diff", "reviews_count"],
        ascending=False,
    ).reset_index()

    top_negative_companies_df = top_good_bad_companies_by_sentiment_diff[
        top_good_bad_companies_by_sentiment_diff["sentiment_diff"] < 0
    ]

    top_negative_companies_df = top_negative_companies_df.sort_values(
        by=["sentiment_diff", "reviews_count"],
        ascending=True,
    ).reset_index()

    return top_positive_companies_df, top_negative_companies_df
