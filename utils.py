from string import punctuation
from typing import List

import nltk
import numpy as np
import pandas as pd

from report_config import ReportConfig

nltk.download("stopwords")

MIN_REVIEWS = 42 / 2  # (reviews median) / 2

STOPWORDS = [word.lower() for word in nltk.corpus.stopwords.words("portuguese")]
STOPWORDS.extend(
    [
        # "empresa",
    ]
)

# Translation table for replacing any special character from a word.
TRANSLATION_TABLE_SPECIAL_CHARACTERS = str.maketrans("", "", punctuation)


def get_sentiment_key_from_value(value):
    key_list = list(ReportConfig.SENTIMENT_DICT.keys())
    val_list = list(ReportConfig.SENTIMENT_DICT.values())

    position = val_list.index(value)
    return key_list[position]


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


def create_predicted_sentiment_plot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a new column 'predicted_sentiment_plot' in the DataFrame based on the 'predicted_sentiment' column.

    The function converts sentiments to be in the following order: Positive, Negative, Neutral.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the 'predicted_sentiment' column.

    Returns:
    pd.DataFrame: A DataFrame with an additional column 'predicted_sentiment_plot' with converted sentiments.
    """

    sentiment_mapping = {0: 3, 1: 1, 2: 2}
    df["predicted_sentiment_plot"] = df["predicted_sentiment"].map(sentiment_mapping)

    return df


def get_ranking_positive_negative_companies(df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Calculate the ranking of companies based on the difference between positive and negative sentiment counts.

    Parameters:
    - df (DataFrame): Input DataFrame containing sentiment analysis data.

    Returns:
    - Tuple: Two DataFrames representing the top positive and top negative companies based on sentiment difference and review counts.
    """
    df["predicted_sentiment_plot"] = np.select(
        condlist=[
            (df["predicted_sentiment"] == 0),
            (df["predicted_sentiment"] == 1),
            (df["predicted_sentiment"] == 2),
        ],
        choicelist=[3, 1, 2],
    )

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

    predicted_sentiment_plot_by_company_df = (
        df.groupby(["company", "predicted_sentiment_plot"])["review_text"]
        .count()
        .unstack(fill_value=0)
    )

    predicted_sentiment_plot_by_company_df = (
        predicted_sentiment_plot_by_company_df.reset_index()
    )

    predicted_sentiment_plot_by_company_df.columns = [
        "company",
        "1positive",
        "2negative",
        "3neutral",
    ]

    predicted_sentiment_plot_by_company_df["sentiment_diff"] = (
        predicted_sentiment_plot_by_company_df["1positive"]
        - predicted_sentiment_plot_by_company_df["2negative"]
    )

    predicted_sentiment_plot_by_company_df = (
        predicted_sentiment_plot_by_company_df.sort_values(
            by="sentiment_diff", ascending=False
        ).reset_index()
    )

    predicted_sentiment_plot_by_company_df = (
        predicted_sentiment_plot_by_company_df.drop(labels="index", axis=1)
    )

    bad_rating_companies = get_bad_rating_companies(
        predicted_sentiment_plot_by_company_df
    )

    good_rating_companies = get_good_rating_companies(
        predicted_sentiment_plot_by_company_df
    )

    reviews_30_plus_df = pd.merge(
        left=df,
        right=predicted_sentiment_plot_by_company_df,
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
            (reviews_30_plus_df["predicted_sentiment_plot"] == 1),  # Positive
            (reviews_30_plus_df["predicted_sentiment_plot"] == 2),  # Negative
            (reviews_30_plus_df["predicted_sentiment_plot"] == 3),  # Neutral
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
