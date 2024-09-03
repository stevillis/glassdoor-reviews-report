import pandas as pd

from report_config import ReportConfig


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
        by="1positive", ascending=False
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
        by="2negative", ascending=False
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
