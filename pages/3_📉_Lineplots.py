import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from report_config import ReportConfig

if __name__ == "__main__":
    st.set_page_config(
        page_title="Lineplots",
        page_icon=":chart_with_downwards_trend:",
    )

    st.markdown(
        ReportConfig.CUSTOM_CSS,
        unsafe_allow_html=True,
    )

    st.title(
        "Desvendando emoções nas avaliações do Glassdoor de empresas de Tecnologia de Cuiabá"
    )

    st.header("Sentimento de Avaliações por Empresa ao longo do tempo")

    if "reviews_df" not in st.session_state:
        reviews_df = pd.read_csv("./glassdoor_reviews_predicted.csv")
        reviews_df["sentiment_label"] = reviews_df["predicted_sentiment"].map(
            ReportConfig.SENTIMENT_DICT
        )
        st.session_state["reviews_df"] = reviews_df
    else:
        reviews_df = st.session_state.get("reviews_df")

    company = st.selectbox(
        label="Empresa",
        options=reviews_df["company"].unique().tolist(),
        key="reviews_along_time_company_input",
    )

    filtered_df = reviews_df[reviews_df["company"] == company][
        [
            "company",
            "review_date",
            "predicted_sentiment",
        ]
    ]
    filtered_df.reset_index(drop=True, inplace=True)

    filtered_df["review_date"] = pd.to_datetime(filtered_df["review_date"])
    filtered_df["year"] = filtered_df["review_date"].dt.year

    sentiment_counts = (
        filtered_df.groupby(["year", "predicted_sentiment"])
        .size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(1, figsize=(12, 6))
    sns.lineplot(
        data=sentiment_counts,
        x="year",
        y="count",
        hue="predicted_sentiment",
        marker="o",
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
