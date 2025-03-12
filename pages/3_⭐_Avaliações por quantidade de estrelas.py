import math
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import load_reviews_df, set_companies_raking_to_session


def rating_star_analysis():
    st.subheader("Avaliações por quantidade de estrelas")

    st.markdown(
        """
        Esta análise mostra a relação entre as avaliações e o número de
        estrelas atribuídas, conforme a empresa selecionada. **Apenas dados
        preditos pelo modelo**.
    """
    )

    reviews_df = load_reviews_df()

    company_options = ["Todas"] + sorted(reviews_df["company"].unique().tolist())
    company = st.selectbox(
        label="Empresa",
        options=company_options,
        key="rating_star_company_input2",
        index=0,
    )

    filtered_df = reviews_df[(reviews_df["company"] == company) | (company == "Todas")][
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
                fontsize=11,
                color="black",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax.set_title(
            "Avaliações por quantidade de estrelas",
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
            labels=["Positivo", "Negativo", "Neutro"],
            bbox_to_anchor=(0.5, 1.1),
            loc="upper center",
            edgecolor="1",
            ncols=3,
        )

        st.pyplot(fig)

        st.write("Avaliações filtradas")
        filtered_print_df = filtered_df.drop(labels="sentiment_plot", axis=1)
        st.dataframe(filtered_print_df)
    else:
        st.error(
            AppMessages.ERROR_EMPTY_DATAFRAME,
            icon="🚨",
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", "use_inf_as_na")

    st.set_page_config(
        page_title="Avaliações por quantidade de estrelas",
        page_icon=":star:",
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
    set_companies_raking_to_session(reviews_df)

    rating_star_analysis()
