import math
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from app_messages import AppMessages
from report_config import ReportConfig
from utils import ROLE_GROUPS, load_reviews_df, set_companies_raking_to_session


def employee_role_analysis():
    st.subheader("Avaliações por grupo de funcionários")

    st.markdown(
        """
        Esta análise mostra avaliações por grupo de funcionários, conforme a
        empresa selecionada. **Apenas dados preditos pelo modelo**.
    """
    )

    reviews_df = load_reviews_df()

    company_options = ["Todas"] + sorted(reviews_df["company"].unique().tolist())
    company = st.selectbox(
        label="Empresa",
        options=company_options,
        key="role_group_input",
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
            "role_group",
        ]
    ]

    filtered_df.reset_index(drop=True, inplace=True)

    if len(filtered_df) > 0:
        sentiment_counts = (
            filtered_df.groupby(["role_group", "sentiment_plot"])
            .size()
            .reset_index(name="sentiment_count")
        )

        fig, ax = plt.subplots(1, figsize=(10, 6))

        sns.barplot(
            data=sentiment_counts,
            x="sentiment_count",
            y="role_group",
            hue="sentiment_plot",
            palette=[
                ReportConfig.POSITIVE_SENTIMENT_COLOR,
                ReportConfig.NEGATIVE_SENTIMENT_COLOR,
                ReportConfig.NEUTRAL_SENTIMENT_COLOR,
            ],
            ax=ax,
            width=0.9,
            orient="h",
        )

        for p in ax.patches:
            height = p.get_height()
            if math.isnan(height):
                height = 0.0

            ax.annotate(
                text=f"{p.get_width():.0f}",
                xy=(p.get_width(), (p.get_y() + p.get_height() / 2)),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
                xytext=(10, 0),
                textcoords="offset points",
            )

        ax.set_title(
            "Distribuição de sentimentos por grupo de funcionários",
            fontsize=ReportConfig.CHART_TITLE_FONT_SIZE,
            y=1.1,
        )

        ax.set_xlabel("")
        ax.set_xticks([])

        ax.set_ylabel("")
        ax.set_yticklabels(
            [ROLE_GROUPS[group] for group in sentiment_counts["role_group"].unique()]
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)

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
            labels=ReportConfig.PLOT_SENTIMENT_LABELS,
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
        page_title="Avaliações por grupo de funcionários",
        page_icon=":office_worker:",
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

    employee_role_analysis()
