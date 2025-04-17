"""Report Config"""

from seaborn import color_palette


class ReportConfig:
    """Configure common used constants."""

    RANDOM_SEED = 103
    COMPANY_NAME_MAX_LENGTH = 100

    SENTIMENT_DICT = {0: "Neutro", 1: "Positivo", 2: "Negativo"}

    PLOT_SENTIMENT_LABELS = ["Positivo", "Negativo", "Neutro"]
    PLOT_SENTIMENT_VALUES = ["1", "2", "0"]

    CHART_TITLE_FONT_SIZE = 14

    NEUTRAL_SENTIMENT_COLOR = color_palette()[0]
    POSITIVE_SENTIMENT_COLOR = color_palette()[2]
    NEGATIVE_SENTIMENT_COLOR = color_palette()[1]

    SENTIMENT_PALETTE = [
        NEUTRAL_SENTIMENT_COLOR,
        POSITIVE_SENTIMENT_COLOR,
        NEGATIVE_SENTIMENT_COLOR,
    ]

    CUSTOM_CSS = """
        <style>
            /* Text Content */
            [data-testid="stMarkdown"] {
                text-align: justify;
            }

            /* Positive Metrics*/
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2),
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(2) p
             {
                color: #2ca02c !important;
            }

            /* Negative Metrics*/
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(3),
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(3) p
            {
                color: #ff7f0e !important;
            }

            /* Neutral Metrics*/
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(4),
            [data-testid="stHorizontalBlock"] > [data-testid="column"]:nth-of-type(4) p
            {
                color: #1f77b4 !important;
            }
        </style>
    """
