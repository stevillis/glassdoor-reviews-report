"""Report Config"""

from seaborn import color_palette


class ReportConfig:
    """Configure common used constants."""

    RANDOM_SEED = 103
    SENTIMENT_DICT = {0: "Neutral", 1: "Positive", 2: "Negative"}
    CUSTOM_CSS = """
        <style>
            [data-testid="stMarkdown"] {
                text-align: justify;
            }
        </style>
    """
    CHART_TITLE_FONT_SIZE = 14
    NEUTRAL_SENTIMENT_COLOR = color_palette()[0]
    POSITIVE_SENTIMENT_COLOR = color_palette()[2]
    NEGATIVE_SENTIMENT_COLOR = color_palette()[1]
    SENTIMENT_PALETTE = [
        NEUTRAL_SENTIMENT_COLOR,
        POSITIVE_SENTIMENT_COLOR,
        NEGATIVE_SENTIMENT_COLOR,
    ]
