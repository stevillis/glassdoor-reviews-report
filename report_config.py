"""Report Config"""


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
