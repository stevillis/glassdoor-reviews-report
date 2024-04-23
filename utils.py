from report_config import ReportConfig


def get_sentiment_key_from_value(value):
    key_list = list(ReportConfig.SENTIMENT_DICT.keys())
    val_list = list(ReportConfig.SENTIMENT_DICT.values())

    position = val_list.index(value)
    return key_list[position]
