import argparse
import os

from definitions import TEXTUAL_RAW_DATA_DIR, TEXTUAL_CLEAN_DATA_DIR
from src.cleaner.news_cleaner import NewsCleaner
from src.model.news import News


def clean(source: str):
    raw_textual_data_path = os.path.join(TEXTUAL_RAW_DATA_DIR, source, 'news.json')
    clean_textual_data_path = os.path.join(TEXTUAL_CLEAN_DATA_DIR, '{}.csv'.format(source))

    raw_data = News.from_json(raw_textual_data_path)
    news_cleaner = NewsCleaner(raw_data=raw_data)
    news_cleaner.to_csv(clean_textual_data_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', type=str, required=True)
    args = parser.parse_args()

    clean(args.source)
