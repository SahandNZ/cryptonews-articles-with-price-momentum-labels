from typing import List

from tqdm import tqdm

from src.cleaner.cleaner import Cleaner
from src.model.news import News


class NewsCleaner(Cleaner):
    def __init__(self, raw_data: List):
        super().__init__(raw_data)

    def clean(self, raw_data: List[News]) -> List[News]:
        clean_news_list = []

        bar = tqdm(raw_data)
        bar.set_description("Cleaning News Data")
        for raw_news in bar:
            clean_datetime = self.clean_datetime(raw_news.date)
            clean_title = self.clean_string(raw_news.title)
            clean_url = raw_news.url
            clean_paragraphs = [self.clean_string(p) for p in raw_news.paragraphs]
            clean_news = News(clean_datetime, clean_title, clean_url, clean_paragraphs)
            clean_news_list.append(clean_news)

        return clean_news_list

    def to_json(self, path: str):
        clean_news = self.clean(self.raw_data)
        News.to_json(clean_news, path)
