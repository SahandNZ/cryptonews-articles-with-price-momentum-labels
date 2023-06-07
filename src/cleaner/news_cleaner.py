from typing import List

from tqdm import tqdm

from src.cleaner.cleaner import Cleaner
from src.model.news import News


class NewsCleaner(Cleaner):
    def __init__(self, raw_data: List):
        super().__init__(raw_data, columns=['datetime', 'title', 'text', 'url'])

    def clean(self, raw_data: List[News]) -> List[List]:
        clean_data = []

        bar = tqdm(raw_data)
        bar.set_description("Cleaning News Data")
        for raw_news in bar:
            clean_datetime = self.clean_datetime(raw_news.date)
            clean_title = self.clean_string(raw_news.title)
            clean_url = raw_news.url
            for paragraph in raw_news.paragraphs:
                clean_paragraph = self.clean_string(paragraph)
                clean_data.append([clean_datetime, clean_title, clean_paragraph, clean_url])

        return clean_data
