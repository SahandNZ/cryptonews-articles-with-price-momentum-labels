import json
from datetime import datetime
from typing import List


class News:
    def __init__(self, date: datetime, title: str, url: str, paragraphs: List[str]):
        self.date: datetime = date
        self.title: str = title
        self.url: str = url
        self.paragraphs: List[str] = paragraphs

    @staticmethod
    def from_json(path: str) -> List:
        with open(path, 'r') as file:
            data = json.load(file)
            news_list = [News(**news) for news in data]

        return news_list

    @staticmethod
    def to_json(news_list: List, path):
        with open(path, 'w') as file:
            data = [news.__dict__ for news in news_list]
            json.dump(data, file, indent=4)
