import os
from abc import abstractmethod, ABC
from typing import List

from tqdm import tqdm

from definitions import TEXTUAL_RAW_DATA_DIR
from src.model.news import News
from src.model.url import URL
from src.utils.directory import create_directory_recursively


class Scraper(ABC):
    def __init__(self, name: str, base_url: str, use_cache: bool):
        self.name: str = name
        self.base_url: str = base_url
        self.use_cache: bool = use_cache

        self.data_root = os.path.join(TEXTUAL_RAW_DATA_DIR, self.name)
        self.urls_csv_path = os.path.join(self.data_root, 'url.csv')
        self.news_json_path = os.path.join(self.data_root, 'news.json')
        create_directory_recursively(self.data_root)

        self.urls: List[URL] = None
        self.news_list: List[News] = None

    def run(self):
        self.pre_collect_urls()
        self.pre_collect_news()

    def pre_collect_urls(self):
        if self.use_cache and os.path.exists(self.urls_csv_path):
            self.urls = URL.from_csv(self.urls_csv_path)
        else:
            self.urls = self.collect_urls()
            URL.to_csv(self.urls, self.urls_csv_path)

    def pre_collect_news(self):
        self.news_list: List[News] = []
        if self.use_cache and os.path.exists(self.news_json_path):
            self.news_list = News.from_json(self.news_json_path)
        collected_urls = [news.url for news in self.news_list]

        bar = tqdm(self.urls)
        bar.set_description('Scrape news pages')
        for item in bar:
            if item.url not in collected_urls:
                try:
                    news = self.collect_news(item.url)
                    self.news_list.append(news)
                    News.to_json(self.news_list, self.news_json_path)

                except Exception as e:
                    print(e)

    @abstractmethod
    def collect_urls(self) -> List[URL]:
        raise NotImplementedError()

    @abstractmethod
    def collect_news(self, url: str) -> News:
        raise NotImplementedError()
