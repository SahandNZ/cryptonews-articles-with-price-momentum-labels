from typing import List

import pandas as pd


class URL:
    def __init__(self, title: str, url: str):
        self.title: str = title
        self.url: str = url

    @staticmethod
    def from_list(values: List):
        return URL(title=values[0], url=values[1])

    def to_list(self) -> List:
        return [self.title, self.url]

    @staticmethod
    def to_dataframe(urls: List):
        columns = ['title', 'url']
        values = [url.to_list() for url in urls]
        df = pd.DataFrame(values, columns=columns)
        return df

    @staticmethod
    def to_csv(urls: List, path: str):
        df = URL.to_dataframe(urls)
        df.to_csv(path, index=False)

    @staticmethod
    def from_csv(path: str) -> List:
        df = pd.read_csv(path)
        urls = [URL.from_list(df.iloc[index].to_list()) for index in range(len(df))]
        return urls
