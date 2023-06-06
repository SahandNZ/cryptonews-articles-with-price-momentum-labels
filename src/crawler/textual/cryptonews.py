from typing import List

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By

from src.crawler.textual.model.news import News
from src.crawler.textual.model.url import URL
from src.crawler.textual.scraper import Scraper


class CryptoNewsScraper(Scraper):
    def __init__(self, threads: int, use_cache: bool = True):
        super().__init__(name='cryptonews', base_url='https://cryptonews.com/news/bitcoin-news/', threads=threads,
                         use_cache=use_cache)

    def collect_urls(self) -> List[URL]:
        driver = webdriver.Chrome()
        driver.maximize_window()
        driver.get(self.base_url)

        print('Collecting urls...')

        urls = []
        while True:
            try:
                elements = driver.find_elements(By.CLASS_NAME, 'mb-30')
                for element in elements:
                    if 'div' == element.tag_name:
                        a_tag = element.find_element(By.CLASS_NAME, 'article__title')
                        title = a_tag.text
                        url = a_tag.get_attribute('href')
                        urls.append(URL(title=title, url=url))

                load_more_button = driver.find_element(By.ID, 'load_more')
                scroll_value = +load_more_button.location['y'] + load_more_button.size['height']
                driver.execute_script("window.scrollBy(0, arguments[0]);", -scroll_value)
                load_more_button.click()
                print('{} urls were collected. On load more button clicked...'.format(len(urls)))

            except Exception as e:
                print(e)
                driver.quit()
                print('Collecting urls Done.')
                break

        return urls

    def collect_news_with_selenium(self, driver: webdriver, url: str):
        driver.get(url)

        datetime = driver.find_element(By.TAG_NAME, 'time').get_attribute('datetime')
        content = driver.find_element(By.CLASS_NAME, 'article-single__content')
        title = content.find_element(By.TAG_NAME, 'h1').text
        paragraphs = [p.text.replace('\n', ' ') for p in content.find_elements(By.TAG_NAME, 'p')]
        news = News(date=datetime, title=title, paragraphs=paragraphs, url=url)

        return news

    def collect_news(self, url: str) -> News:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')

        datetime = soup.time.get('datetime')
        content = soup.find(class_='article-single__content')
        title = content.find('h1').text
        paragraphs = [p.text for p in content.find_all('p')]
        news = News(date=datetime, title=title, paragraphs=paragraphs, url=url)

        return news
