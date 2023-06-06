import argparse
import os

from src.crawler.textual.cryptonews import CryptoNewsScraper


def crawl(source: str, selenium_driver_path: str):
    os.environ['PATH'] += selenium_driver_path

    if 'cryptonews' == source:
        scraper = CryptoNewsScraper()
        scraper.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', action='store', type=str, required=True)
    parser.add_argument('--selenium-driver-path', action='store', type=str, required=True)
    args = parser.parse_args()

    crawl(source=args.source, selenium_driver_path=args.selenium_driver_path)
