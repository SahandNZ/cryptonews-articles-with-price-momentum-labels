import argparse

from scripts.numerical import crawl as crawl_numerical
from scripts.textual import crawl as crawl_textual
from src.crawler.numerical.constant.symbol import Symbol
from src.crawler.numerical.constant.time_frame import TimeFrame


def crawl():
    crawl_numerical(symbol=Symbol.BTCUSDT, time_frame=TimeFrame.DAY1)
    crawl_textual(source='cryptonews', selenium_driver_path='selenium')


def clean():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl', action='store_true', required=False)
    parser.add_argument('--clean', action='store_true', required=False)
    args = parser.parse_args()

    if args.crawl:
        crawl()

    if args.clean:
        clean()
