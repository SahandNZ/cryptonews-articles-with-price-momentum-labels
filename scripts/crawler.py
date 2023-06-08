import argparse
import math
import os
from datetime import datetime

from definitions import NUMERICAL_CLEAN_DATA_DIR
from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame
from src.crawler.numerical.binance import BinanceDataAPI
from src.crawler.textual.cryptonews import CryptoNewsScraper
from src.model.candle import Candle
from src.utils.directory import create_directory_recursively


def crawl_numerical(symbol: Symbol, time_frame: TimeFrame):
    api = BinanceDataAPI()
    to_timestamp = math.floor(datetime.now().timestamp() / time_frame) * time_frame
    from_timestamp = to_timestamp - time_frame * api.maximum_requestable_candles
    candles = api.get_candles(symbol, time_frame, from_timestamp, to_timestamp)

    df = Candle.to_dataframe(candles[:-1])
    data_root = os.path.join(NUMERICAL_CLEAN_DATA_DIR, api.name, symbol)
    data_path = os.path.join(data_root, '{}.csv'.format(TimeFrame.to_str(time_frame)))
    create_directory_recursively(data_root)
    df.to_csv(data_path, index=False)


def crawl_textual(source: str, selenium_driver_path: str):
    os.environ['PATH'] += selenium_driver_path

    if 'cryptonews' == source:
        scraper = CryptoNewsScraper()
        scraper.run()


def crawl(symbol: Symbol, time_frame: TimeFrame, textual_source: str, selenium_driver_path: str):
    crawl_numerical(symbol=symbol, time_frame=time_frame)
    crawl_textual(source=textual_source, selenium_driver_path=selenium_driver_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numerical-source', action='store', type=str, required=False, default='binance')
    parser.add_argument('--symbol', action='store', type=str, required=False, default=Symbol.BTCUSDT)
    parser.add_argument('--timeframe', action='store', type=int, required=False, default=TimeFrame.DAY1)
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--selenium-driver-path', action='store', type=str, required=False, default='selenium')
    args = parser.parse_args()

    crawl(args.symbol, args.time_frame, args.textual_source, args.selenium_driver_path)
