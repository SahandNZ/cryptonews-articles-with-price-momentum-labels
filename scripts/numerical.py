import argparse
import math
from datetime import datetime

from definitions import *
from src.constant.time_frame import TimeFrame
from src.crawler.numerical.binance import BinanceDataAPI
from src.model.candle import Candle
from src.utils.directory import create_directory_recursively


def crawl(symbol: str, time_frame: int):
    api = BinanceDataAPI()
    to_timestamp = math.floor(datetime.now().timestamp() / time_frame) * time_frame
    from_timestamp = to_timestamp - time_frame * api.maximum_requestable_candles
    candles = api.get_candles(symbol, time_frame, from_timestamp, to_timestamp)

    df = Candle.to_dataframe(candles[:-1])
    data_root = os.path.join(NUMERICAL_CLEAN_DATA_DIR, api.name, symbol)
    data_path = os.path.join(data_root, '{}.csv'.format(TimeFrame.to_str(time_frame)))
    create_directory_recursively(data_root)
    df.to_csv(data_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', action='store', type=str, required=False, default='BTC-USDT')
    parser.add_argument('--timeframe', action='store', type=int, required=False, default=60 * 60 * 24)
    args = parser.parse_args()

    crawl(args.symbol, args.timeframe)
