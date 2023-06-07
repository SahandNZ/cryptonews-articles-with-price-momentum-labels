import argparse

from scripts.cleaner import clean as clean_textual
from scripts.labler import add_label
from scripts.numerical import crawl as crawl_numerical
from scripts.textual import crawl as crawl_textual
from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame


def crawl(symbol: Symbol, time_frame: TimeFrame, textual_source: str, selenium_driver_path: str):
    crawl_numerical(symbol=symbol, time_frame=time_frame)
    crawl_textual(source=textual_source, selenium_driver_path=selenium_driver_path)


def clean(source: str):
    clean_textual(source=source)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--crawl', action='store_true', required=False)
    parser.add_argument('--clean', action='store_true', required=False)
    parser.add_argument('--add-label', action='store_true', required=False)
    parser.add_argument('--create-stats', action='store_true', required=False)

    parser.add_argument('--selenium-driver-path', action='store', type=str, required=False, default='selenium')
    parser.add_argument('--numerical-source', action='store', type=str, required=False, default='binance')
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--symbol', action='store', type=str, required=False, default=Symbol.BTCUSDT)
    parser.add_argument('--time-frame', action='store', type=int, required=False, default=TimeFrame.DAY1)
    parser.add_argument('--look-ahead', action='store', type=int, required=False, default=7)
    parser.add_argument('--labeling-method', action='store', type=str, required=False, default='roc')
    args = parser.parse_args()

    if args.crawl:
        crawl(args.symbol, args.time_frame, args.textual_source, args.selenium_driver_path)

    if args.clean:
        clean(args.textual_source)

    if args.add_label:
        add_label(args.labeling_method, args.numerical_source, args.symbol, args.time_frame, args.look_ahead,
                  args.textual_source)
