import argparse

from scripts.analyser import analyse
from scripts.cleaner import clean
from scripts.crawler import crawl
from scripts.labler import add_label
from src.constant.symbol import Symbol
from src.constant.time_frame import TimeFrame

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--crawl', action='store_true', required=False)
    parser.add_argument('--clean', action='store_true', required=False)
    parser.add_argument('--add-label', action='store_true', required=False)
    parser.add_argument('--analyse', action='store_true', required=False)

    parser.add_argument('--numerical-source', action='store', type=str, required=False, default='binance')
    parser.add_argument('--symbol', action='store', type=str, required=False, default=Symbol.BTCUSDT)
    parser.add_argument('--time-frame', action='store', type=int, required=False, default=TimeFrame.DAY1)
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--selenium-driver-path', action='store', type=str, required=False, default='selenium')
    parser.add_argument('--look-ahead', action='store', type=int, required=False, default=7)
    parser.add_argument('--labeling-method', action='store', type=str, required=False, default='color')
    args = parser.parse_args()

    if args.crawl:
        crawl(args.symbol, args.time_frame, args.textual_source, args.selenium_driver_path)

    if args.clean:
        clean(args.textual_source)

    if args.add_label:
        add_label(args.labeling_method, args.numerical_source, args.symbol, args.time_frame, args.look_ahead,
                  args.textual_source)

    if args.analyse:
        analyse(args.textual_source, args.labeling_method)
