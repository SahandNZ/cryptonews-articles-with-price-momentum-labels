import argparse
import os

from src.crawler.textual.cryptonews import CryptoNewsScraper

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chrome-driver-path', action='store', type=str, required=True)
    parser.add_argument('--source', action='store', type=str, required=True)
    parser.add_argument('--threads', action='store', type=str, required=False, default=4)
    args = parser.parse_args()

    os.environ['PATH'] += args.chrome_driver_path

    if 'cryptonews' == args.source:
        scraper = CryptoNewsScraper(threads=args.threads)
        scraper.run()
