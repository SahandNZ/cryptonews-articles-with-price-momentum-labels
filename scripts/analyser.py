import argparse
import os

import pandas as pd

from definitions import DATASET_DIR
from src.analyser.dataset import DatasetAnalyser


def analyse(textual_source: str, labeling_method: str):
    dataset_csv_path = os.path.join(DATASET_DIR, textual_source, labeling_method, 'dataset.csv')
    df = pd.read_csv(dataset_csv_path)
    df = df[['text', 'label']]
    df['label'] = df.label + 1

    textual_analysis = DatasetAnalyser(df)
    textual_analysis.save()
    print(textual_analysis)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--textual-source', action='store', type=str, required=False, default='cryptonews')
    parser.add_argument('--labeling-method', action='store', type=str, required=False, default='color')
    args = parser.parse_args()

    analyse(args.textual_source, args.labeling_method)
