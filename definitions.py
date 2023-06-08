import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
CLEAN_DATA_DIR = os.path.join(DATA_DIR, 'clean')
TEXTUAL_CLEAN_DATA_DIR = os.path.join(CLEAN_DATA_DIR, 'textual')
NUMERICAL_CLEAN_DATA_DIR = os.path.join(CLEAN_DATA_DIR, 'numerical')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')
STATS_DIR = os.path.join(ROOT_DIR, 'stats')
